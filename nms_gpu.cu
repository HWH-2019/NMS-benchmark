#include<iostream>
#include<vector>
#include<random>
#include<ATen/ATen.h>
#include<ATen/ceil_div.h>
#include<ATen/cuda/CUDAContext.h> 
#include<ATen/cuda/ThrustAllocator.h>


#include"nms_gpu.h"


#define CUDA_WARN(XXX) \
  do { if(XXX != cudaSuccess) std::cout<< "CUDA Error: " << \
  cudaGetErrorString(XXX) << ", at line " << __LINE__ << std::endl; \
  cudaDeviceSynchronize(); } while (0)

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
static const int threadsPerBlock = sizeof(unsigned long long) * 8;

__device__ inline float devIoU(const float* const a, const float* const b){
    float left = max(a[0], b[0]), right = min(a[2], b[2]);
    float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
    float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
    float interS = width * height;
    float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
    float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
    return interS / (Sa + Sb - interS);
}

__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float* dev_boxes, unsigned long long* dev_mask, const int boxes_dim=4){
    
    // blockIdx 是列主序的，因此 blockIdx.x 代表的是列数，blockIdx.y 代表的是行数
    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;

    // 计算当前block中使用到的线程数，最多是 64，不足 64 取余数
    const int row_size = 
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
    const int col_size = 
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

    // 采用动态内存分配，分配大小为 block_size * boxes_dim
    extern __shared__ float block_boxes[];

    // 将当前 block 中需要处理的 box 复制到共享内存中
    if(threadIdx.x < col_size){
        for(int i = 0; i < boxes_dim; ++i){
            block_boxes[threadIdx.x * boxes_dim + i] = 
                dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * boxes_dim + i];
        }
        
        // block_boxes[threadIdx.x * boxes_dim + 0] = 
        //     dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * boxes_dim + 0];
        // block_boxes[threadIdx.x * boxes_dim + 1] = 
        //     dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * boxes_dim + 1];
        // block_boxes[threadIdx.x * boxes_dim + 2] = 
        //     dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * boxes_dim + 2];
        // block_boxes[threadIdx.x * boxes_dim + 3] = 
        //     dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * boxes_dim + 3];
    }

    __syncthreads();

    if(threadIdx.x < row_size){
        const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
        const float *cur_box = dev_boxes + cur_box_idx * boxes_dim;
        int i = 0;
        unsigned long long t = 0;
        int start = 0;
        if(row_start == col_start){
            start = threadIdx.x + 1;
        }
        for(i = start; i < col_size; ++i){
            if(devIoU(cur_box, block_boxes + i * boxes_dim) > nms_overlap_thresh){
                t |= 1ULL << i;
            }
        }
        const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
        dev_mask[cur_box_idx * col_blocks + col_start] = t;
    }

}

void nms_cuda_compute(int* keep_out, int* num_out, float* boxes_host, int boxes_num, int boxes_dim, float nms_overlap_thresh){
    float* boxes_dev = NULL;
    unsigned long long *mask_dev = NULL;

    const int col_blocks = DIVUP(boxes_num, threadsPerBlock);

    // 分配显存空间和 boxes_host 的大小一致
    CUDA_CHECK(cudaMalloc(&boxes_dev, boxes_num * boxes_dim * sizeof(float)));
    // 将 boxes_host 复制到 boxes_dev, 即 cpu -> gpu
    CUDA_CHECK(cudaMemcpy(boxes_dev, boxes_host, boxes_num * boxes_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // 分配显存空间用于记录 boxes_dev 中每个 box 与其他 box 的重叠情况
    // 原本大小应该是 boxes_num * boxes_num * sizeof(unsigned)
    // 使用 64 位的数重新存储，每位表示一个 box 与其他 box 的重叠情况 0 表示不重叠，1 表示重叠
    // 则需要 boxes_num * （boxes_num / 64） * sizeof(unsigned long long)
    // 即 boxes_num * col_blocks * sizeof(unsigned long long)
    CUDA_CHECK(cudaMalloc(&mask_dev, boxes_num * col_blocks * sizeof(unsigned long long)));
    
    // 设置 blocks 和 threads 数量
    dim3 blocks(DIVUP(boxes_num, threadsPerBlock), DIVUP(boxes_num, threadsPerBlock));
    
    dim3 threads(threadsPerBlock);

    // 调用核函数
    nms_kernel<<<blocks, threads, threadsPerBlock * boxes_dim>>>(boxes_num, nms_overlap_thresh, boxes_dev, mask_dev, boxes_dim);

    std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
    CUDA_CHECK(cudaMemcpy(&mask_host[0], mask_dev, sizeof(unsigned long long) * boxes_num * col_blocks, cudaMemcpyDeviceToHost));

    std::vector<unsigned long long> remv(col_blocks);
    memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);
    
    int num_to_keep = 0;
    for(int i=0; i < boxes_num; ++i){
        int nblock = i / threadsPerBlock;
        int inblock = i % threadsPerBlock;
        
        if(!(remv[nblock] & (1ULL << inblock))){
            keep_out[num_to_keep++] = i;
            unsigned long long *p = &mask_host[0] + i * col_blocks;
            for(int j = nblock; j < col_blocks; ++j){
                remv[j] |= p[j];
            }
        }
    }

    *num_out = num_to_keep;
    CUDA_CHECK(cudaFree(boxes_dev));
    CUDA_CHECK(cudaFree(mask_dev));
}

at::Tensor nms_cuda_compute(const at::Tensor boxes, float nms_overlap_thresh){
    using scalar_t = float;
    AT_ASSERTM(boxes.type().is_cuda(), "boxes must be a CUDA tensor");
    auto scores = boxes.select(1, 4);
    auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));
    auto boxes_sorted = boxes.index_select(0, order_t);

    int boxes_num = boxes.size(0);
    int boxes_dim = boxes.size(1);

    const int col_blocks = at::ceil_div(boxes_num, threadsPerBlock);
    
    scalar_t* boxes_dev = boxes_sorted.data<scalar_t>();

    unsigned long long* mask_dev = NULL;

    mask_dev = (unsigned long long*) c10::cuda::CUDACachingAllocator::raw_alloc(boxes_num * col_blocks * sizeof(unsigned long long));

    dim3 blocks(at::ceil_div(boxes_num, threadsPerBlock),
               at::ceil_div(boxes_num, threadsPerBlock));
    
    dim3 threads(threadsPerBlock);

    nms_kernel<<<blocks, threads, threadsPerBlock * boxes_dim>>>(boxes_num, nms_overlap_thresh, boxes_dev, mask_dev, boxes_dim);

    std::vector<unsigned long long> mask_host(boxes_num * col_blocks);

    C10_CUDA_CHECK(cudaMemcpy(&mask_host[0], mask_dev, sizeof(unsigned long long) * boxes_num * col_blocks, cudaMemcpyDeviceToHost));

    std::vector<unsigned long long> remv(col_blocks);
    memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

    at::Tensor keep = at::empty({boxes_num}, boxes.options().dtype(at::kLong).device(at::kCPU));
    int64_t* keep_out = keep.data<int64_t>();

    int num_to_keep = 0;
    for(int i=0;i<boxes_num;++i){
        int nblock = i / threadsPerBlock;
        int inblock = i % threadsPerBlock;
        if(!(remv[nblock] & (1ULL << inblock))){
            keep_out[num_to_keep++] = i;
            unsigned long long *p = &mask_host[0] + i * col_blocks;
            for(int j = nblock; j < col_blocks; ++j){
                remv[j] |= p[j];
            }
        }
    }

    c10::cuda::CUDACachingAllocator::raw_delete(mask_dev);
    // TODO improve this part
    return std::get<0>(order_t.index({
                       keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep).to(
                         order_t.device(), keep.scalar_type())
                     }).sort(0, false));
}