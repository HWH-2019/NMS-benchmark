#include <torch/extension.h>
void nms_cuda_compute(int* keep_out, int *num_out, float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh);


at::Tensor nms_cuda_compute(const at::Tensor boxes, float nms_overlap_thresh);