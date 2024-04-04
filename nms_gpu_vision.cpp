#include "nms_gpu.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>

namespace py = pybind11;

int nms_gpu(py::array_t<int> keep_out, py::array_t<float> boxes_host){
    py::buffer_info boxes_host_buf = boxes_host.request();
    int boxes_num = boxes_host_buf.shape[0];
    int boxes_dim = boxes_host_buf.shape[1];

    py::buffer_info keep_out_buf = keep_out.request();

    float nms_overlap_thresh = 0.7;
    int num_out = 0;

    nms_cuda_compute((int*)keep_out_buf.ptr, &num_out, (float*)boxes_host_buf.ptr, boxes_num, boxes_dim, nms_overlap_thresh);

    return num_out;
}

at::Tensor nms_gpu_tensor(const at::Tensor& dets,
               const at::Tensor& scores,
               const float threshold){
    if(dets.type().is_cuda()){
#ifdef WITH_CUDA
    // TODO raise error if not compiled with CUDA
    if (dets.numel() == 0)
      return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
    auto b = at::cat({dets, scores.unsqueeze(1)}, 1);
    return nms_cuda_compute(b, threshold);
#else
    AT_ERROR("Not compiled with GPU support");
#endif        
    }else{
        AT_ERROR("Please use CPU version");
        return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "nms_gpu";
    m.def("nms_gpu", &nms_gpu, "A function with nms_gpu");
    m.def("nms_gpu_tensor", &nms_gpu_tensor, "A function with nms_gpu_tensor");
}