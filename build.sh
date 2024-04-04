nvcc --shared -Xcompiler -fPIC nms_gpu_vision.cpp nms_gpu.cu -o nms_gpu.so  \
-I/home/huangwenhuan/.conda/envs/ETT/lib/python3.8/site-packages/torch/include \
-I/home/huangwenhuan/.conda/envs/ETT/lib/python3.8/site-packages/torch/include/torch/csrc/api/include \
-I/home/huangwenhuan/.conda/envs/ETT/include/python3.8


# nvcc -DWITH_CUDA \
# -I/home/huangwenhuan/.conda/envs/ETT/lib/python3.8/site-packages/torch/include \
# -I/home/huangwenhuan/.conda/envs/ETT/lib/python3.8/site-packages/torch/include/torch/csrc/api/include \
# -I/home/huangwenhuan/.conda/envs/ETT/lib/python3.8/site-packages/torch/include/TH \
# -I/home/huangwenhuan/.conda/envs/ETT/lib/python3.8/site-packages/torch/include/THC \
# -I/usr/local/cuda/include \
# -I/home/huangwenhuan/.conda/envs/ETT/include/python3.8 \
# --shared -Xcompiler -fPIC nms_gpu.cu nms_gpu_vision.cpp -o nms_gpu.so \
# -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options '-fPIC' -DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -DTORCH_API_INCLUDE_EXTENSION_H \
# -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" \
# -DTORCH_EXTENSION_NAME=nms_gpu \
# -gencode=arch=compute_86,code=compute_86 -gencode=arch=compute_86,code=sm_86 \
# -std=c++14