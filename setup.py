import torch
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME

def get_extensions():
    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources = ["nms_gpu_vision.cpp", "nms_gpu.cu"]
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    ext_modules = [
        extension(
            "nms_gpu",
            sources,
            extra_compile_args=extra_compile_args,
            define_macros=define_macros,
        )
    ]
    return ext_modules

setup(
    name='nms_gpu',
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext': BuildExtension
    })