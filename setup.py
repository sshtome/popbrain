import os
os.environ['DISTUTILS_USE_SDK'] = '1'
if not os.environ.get('TORCH_CUDA_ARCH_LIST'):
    os.environ['TORCH_CUDA_ARCH_LIST'] = '7.5 8.0 8.6 8.9+PTX'
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

build_ext = os.environ.get('BUILD_CUDA_EXT', '0') == '1'

import torch.utils.cpp_extension as cpp
cpp._check_cuda_version = lambda *args, **kwargs: None

ext_modules = []
if build_ext:
    ext_modules = [
        CUDAExtension(
            name='popbrain.snn_cuda',
            sources=['popbrain/csrc/snn_kernel.cu'],
            extra_compile_args={
                'cxx': (['-O3', '/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH'] if os.name == 'nt' else ['-O3']),
                'nvcc': ['-O3', '--use_fast_math'] + (['-allow-unsupported-compiler', '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH', '-Xcompiler', '/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH', '-Xcompiler', '/Zc:preprocessor', '-Xcompiler', '/Zm500', '-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING'] if os.name == 'nt' else [])
            }
        )
    ]

setup(
    name='popbrain',
    version='1.2.0',
    description='A highly optimized Spiking Neural Network (SNN) engine using PyTorch Sparse CSR and custom CUDA kernels.',
    author='PopBrain Contributors',
    packages=find_packages(),
    include_package_data=True,
    package_data={'popbrain': ['csrc/*']},
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    } if build_ext else {},
    install_requires=[
        'torch>=2.0.0',
        'numpy'
    ],
    python_requires='>=3.8',
)
