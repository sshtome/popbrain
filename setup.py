import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

build_ext = os.environ.get('BUILD_CUDA_EXT', '0') == '1'

ext_modules = []
if build_ext:
    ext_modules = [
        CUDAExtension(
            name='popbrain.snn_cuda',
            sources=['popbrain/csrc/snn_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ]

setup(
    name='popbrain',
    version='1.0.0',
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
