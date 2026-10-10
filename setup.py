from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='popbrain',
    version='1.0.0',
    description='A highly optimized Spiking Neural Network (SNN) engine using PyTorch Sparse CSR and custom CUDA kernels.',
    author='PopBrain Contributors',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='popbrain.snn_cuda',
            sources=[
                os.path.join('popbrain', 'csrc', 'snn_bind.cpp'),
                os.path.join('popbrain', 'csrc', 'snn_kernel.cu'),
            ],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch>=2.0.0',
        'numpy'
    ],
    python_requires='>=3.8',
)
