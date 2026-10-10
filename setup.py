from setuptools import setup, find_packages

setup(
    name='popbrain',
    version='1.0.0',
    description='A highly optimized Spiking Neural Network (SNN) engine using PyTorch Sparse CSR and custom CUDA kernels.',
    author='PopBrain Contributors',
    packages=find_packages(),
    include_package_data=True,
    package_data={'popbrain': ['csrc/*']},
    install_requires=[
        'torch>=2.0.0',
        'numpy'
    ],
    python_requires='>=3.8',
)
