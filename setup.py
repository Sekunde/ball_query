from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ball_query',
    ext_modules=[
        CUDAExtension('ball_query', [
            'ball_query.cpp',
            'ball_query_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
