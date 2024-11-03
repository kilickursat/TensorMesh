from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='spsolve_cpp',
    ext_modules=[
        CppExtension('spsolve_cpp', ['spsolve.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
