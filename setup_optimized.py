import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

extra_compile_args = []
if os.name == 'nt':
    # MSVC Flags: AVX2, AVX-512, Fast Floating Point, Intel64 optimization, Max Optimization, OpenMP
    extra_compile_args = ['/arch:AVX512', '/fp:fast', '/favor:INTEL64', '/Ox', '/DNDEBUG', '/openmp', '/Oi', '/Ot', '/Qpar']
else:
    # GCC/Clang Flags: AVX-512, OpenMP
    extra_compile_args = ['-mavx512f', '-mavx512dq', '-mavx512f', '-mavx512bw', '-mavx512vl', '-mavx512vnni', '-mavx2', '-mfma', '-O3', '-march=native', '-ffast-math', '-fopenmp']

setup(
    name='cpp_loader',
    ext_modules=[
        CppExtension(
            'cpp_loader',
            ['cpp_loader_optimized.cpp'],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
