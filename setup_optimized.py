import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

# When running inside an already-initialized VS Developer shell on Windows,
# torch cpp_extension expects DISTUTILS_USE_SDK=1 to avoid re-activation warnings/errors.
if os.name == 'nt' and os.environ.get('VSCMD_VER'):
    os.environ.setdefault('DISTUTILS_USE_SDK', '1')
    os.environ.setdefault('MSSdk', '1')

def is_cuda_available():
    try:
        if not torch.cuda.is_available():
            return False
        # Optional: check if NVCC is in path
        return True
    except Exception as e:
        raise RuntimeError(
            f"CUDA capability probe failed in setup_optimized.py: {type(e).__name__}: {e}"
        ) from e

cuda_available = is_cuda_available()
extra_compile_args = {'cxx': []}

if os.name == 'nt':
    # MSVC Flags
    args = ['/arch:AVX512', '/fp:fast', '/favor:INTEL64', '/Ox', '/DNDEBUG', '/openmp', '/Oi', '/Ot', '/Qpar']
    extra_compile_args['cxx'] = args
    if cuda_available:
        extra_compile_args['nvcc'] = ['-O3', '--use_fast_math']
else:
    # GCC/Clang Flags
    args = ['-mavx512f', '-mavx512dq', '-mavx512bw', '-mavx512vl', '-mavx512vnni', '-mavx2', '-mfma', '-O3', '-march=native', '-ffast-math', '-fopenmp']
    extra_compile_args['cxx'] = args
    if cuda_available:
        extra_compile_args['nvcc'] = ['-O3', '--use_fast_math']

ext_class = CUDAExtension if cuda_available else CppExtension
sources = ['cpp_loader_optimized.cpp']

setup(
    name='cpp_loader',
    ext_modules=[
        ext_class(
            'cpp_loader',
            sources,
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
