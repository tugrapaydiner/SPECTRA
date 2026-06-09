"""Build the SPECTRA sparse ternary kernel as a PyTorch C++ extension.

    python deploy/cpp_sparse_kernel/setup.py build_ext --inplace

Produces a native ``spectra_kernel_ext`` module whose ops take ``torch.Tensor``
directly, so the deep B=1 recursion never crosses the Python boundary per matmul.
Requires a C++ toolchain ABI-compatible with the installed torch (gcc on Linux,
MSVC on Windows).
"""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

_HERE = Path(__file__).resolve().parent

setup(
    name="spectra_kernel_ext",
    ext_modules=[
        CppExtension(
            name="spectra_kernel_ext",
            sources=[str(_HERE / "extension.cpp")],
            extra_compile_args=["-O3", "-mavx2", "-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
