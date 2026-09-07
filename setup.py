"""Optional native build for the SPECTRA fused AVX2 ternary kernel.

The core of SPECTRA is pure PyTorch and needs NO compilation — the 5-minute clone
test is simply:

    pip install -r requirements.txt
    pytest -m "not slow"

This script builds the fused B=1 ternary GEMV kernel (``deploy/cpp_sparse_kernel``)
as a PyTorch C++ extension for edge deployment, so the deep recursion never crosses
the Python boundary per matmul:

    python setup.py build_ext --inplace

It requires a C++ toolchain ABI-compatible with the installed torch — gcc on
Linux, MSVC on Windows — with AVX2. If torch or a compiler is unavailable the
build degrades gracefully (the package still installs; the kernel falls back to
the validated ctypes path / pure-PyTorch reference).
"""

import sys
from pathlib import Path

from setuptools import setup

_SRC = Path(__file__).parent / "deploy" / "cpp_sparse_kernel" / "extension.cpp"

# Current PyTorch headers require C++20. MSVC and GCC/Clang spell the AVX2 +
# optimization flags differently.
if sys.platform == "win32":
    _CXX_FLAGS = ["/O2", "/std:c++20", "/arch:AVX2"]
else:
    _CXX_FLAGS = ["-O3", "-std=c++20", "-mavx2"]

ext_modules: list = []
cmdclass: dict = {}
try:  # torch is required to build the extension, but not to install the package.
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    ext_modules = [
        CppExtension(
            name="spectra_kernel_ext",
            sources=[str(_SRC)],
            extra_compile_args={"cxx": _CXX_FLAGS},
        )
    ]
    cmdclass = {"build_ext": BuildExtension}
except Exception as exc:  # pragma: no cover - environment dependent
    print(f"[spectra] native kernel build skipped ({exc}); using PyTorch fallback.")

# Package metadata lives in pyproject.toml; this only adds the C++ extension.
setup(ext_modules=ext_modules, cmdclass=cmdclass)
