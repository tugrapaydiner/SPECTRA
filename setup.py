"""Native compilation is explicit; ordinary wheels are compiler-free.

``python setup.py build_ext --inplace`` remains supported for historical
workflows. To build a binary wheel, set SPECTRA_BUILD_NATIVE=1 with PyTorch
already installed and use --no-build-isolation. Build failures are errors.
"""
from __future__ import annotations
import os
import platform
import sys
from setuptools import setup


def _cpu_supports_avx2(torch_module) -> bool:
    if platform.machine().lower() not in {"x86_64", "amd64", "i386", "i686", "x86"}:
        return False
    try:
        cap = str(torch_module.backends.cpu.get_cpu_capability()).upper()
        return "AVX2" in cap or "AVX512" in cap
    except Exception:
        return False


def native_options() -> dict:
    value = os.environ.get("SPECTRA_BUILD_NATIVE", "0")
    if value not in {"0", "1"}:
        raise ValueError("SPECTRA_BUILD_NATIVE must be 0 or 1")
    if "build_ext" not in sys.argv and value != "1":
        return {"ext_modules": [], "cmdclass": {}}
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CppExtension
    except ImportError as error:
        raise RuntimeError("Explicit native builds require PyTorch; install the native extra first") from error
    avx2 = _cpu_supports_avx2(torch)
    if sys.platform == "win32":
        flags = ["/O2", "/std:c++20"] + (["/arch:AVX2"] if avx2 else [])
    else:
        flags = ["-O3", "-std=c++20"] + (["-mavx2"] if avx2 else [])
    return {"ext_modules": [CppExtension(name="spectra_kernel_ext",
                sources=["deploy/cpp_sparse_kernel/extension.cpp"],
                extra_compile_args={"cxx": flags})],
            "cmdclass": {"build_ext": BuildExtension}}

setup(**native_options())
