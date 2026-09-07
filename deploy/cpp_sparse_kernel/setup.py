"""Build the checked SPECTRA PyTorch C++ extension with safe backend selection."""
from __future__ import annotations

import platform
import sys
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

_HERE = Path(__file__).resolve().parent


def cpu_supports_avx2() -> bool:
    machine = platform.machine().lower()
    if machine not in {"x86_64", "amd64", "i386", "i686", "x86"}:
        return False
    try:
        cap = str(torch.backends.cpu.get_cpu_capability()).upper()
        return "AVX2" in cap or "AVX512" in cap
    except Exception:
        return False


use_avx2 = cpu_supports_avx2()
if sys.platform == "win32":
    flags = ["/O2", "/std:c++20"] + (["/arch:AVX2"] if use_avx2 else [])
else:
    flags = ["-O3", "-std=c++20"] + (["-mavx2"] if use_avx2 else [])
print(f"[spectra] building {'AVX2' if use_avx2 else 'scalar'} torch extension")

setup(
    name="spectra_kernel_ext",
    ext_modules=[CppExtension(
        name="spectra_kernel_ext",
        sources=[str(_HERE / "extension.cpp")],
        extra_compile_args=flags,
    )],
    cmdclass={"build_ext": BuildExtension},
)
