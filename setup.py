"""Optional native build for the SPECTRA fused ternary CPU extension.

The pure-PyTorch baseline needs no compilation. ``python setup.py build_ext
--inplace`` builds a host-safe native extension: AVX2 is enabled only when the
current x86 CPU reports AVX2; otherwise the same source compiles its scalar path.
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path

from setuptools import setup

_SRC = Path(__file__).parent / "deploy" / "cpp_sparse_kernel" / "extension.cpp"


def _cpu_supports_avx2(torch_module) -> bool:
    machine = platform.machine().lower()
    if machine not in {"x86_64", "amd64", "i386", "i686", "x86"}:
        return False
    try:
        cap = str(torch_module.backends.cpu.get_cpu_capability()).upper()
        return "AVX2" in cap or "AVX512" in cap
    except Exception:
        return False


ext_modules: list = []
cmdclass: dict = {}
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    use_avx2 = _cpu_supports_avx2(torch)
    if sys.platform == "win32":
        cxx_flags = ["/O2", "/std:c++20"] + (["/arch:AVX2"] if use_avx2 else [])
    else:
        cxx_flags = ["-O3", "-std=c++20"] + (["-mavx2"] if use_avx2 else [])
    print(f"[spectra] building native {'AVX2' if use_avx2 else 'scalar'} backend")
    ext_modules = [
        CppExtension(
            name="spectra_kernel_ext",
            sources=[str(_SRC)],
            extra_compile_args={"cxx": cxx_flags},
        )
    ]
    cmdclass = {"build_ext": BuildExtension}
except Exception as exc:
    print(f"[spectra] native extension configuration unavailable: {exc}")

setup(ext_modules=ext_modules, cmdclass=cmdclass)
