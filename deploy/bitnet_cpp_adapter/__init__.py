"""Adapter to the native W1.58A8 backends (BLUEPRINT Phase 11/12, sections 25-26).

Two native backends sit behind this adapter:

  * **Dense `bitnet.cpp`** (Phase D1): the dense W1.58A8 CPU baseline. Point
    ``SPECTRA_BITNET_CPP`` at a built ``bitnet.cpp`` checkout to use it.
  * **Fused sparse kernel** (Phase D3): ``deploy/cpp_sparse_kernel`` built to a
    shared library and loaded here via ctypes.

Neither is built on the Windows dev box, so every entry point degrades gracefully
(``is_sparse_kernel_available()`` is False) and the PyTorch fake-quant path stays
the correctness reference. On the Linux/x86 eval box, build the kernel (see
``deploy/cpp_sparse_kernel/README.md``) and these functions load it.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

_KERNEL_DIR = Path(__file__).resolve().parents[1] / "cpp_sparse_kernel"


def _shared_lib_name() -> str:
    if sys.platform == "win32":
        return "spectra_kernel.dll"
    if sys.platform == "darwin":
        return "libspectra_kernel.dylib"
    return "libspectra_kernel.so"


def sparse_kernel_path() -> Path:
    """Expected path of the compiled sparse kernel shared library."""
    return _KERNEL_DIR / "build" / _shared_lib_name()


def is_sparse_kernel_available() -> bool:
    """True if the fused sparse kernel has been built for this platform."""
    return sparse_kernel_path().exists()


def load_sparse_kernel() -> ctypes.CDLL:
    """Load the compiled sparse kernel, or raise an informative error."""
    path = sparse_kernel_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Sparse kernel not built at {path}. Build it on Linux/x86 with "
            f"`cmake -B build && cmake --build build` in {_KERNEL_DIR} "
            f"(see README.md). The PyTorch fake-quant path is the fallback."
        )
    lib = ctypes.CDLL(str(path))
    lib.spectra_sparse_ternary_gemv.restype = None
    lib.spectra_sparse_ternary_gemv.argtypes = [
        ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int32), ctypes.c_int,  # X, active_idx, num_active
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int32), ctypes.c_int,  # W_packed, requant_mult, requant_shift
        ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int8),                     # hidden_dim, out_dim, Y
    ]
    return lib


def bitnet_cpp_root() -> Path | None:
    """Path to a built ``bitnet.cpp`` checkout, from ``$SPECTRA_BITNET_CPP``."""
    root = os.environ.get("SPECTRA_BITNET_CPP")
    return Path(root) if root and Path(root).exists() else None


def is_bitnet_cpp_available() -> bool:
    """True if a dense ``bitnet.cpp`` backend is configured."""
    return bitnet_cpp_root() is not None
