"""2-bit ternary weight packing (BLUEPRINT section 26.1).

Ternary weights ``{-1, 0, +1}`` are packed into 2 bits each (4 per byte) for the
W1.58 storage target. Code map (section 26.1):

    00 -> 0,  01 -> +1,  10 -> -1,  11 -> reserved

A 7M-parameter ternary core is ~1.4 MB packed, which is why cache-resident
inference is plausible. This module is pure NumPy so the round-trip can be tested
without the C++ kernel.
"""

from __future__ import annotations

import numpy as np

# Ternary value <-> 2-bit code.
_VALUE_TO_CODE = {0: 0b00, 1: 0b01, -1: 0b10}
_CODE_TO_VALUE = np.array([0, 1, -1, 0], dtype=np.int8)  # index by 2-bit code


def ternary_to_codes(weights: np.ndarray) -> np.ndarray:
    """Map ternary weights ``{-1,0,1}`` to 2-bit codes ``{0,1,2}`` (uint8)."""
    w = np.asarray(weights)
    uniq = set(np.unique(w).tolist())
    if not uniq.issubset({-1, 0, 1}):
        raise ValueError(f"weights must be ternary, got values {sorted(uniq)}")
    codes = np.zeros(w.shape, dtype=np.uint8)
    codes[w == 1] = _VALUE_TO_CODE[1]
    codes[w == -1] = _VALUE_TO_CODE[-1]
    return codes


def pack_ternary(weights: np.ndarray) -> tuple[np.ndarray, int]:
    """Pack a ternary weight array into bytes (4 weights/byte, LSB-first).

    Returns ``(packed uint8 [ceil(n/4)], n)`` where ``n`` is the original element
    count (needed to trim padding on unpack).
    """
    codes = ternary_to_codes(weights).reshape(-1)
    n = codes.size
    pad = (-n) % 4
    if pad:
        codes = np.concatenate([codes, np.zeros(pad, dtype=np.uint8)])
    codes = codes.reshape(-1, 4)
    packed = (
        codes[:, 0]
        | (codes[:, 1] << 2)
        | (codes[:, 2] << 4)
        | (codes[:, 3] << 6)
    ).astype(np.uint8)
    return packed, n


def unpack_ternary(packed: np.ndarray, n: int, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Unpack bytes back to ternary values ``{-1,0,1}`` (int8).

    Args:
        packed: Packed bytes from :func:`pack_ternary`.
        n: Original element count.
        shape: Optional shape to restore (must have ``prod(shape) == n``).
    """
    packed = np.asarray(packed, dtype=np.uint8)
    codes = np.empty((packed.size, 4), dtype=np.uint8)
    codes[:, 0] = packed & 0b11
    codes[:, 1] = (packed >> 2) & 0b11
    codes[:, 2] = (packed >> 4) & 0b11
    codes[:, 3] = (packed >> 6) & 0b11
    values = _CODE_TO_VALUE[codes.reshape(-1)[:n]]
    if shape is not None:
        values = values.reshape(shape)
    return values


def packed_size_bytes(n_weights: int) -> int:
    """Number of bytes needed to pack ``n_weights`` ternary values."""
    return (n_weights + 3) // 4
