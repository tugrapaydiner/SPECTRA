"""2-bit ternary weight packing (BLUEPRINT section 26.1).

Ternary weights ``{-1, 0, +1}`` are packed into 2 bits each (4 per byte) for the
W1.58 storage target. Code map:

    00 -> 0,  01 -> +1,  10 -> -1,  11 -> reserved/invalid

``pack_ternary`` preserves the historical flat-array packing behavior.
``pack_ternary_rows`` is the native-kernel producer: every output row is padded
independently to ``ceil(hidden/4)`` bytes, with all padding codes zero.
"""

from __future__ import annotations

import numpy as np

_VALUE_TO_CODE = {0: 0b00, 1: 0b01, -1: 0b10}
_CODE_TO_VALUE = np.array([0, 1, -1, 0], dtype=np.int8)


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
    """Pack a ternary array flat, four weights per byte, LSB-first."""
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


def pack_ternary_rows(weights: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Pack a 2-D ternary matrix with independent zero padding per output row.

    Args:
        weights: Ternary matrix ``[out_dim, hidden_dim]``.

    Returns:
        ``(packed_flat, (out_dim, hidden_dim))`` where ``packed_flat`` is uint8,
        contiguous, and has exactly ``out_dim * ceil(hidden_dim/4)`` bytes.

    This layout is the required input contract for the native GEMV/FFN kernels.
    It differs from simply calling :func:`pack_ternary` on a 2-D matrix when the
    hidden dimension is not divisible by four, because flat packing would let a
    row continue into the next row's first byte.
    """
    w = np.asarray(weights)
    if w.ndim != 2:
        raise ValueError(f"weights must have shape [out_dim, hidden_dim], got {w.shape}")
    out_dim, hidden_dim = map(int, w.shape)
    if out_dim <= 0 or hidden_dim <= 0:
        raise ValueError(f"weights dimensions must be positive, got {w.shape}")
    row_bytes = packed_size_bytes(hidden_dim)
    packed = np.zeros((out_dim, row_bytes), dtype=np.uint8)
    for o in range(out_dim):
        row, n = pack_ternary(w[o])
        assert n == hidden_dim
        packed[o, : row.size] = row
    return np.ascontiguousarray(packed.reshape(-1)), (out_dim, hidden_dim)


def unpack_ternary(packed: np.ndarray, n: int, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Unpack flat packed bytes back to ternary values ``{-1,0,1}``."""
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


def unpack_ternary_rows(
    packed: np.ndarray, out_dim: int, hidden_dim: int
) -> np.ndarray:
    """Unpack native row-padded layout to ``[out_dim, hidden_dim]``."""
    if out_dim <= 0 or hidden_dim <= 0:
        raise ValueError("out_dim and hidden_dim must be positive")
    p = np.asarray(packed, dtype=np.uint8).reshape(-1)
    row_bytes = packed_size_bytes(hidden_dim)
    expected = out_dim * row_bytes
    if p.size != expected:
        raise ValueError(f"packed length must be {expected}, got {p.size}")
    out = np.empty((out_dim, hidden_dim), dtype=np.int8)
    for o in range(out_dim):
        row = p[o * row_bytes : (o + 1) * row_bytes]
        out[o] = unpack_ternary(row, hidden_dim)
    return out


def packed_size_bytes(n_weights: int) -> int:
    """Number of bytes needed to pack ``n_weights`` ternary values."""
    if n_weights < 0:
        raise ValueError("n_weights must be non-negative")
    return (n_weights + 3) // 4
