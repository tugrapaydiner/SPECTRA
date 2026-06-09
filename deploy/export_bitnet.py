"""Export a trained ternary core to a packed W1.58 representation (Phase 11).

Walks the model's ``FakeBitLinear`` layers, hard-ternarizes each master weight to
``{-1,0,+1}`` with its per-output-channel scale, and packs the ternary codes to
2 bits (``deploy/pack_ternary``). The packed code + FP scales are exactly the
``gamma * W_q`` the fake-quant forward uses, so reconstruction is lossless and the
deployed weights match PyTorch bit-for-bit.

This is the format a ``bitnet.cpp`` / custom-kernel backend consumes; the adapter
that actually runs it lives in ``deploy/bitnet_cpp_adapter`` (Linux/native).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch.nn as nn

from deploy.pack_ternary import pack_ternary, packed_size_bytes, unpack_ternary
from model.stability import iter_bitlinears


def export_ternary_weights(model: nn.Module) -> dict[str, dict]:
    """Return ``{layer_name: {packed, n, shape, scale}}`` for every ternary layer."""
    export: dict[str, dict] = {}
    for name, module in iter_bitlinears(model):
        w_q, scale = module._ternarize_hard(module.weight)
        packed, n = pack_ternary(w_q.detach().cpu().numpy().astype(np.int8))
        export[name] = {
            "packed": packed,
            "n": n,
            "shape": tuple(int(s) for s in w_q.shape),
            "scale": scale.detach().cpu().numpy().astype(np.float32),
        }
    return export


def reconstruct_weight(entry: dict) -> np.ndarray:
    """Reconstruct ``gamma * W_q`` ``[out, in]`` from a packed export entry."""
    w_q = unpack_ternary(entry["packed"], entry["n"], entry["shape"]).astype(np.float32)
    return w_q * entry["scale"]  # scale is [out, 1], broadcasts over in


def export_summary(model: nn.Module) -> dict:
    """Packed-size accounting for the ternary layers (section 6.4)."""
    n_weights = 0
    packed_bytes = 0
    scale_floats = 0
    for _, module in iter_bitlinears(model):
        n = module.weight.numel()
        n_weights += n
        packed_bytes += packed_size_bytes(n)
        scale_floats += module.weight.shape[0]
    return {
        "ternary_params": n_weights,
        "packed_weight_bytes": packed_bytes,
        "packed_weight_mb": packed_bytes / (1024 * 1024),
        "scale_floats": scale_floats,
        "bits_per_weight": (packed_bytes * 8 / n_weights) if n_weights else 0.0,
    }


def save_export(model: nn.Module, path: str | Path) -> dict:
    """Save the packed ternary export to a ``.npz`` file; returns the summary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    export = export_ternary_weights(model)
    flat: dict[str, np.ndarray] = {}
    for name, entry in export.items():
        flat[f"{name}.packed"] = entry["packed"]
        flat[f"{name}.scale"] = entry["scale"]
        flat[f"{name}.meta"] = np.array([entry["n"], *entry["shape"]], dtype=np.int64)
    np.savez(path, **flat)
    return export_summary(model)
