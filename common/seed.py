"""Deterministic seeding, independent RNG streams, and RNG-state capture."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os
import random
from typing import Any, Iterator

import numpy as np
import torch


@dataclass(frozen=True)
class SeedStreams:
    """Stable independent seeds derived from one experiment seed."""

    base: int
    data: int
    model: int
    train: int
    eval: int

    def as_dict(self) -> dict[str, int]:
        return {
            "base": self.base,
            "data": self.data,
            "model": self.model,
            "train": self.train,
            "eval": self.eval,
        }


def derive_seed(seed: int, stream: str) -> int:
    """Derive a stable 63-bit seed without Python hash randomization."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not stream:
        raise ValueError("stream name must be non-empty")
    payload = f"spectra-seed-stream-v1\0{seed}\0{stream}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return value % (2**63 - 1)


def make_seed_streams(seed: int) -> SeedStreams:
    """Create independent data/model/train/eval streams from a base seed."""
    return SeedStreams(
        base=int(seed),
        data=derive_seed(seed, "data"),
        model=derive_seed(seed, "model"),
        train=derive_seed(seed, "train"),
        eval=derive_seed(seed, "eval"),
    )


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch.

    ``deterministic=True`` is exercised by the M04 CPU reference gate. It is not
    a promise of universal GPU bitwise determinism.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    seed64 = int(seed) % (2**63 - 1)
    os.environ["PYTHONHASHSEED"] = str(seed64)
    random.seed(seed64)
    np.random.seed(seed64 % (2**32))
    torch.manual_seed(seed64)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed64)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)


def capture_rng_state() -> dict[str, Any]:
    """Return a torch-save-safe snapshot of all RNGs SPECTRA uses."""
    py = random.getstate()
    np_state = np.random.get_state()
    return {
        "schema_version": 1,
        "python": {
            "version": int(py[0]),
            "state": list(py[1]),
            "gauss": py[2],
        },
        "numpy": {
            "bit_generator": str(np_state[0]),
            "state": np_state[1].astype(np.uint32, copy=False).tolist(),
            "position": int(np_state[2]),
            "has_gauss": int(np_state[3]),
            "cached_gaussian": float(np_state[4]),
        },
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "torch_cuda": [
            state.cpu().clone() for state in torch.cuda.get_rng_state_all()
        ] if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a snapshot produced by :func:`capture_rng_state`."""
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise ValueError("unsupported RNG-state schema")

    py = state["python"]
    random.setstate((
        int(py["version"]),
        tuple(int(v) for v in py["state"]),
        py["gauss"],
    ))

    ns = state["numpy"]
    np.random.set_state((
        str(ns["bit_generator"]),
        np.asarray(ns["state"], dtype=np.uint32),
        int(ns["position"]),
        int(ns["has_gauss"]),
        float(ns["cached_gaussian"]),
    ))

    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda", [])
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA RNG-state count does not match the current device count"
            )
        torch.cuda.set_rng_state_all([s.cpu() for s in cuda_states])


@contextmanager
def isolated_seed(seed: int) -> Iterator[None]:
    """Run under an isolated seed without perturbing training RNG state."""
    state = capture_rng_state()
    try:
        set_seed(seed, deterministic=torch.are_deterministic_algorithms_enabled())
        yield
    finally:
        restore_rng_state(state)


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
