"""Deterministic seeding and device resolution.

Reproducibility is a Phase 0 requirement (BLUEPRINT section 28). A single call to
``set_seed`` fixes Python, NumPy, and PyTorch RNGs so that experiments and tests
are repeatable.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed all RNGs SPECTRA depends on.

    Args:
        seed: The integer seed to apply to Python, NumPy, and PyTorch (CPU+CUDA).
        deterministic: If True, request deterministic cuDNN/algorithm behaviour.
            This trades a little speed for bit-reproducibility and is mainly
            useful when chasing nondeterministic collapse bugs (section 11).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuDNN determinism: disable autotuning and nondeterministic kernels.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Best-effort: not all ops have deterministic implementations, so we do
        # not set ``warn_only=False`` which would hard-error on those.
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve a device spec to a concrete ``torch.device``.

    ``"auto"`` selects CUDA when available, otherwise CPU. Any other value is
    passed straight through to ``torch.device``.
    """
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
