"""Explicitly targeted state-value networks and a hash-bound M16 loader.

The neural body is exactly GroundedStateVerifier's architecture, but these models
are not implicitly labelled one-cycle improvement predictors. Python modules are
not tamper-proof objects: hashes bind checkpoint bytes at load, not future user
mutation of parameters. Learned scores are never exact-checker certificates.
"""
from __future__ import annotations
import io
from pathlib import Path
import torch

from data.ancestry import digest, require_sha
from eval.verified_search import ValueContract, ValueTarget
from model.grounded_verifier import GroundedStateVerifier


class TypedStateValue(GroundedStateVerifier):
    """Same state encoder; sigmoid predicts the explicitly declared target."""
    def __init__(self, target: ValueTarget, **architecture):
        if not isinstance(target, ValueTarget):
            raise TypeError("explicit ValueTarget required")
        super().__init__(**architecture)
        self.target = target

    def forward_logits(self, x, y, z, width=9):
        """Logits for self.target, not implicitly improvement logits."""
        return super().forward_logits(x, y, z, width)

    def forward(self, x, y, z, width=9):
        """Probability/normalized quality for the declared target."""
        return torch.sigmoid(self.forward_logits(x, y, z, width))

    def value_state(self, x, y, z, width=9):
        return self.forward(x, y, z, width)


def load_m16_value(path: str | Path, *, expected_sha256: str,
                   expected_core_sha256: str, expected_training_manifest_sha256: str,
                   diagnostic_improvement: bool = False):
    """Strict-load v1's fixed architecture and consumed-data metadata.

    expected_core_sha256 must come from the independently verified reasoner, not
    from the value checkpoint itself. The caller separately closes the manifest's
    ancestor graph before drawing evaluation data. Diagnostics must explicitly
    opt in to loading an improvement predictor; search still rejects that target.
    """
    raw = Path(path).read_bytes()
    if digest(raw) != require_sha(expected_sha256):
        raise ValueError("value checkpoint content hash mismatch")
    require_sha(expected_core_sha256)
    require_sha(expected_training_manifest_sha256)
    p = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    expected_keys = {"format", "target", "core_sha256", "model_state", "architecture", "data_ancestry"}
    if not isinstance(p, dict) or set(p) != expected_keys or p["format"] != "spectra.m16_typed_verifier.v1":
        raise ValueError("unsupported typed-value checkpoint inventory/format")
    if p["core_sha256"] != expected_core_sha256:
        raise ValueError("value checkpoint is bound to another reasoner")
    if p["architecture"] != {"dim": 64, "n_layers": 1, "heads": 4, "act_bits": 8}:
        raise ValueError("unsupported v1 value architecture")
    ancestry = {"parent_checkpoints": [expected_core_sha256],
                "consumed_manifests": [{"sha256": expected_training_manifest_sha256,
                                        "splits": ["train"], "role": "training"}]}
    if p["data_ancestry"] != ancestry:
        raise ValueError("missing or incompatible value training ancestry")
    target = ValueTarget(p["target"])
    if target is ValueTarget.BUDGET:
        raise ValueError("v1 has no budget-conditioned target metadata")
    if target is ValueTarget.IMPROVEMENT and not diagnostic_improvement:
        raise ValueError("improvement checkpoint is diagnostic, not an absolute selector")
    model = TypedStateValue(target, num_tokens=5, dim=64, n_layers=1, heads=4,
                            max_grid_size=8, act_bits=8, include_y=True)
    model.load_state_dict(p["model_state"], strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    contract = ValueContract(target, expected_core_sha256, expected_sha256, "fp64_cycle_action_v1")
    return model, contract
