"""Extreme malformed geometry must be rejected before signed size arithmetic."""
import pytest
import torch
from deploy.m10_native import load_extension


@pytest.mark.parametrize("hidden", [2**63-1, 2**63-2, 2**63-3])
def test_native_handle_rejects_extreme_hidden_geometry(hidden):
    extension = load_extension()
    with pytest.raises(RuntimeError, match="packed_weight must contain"):
        extension.ValidatedPackedLinear(torch.zeros(1, dtype=torch.uint8),
            torch.ones(1), torch.empty(0), 1, hidden)
