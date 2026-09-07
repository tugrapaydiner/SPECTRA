import torch

import scripts.m14_attempt2_single_stream as a2
from scripts.m14_attempt4_anytime_supervision import (
    FRONT_LOADED_WEIGHTS,
    frontloaded_candidate_training_loss,
)
from scripts.m14_primary_experiment import practical_gate


class DummyFourStep:
    def __init__(self, logits):
        self.logits = logits
    def __call__(self, x, height=4, width=4):
        assert height == width == 4
        return self.logits[-1], [{"logits": z} for z in self.logits]


def test_attempt4_weights_are_fixed_exact_reverse_of_historical_schedule():
    assert FRONT_LOADED_WEIGHTS == [0.4, 0.3, 0.2, 0.1]
    assert sum(FRONT_LOADED_WEIGHTS) == 1.0
    assert list(reversed(FRONT_LOADED_WEIGHTS)) == [0.1, 0.2, 0.3, 0.4]


def test_frontloaded_loss_matches_hand_composition():
    torch.manual_seed(4)
    x = torch.tensor([[1,0,0,4,0,4,1,0,0,1,4,0,4,0,0,1]])
    y = torch.tensor([[1,2,3,4,3,4,1,2,2,1,4,3,4,3,2,1]])
    logits = [torch.randn(1,16,5) + i * 0.15 for i in range(4)]
    model = DummyFourStep(logits)
    observed = frontloaded_candidate_training_loss(model, x, y)
    expected = sum(
        w * a2.blank_ce(z, x, y)
        for w, z in zip([0.4,0.3,0.2,0.1], logits)
    )
    assert torch.allclose(observed, expected, atol=0, rtol=0)


def test_shared_attempt2_training_helper_is_patched_before_use():
    assert a2.candidate_training_loss is frontloaded_candidate_training_loss


def test_attempt4_does_not_change_m14_numerical_gate():
    good = practical_gate({
        "quality_difference": 0.03,
        "quality_ci95": [0.001, 0.05],
        "latency_ratio_candidate_over_baseline": 1.15,
        "latency_ratio_ci95": [1.0, 1.20],
    })
    assert good["pass"] is True
    bad = practical_gate({
        "quality_difference": 0.0299,
        "quality_ci95": [0.001, 0.05],
        "latency_ratio_candidate_over_baseline": 1.15,
        "latency_ratio_ci95": [1.0, 1.20],
    })
    assert bad["pass"] is False
