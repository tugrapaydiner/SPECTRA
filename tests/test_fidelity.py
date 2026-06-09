"""RFC fidelity-audit compliance tests.

Locks the codebase to BLUEPRINT.md where it deviated:
  * the deep-supervision loss equals the section-17 equation exactly,
  * the C++ kernel bridge refuses to silently upcast FP32 -> INT8,
  * the latent collapse detector covers dimensional (rank-1) collapse,
  * training hyperparameters come from config/*.yaml, not hardcoded literals.
"""

import pytest
import torch
import torch.nn.functional as F

from common import load_config
from deploy import torch_kernel
from train.collapse_watch import check_collapse, dimensional_collapse_ratio
from train.losses import deep_supervision_loss
from train.trainer import TrainConfig


# --------------------------------------------------------------------------- #
# 1. Math-to-tensor fidelity (section 17)
# --------------------------------------------------------------------------- #
def test_deep_supervision_matches_blueprint_section17():
    torch.manual_seed(0)
    b, length, vocab = 2, 4, 5
    steps = [
        {"logits": torch.randn(b, length, vocab), "halt_logit": torch.randn(b)}
        for _ in range(3)
    ]
    y = torch.randint(0, vocab, (b, length))

    got = deep_supervision_loss(steps, y, lambda_h=0.5, lambda_improve=0.1, margin=0.01)

    total = torch.zeros(())
    prev = None
    for s in steps:
        lg = s["logits"]
        ce = F.cross_entropy(lg.reshape(-1, vocab), y.reshape(-1))
        correct = (lg.argmax(-1) == y).all(dim=1).float()
        hb = F.binary_cross_entropy_with_logits(s["halt_logit"], correct)
        tlp = F.log_softmax(lg, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
        imp = torch.zeros(())
        if prev is not None:
            imp = F.relu(prev.detach() + 0.01 - tlp).mean()
        prev = tlp
        total = total + ce + 0.5 * hb + 0.1 * imp
    assert torch.allclose(got, total / len(steps), atol=1e-6)


# --------------------------------------------------------------------------- #
# 2. Hardware-boundary: no silent FP32 upcast across the kernel bridge
# --------------------------------------------------------------------------- #
def test_kernel_bridge_rejects_fp32_upcast():
    fp32_x = torch.zeros(1, 4)  # forgot .to(int8)
    with pytest.raises(TypeError):
        torch_kernel.sparse_ternary_gemv(
            fp32_x, torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.uint8), torch.zeros(1, dtype=torch.int32), 15, 1,
        )
    # Correct dtypes pass the guard (then only fail because the ext isn't built here).
    if not torch_kernel.available():
        with pytest.raises(RuntimeError):
            torch_kernel.sparse_ternary_gemv(
                torch.zeros(1, 4, dtype=torch.int8), torch.zeros(1, dtype=torch.int32),
                torch.zeros(1, dtype=torch.uint8), torch.zeros(1, dtype=torch.int32), 15, 1,
            )


# --------------------------------------------------------------------------- #
# 3. Collapse detector covers dimensional (rank-1) collapse
# --------------------------------------------------------------------------- #
def test_dimensional_collapse_detector():
    b, length, dim = 4, 8, 16
    z_collapsed = torch.randn(b, 1, dim).expand(b, length, dim).contiguous()  # all tokens equal
    z_diverse = torch.randn(b, length, dim)
    assert dimensional_collapse_ratio(z_collapsed) < 1e-6
    assert dimensional_collapse_ratio(z_diverse) > 0.1

    def steps(z):
        return [{"logits": torch.randn(b, length, 5), "y": z, "z": z}]

    rep_c = check_collapse(steps(z_collapsed), accuracy=0.0)
    rep_d = check_collapse(steps(z_diverse), accuracy=0.0)
    assert rep_c.dimensional_collapse and rep_c.collapsed
    assert not rep_d.dimensional_collapse


# --------------------------------------------------------------------------- #
# 4. Hyperparameters come from config, not hardcoded literals
# --------------------------------------------------------------------------- #
def test_trainconfig_is_driven_by_yaml():
    cfg = load_config("config/sudoku.yaml")
    tc = TrainConfig.from_config(cfg)
    assert tc.lr == cfg.train.lr           # 1e-3 from base.yaml, not a literal
    assert tc.ema_decay == 0.999
    assert tc.lambda_h == 0.5
    assert tc.lambda_improve == 0.1
    assert tc.margin == 0.01
    assert tc.lr_warmup_steps == 500       # warmup_steps mapped through
    assert tc.quant_warmup_steps == 2000

    overridden = TrainConfig.from_config(
        load_config("config/sudoku.yaml", overrides=["train.lr=0.005"])
    )
    assert overridden.lr == 0.005          # config overrides flow into training
