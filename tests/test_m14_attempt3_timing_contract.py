import torch

from model.single_stream_trm import InputConditionedSingleStreamTRM
from scripts.m14_attempt3_run import semantic_early_exit_solve_exact


def test_exact_runtime_records_final_semantic_without_learned_halter():
    torch.manual_seed(0)
    model = InputConditionedSingleStreamTRM(
        dim=16, num_tokens=5, seq_len=16, n_layers=1, T=1, N_sup=4,
        heads=4, alpha_y=0.1, max_grid_size=8,
    ).eval()
    x = torch.tensor([[1,0,0,4,0,4,1,0,0,1,4,0,4,0,0,1]])
    calls = 0
    def validator(_x, _a, _box):
        nonlocal calls
        calls += 1
        return torch.tensor([calls == 2])
    _, work = semantic_early_exit_solve_exact(model, x, 4, validator=validator)
    assert calls == 2
    assert work["executed_steps"] == 2
    assert work["semantic_checks"] == 2
    assert work["final_semantic"] is True
    assert work["halt_head_calls"] == 0
