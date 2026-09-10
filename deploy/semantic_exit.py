"""M14-equivalent FP recurrence with an exact, fused native Sudoku exit.

Only finite FP32 logits, CPU int64 B=1 puzzles and square-box Sudoku are accepted.
The loaded extension is setup cost; a new owning checker is constructed INSIDE
each solve. No cached per-test-instance answer or reference target is accepted.
"""
from __future__ import annotations

import torch
from deploy.m10_native import load_extension
from model.trm import TRM


@torch.inference_mode()
def native_semantic_exit(model: TRM, x: torch.Tensor, max_steps: int = 4, *, box: int = 2):
    if type(box) is not int or not 1 <= box <= 8:
        raise ValueError("box must be an integer in [1,8]")
    n = box * box
    if x.shape != (1, n*n) or x.dtype != torch.int64 or x.device.type != "cpu":
        raise ValueError("native semantic exit requires CPU int64 x[1,N*N]")
    if type(max_steps) is not int or not 1 <= max_steps <= model.N_sup:
        raise ValueError("max_steps must be within the trained supervision budget")
    if model.n != 1 or model.T != 1 or model.ternary or model.act8:
        raise ValueError("supported fidelity boundary is FP recurrence with n=T=1")
    model.eval()
    problem = load_extension().SudokuProblem(x.contiguous(), box)
    x_emb = model.token_embed(x) + model.encode_positions(x, n, n)
    y, z = torch.zeros_like(x_emb), torch.zeros_like(x_emb)
    valid = False
    answer = None
    for step in range(1, max_steps+1):
        y, z = model.recursive_cycle(x_emb, y, z)
        answer, valid = problem.decode(model.out_head(y).contiguous())
        if valid:
            break
    return answer, {"executed_steps": step, "block_applications": step * 2 * len(model.blocks),
                    "semantic_checks": step, "checker_constructions": 1,
                    "final_semantic": valid, "target_used": False,
                    "checker": "native_exact_sudoku_v1",
                    "stop_reason": "semantic_valid" if valid else "budget_exhausted"}
