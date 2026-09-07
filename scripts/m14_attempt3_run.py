#!/usr/bin/env python3
"""Launch M14 Attempt 3 with exactly-once final semantic validation timing.

The candidate's early-exit loop necessarily evaluates semantic validity at every
executed step, including the returned step.  The generic evaluator used by older
M14 attempts performs a final validator call after ``solve``.  Repeating that call
for Attempt 3 would charge the candidate twice for its final validity decision.

This launcher installs two pre-result runtime refinements into the preregistered
Attempt-3 module:

1. return the final semantic decision in the candidate work record;
2. reuse that recorded decision in the evaluator, while controllers without an
   internal semantic check (the single-pass and symbolic baselines) receive exactly
   one final semantic check inside their timed complete solve.

No model, checkpoint, threshold, seed, split, selection rule or bootstrap rule is
changed.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.m14_attempt3_semantic_early_exit as m14
from model.single_stream_trm import InputConditionedSingleStreamTRM
from model.verifier import sudoku_correct
from scripts.m14_primary_experiment import append_jsonl, clamp_givens, SolverBundle


@torch.inference_mode()
def semantic_early_exit_solve_exact(
    model: InputConditionedSingleStreamTRM,
    x: torch.Tensor,
    max_steps: int,
    *,
    validator: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor] = sudoku_correct,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if x.ndim != 2 or x.shape != (1, 16):
        raise ValueError("semantic early exit requires x[1,16]")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or not 1 <= max_steps <= 4:
        raise ValueError("max_steps must be an integer in [1,4]")
    model.eval()
    x_emb = model.token_embed(x) + model.encode_positions(x, 4, 4)
    y = torch.zeros_like(x_emb); z = torch.zeros_like(x_emb)
    answer = None; final_semantic = False
    executed_steps = block_applications = semantic_checks = 0
    stop_reason = "budget_exhausted"
    for _ in range(max_steps):
        y, z = model.recursive_cycle(x_emb, y, z)
        executed_steps += 1
        block_applications += int(model.T * len(model.blocks))
        answer = clamp_givens(x, model.out_head(y).argmax(dim=-1))
        raw = validator(x, answer, 2).bool()
        semantic_checks += 1
        if raw.numel() != 1:
            raise RuntimeError("semantic validator must return one B=1 decision")
        final_semantic = bool(raw.item())
        if final_semantic:
            stop_reason = "semantic_valid"
            break
    assert answer is not None
    return answer, {
        "executed_steps": executed_steps,
        "block_applications": block_applications,
        "semantic_checks": semantic_checks,
        "max_steps": int(max_steps),
        "stopped_on_valid": final_semantic,
        "final_semantic": final_semantic,
        "stop_reason": stop_reason,
        "output_head_calls": executed_steps,
        "halt_head_calls": 0,
        "target_used": False,
        "semantics": model.SEMANTICS,
    }


def evaluate_bundle_exact(
    bundle: SolverBundle,
    ds,
    split: str,
    *,
    cost_n: int,
    repeats: int,
    out_rows: Path,
):
    xs = torch.from_numpy(ds.inputs).long(); ys = torch.from_numpy(ds.targets).long()
    n_cost = min(int(cost_n), len(ds))

    def checked_solve(xi: torch.Tensor):
        answer = bundle.solve(xi)
        work = bundle.work() if bundle.work is not None else {}
        if work.get("final_semantic") is not None:
            semantic = bool(work["final_semantic"])
        else:
            semantic = bool(sudoku_correct(xi, answer, 2).bool().item())
        return answer, semantic, dict(work)

    for i in range(min(4, len(ds))):
        checked_solve(xs[i:i+1])

    rows=[]
    for i in range(len(ds)):
        xi=xs[i:i+1]; latency=None; answer=None; semantic=None; first_work={}
        if i < n_cost:
            times=[]
            for _ in range(max(1,int(repeats))):
                t0=time.perf_counter_ns()
                a,ok,work=checked_solve(xi)
                times.append((time.perf_counter_ns()-t0)/1e6)
                if answer is None:
                    answer=a.detach().cpu(); semantic=ok; first_work=work
            latency=float(statistics.median(times))
        else:
            a,ok,work=checked_solve(xi)
            answer=a.detach().cpu(); semantic=ok; first_work=work
        target=ys[i:i+1]; blank=xi.cpu()==0
        row={
            "attempt":3,"split":split,"config_id":bundle.config_id,"family":bundle.family,
            "seed":bundle.seed,"example_index":i,"example_id":ds.ids[i],
            "semantic_success":int(bool(semantic)),
            "exact_reference_match":int(bool((answer==target).all().item())),
            "blank_cell_accuracy":float((answer[blank]==target[blank]).float().mean()),
            "latency_ms":latency,"latency_repeats":repeats if i<n_cost else 0,
            "complete_solve_timing_includes_semantic_check":True,"target_used_inside_solver":False,
            "work_executed_steps":first_work.get("executed_steps"),
            "work_block_applications":first_work.get("block_applications"),
            "work_semantic_checks_internal":first_work.get("semantic_checks"),
            "work_stopped_on_valid":first_work.get("stopped_on_valid"),
            "work_stop_reason":first_work.get("stop_reason"),
            "work_output_head_calls":first_work.get("output_head_calls"),
            "work_halt_head_calls":first_work.get("halt_head_calls"),
            "final_semantic_reused_without_duplicate_check":first_work.get("final_semantic") is not None,
        }
        append_jsonl(out_rows,row); rows.append(row)
    return rows


# Patch module globals used by candidate_bundle/run_validation/evaluate_surface.
m14.semantic_early_exit_solve = semantic_early_exit_solve_exact
m14.evaluate_bundle = evaluate_bundle_exact


if __name__ == "__main__":
    raise SystemExit(m14.main())
