"""Independent reconstruction of the frozen M17 candidate-selection experiment.

This module deliberately does not call m17_models.candidate_pool, TaskSpec.restore,
TaskSpec.quality, or the Torch/native semantic checkers. Neural core operations and
frozen value heads are shared; answer restoration and target construction use
NumPy and the reference task validators. No reference answer enters reconstruction.
The frozen 32/64 batch geometry is retained for numerical reproducibility.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import torch

from data import maze, sudoku
from eval.checkable_tasks import TaskSpec, require_core
from eval.fixed_pool import TARGET_NAMES, score_pool

POOL_BATCH = 32
VALUE_BATCH = 64
# Bounded CPU FP32 roundoff only. Labels, identities, choices and successes must
# match exactly; a near tie selecting a different candidate still FAILS.
SCORE_ABSOLUTE_TOLERANCE = 2e-6


def decode_metrics(x: torch.Tensor, logits: torch.Tensor, spec: TaskSpec):
    """Independently restore/decode answers and construct reference-free labels."""
    spec.validate_input(x)
    if (not isinstance(logits, torch.Tensor) or logits.device.type != "cpu"
        or logits.dtype != torch.float32
        or tuple(logits.shape) != (*x.shape, spec.num_tokens)
        or not bool(torch.isfinite(logits).all())):
        raise ValueError("replay logits must be finite CPU FP32 [B,L,V]")
    inp = x.detach().numpy()
    prediction = logits.detach().numpy().argmax(axis=-1)
    if spec.task == "sudoku":
        answer = np.where(inp != 0, inp, prediction)
    else:
        answer = np.where((inp == maze.OPEN) & (prediction == maze.PATH), maze.PATH, inp)
    validity, quality = [], []
    for puzzle, candidate in zip(inp, answer):
        p = puzzle.reshape(spec.height, spec.width)
        a = candidate.reshape(spec.height, spec.width)
        if spec.task == "sudoku":
            n, box = spec.width, spec.box
            puzzle_ok = sudoku.is_valid_grid(p, box)
            valid = (puzzle_ok and sudoku.is_solved(a, box)
                     and sudoku.respects_clues(p, a, box))
            target = np.arange(1, n + 1)
            blocks = np.asarray([a[r:r+box, c:c+box].reshape(-1)
                                 for r in range(0, n, box) for c in range(0, n, box)])
            fractions = [np.mean(np.all(np.sort(g, axis=-1) == target, axis=-1), dtype=np.float32)
                         for g in (a, a.T, blocks)]
            given = p != 0
            fractions.append(np.mean(a[given] == p[given], dtype=np.float32)
                             if given.any() else np.float32(1))
            score = np.float32(0)
            for fraction in fractions:
                score = np.float32(score + np.float32(.25) * fraction)
            score = np.float32(score * puzzle_ok)
        else:
            valid = maze.candidate_success(p, a, spec.height, spec.width, require_optimal=True)
            walls = np.mean((p == maze.WALL) == (a == maze.WALL), dtype=np.float32)
            endpoints = np.mean(((p == maze.START) == (a == maze.START)) &
                                ((p == maze.GOAL) == (a == maze.GOAL)), dtype=np.float32)
            path = a == maze.PATH
            overlay = (np.float32(np.count_nonzero(path & (p == maze.OPEN))) /
                       np.float32(np.count_nonzero(path))) if path.any() else np.float32(0)
            score = np.float32((walls + endpoints + overlay + np.float32(valid)) / np.float32(4))
        validity.append(bool(valid))
        quality.append(score)
    return (torch.from_numpy(answer.astype(np.int64, copy=False)),
            torch.tensor(validity, dtype=torch.bool), torch.tensor(np.asarray(quality, dtype=np.float32)))


@torch.no_grad()
def reconstruct_pool(core, inputs: torch.Tensor, spec: TaskSpec, *, batch_size: int = POOL_BATCH):
    """Execute all 16 frozen candidates and their declared one-cycle continuations.

    Action zero advances the shared identity spine. Other actions branch from
    that same unmodified parent. Returned tensor fields intentionally match the
    historical digest inventory; diagnostics are returned separately.
    """
    require_core(core, spec)
    spec.validate_input(inputs)
    if not len(inputs) or type(batch_size) is not int or batch_size < 1:
        raise ValueError("replay needs nonempty inputs and a positive integer batch")
    rng = torch.Generator(device="cpu").manual_seed(16111)
    random = torch.randn(3, spec.dim, generator=rng, device="cpu")
    directions = torch.cat([torch.zeros(1, spec.dim), .5 * random / random.norm(dim=1, keepdim=True)])
    pieces = []
    diagnostic = {"candidate_decodes": 0, "continuation_decodes": 0,
                  "maze_restricted_score_checks": 0, "maze_restricted_score_failures": 0,
                  "invalid_quality_histogram": Counter(), "empty_path_candidates": 0,
                  "maze_nonempty_improvement_checks": 0, "maze_nonempty_improvement_failures": 0,
                  "maze_nonempty_improvement_positives": 0}
    for start in range(0, len(inputs), batch_size):
        x = inputs[start:start+batch_size]
        embedded = core.token_embed(x) + core.encode_positions(x, spec.height, spec.width)
        y = torch.zeros_like(embedded)
        z = torch.zeros_like(embedded)
        for depth in range(1, 5):
            spine = None
            for action in range(4):
                child_y, child_z = core.recursive_cycle(embedded, y, z + directions[action])
                if action == 0:
                    spine = child_y, child_z
                answer, valid, before = decode_metrics(x, core.out_head(child_y), spec)
                next_y, _ = core.recursive_cycle(embedded, child_y, child_z)
                _, next_valid, after = decode_metrics(x, core.out_head(next_y), spec)
                labels = torch.stack([(after > before + 1e-6).float(), valid.float(), before], dim=1)
                diagnostic["candidate_decodes"] += len(x)
                diagnostic["continuation_decodes"] += len(x)
                diagnostic["invalid_quality_histogram"].update(float(v) for v in before[~valid])
                if spec.task == "maze":
                    nonempty = (answer == maze.PATH).any(dim=1)
                    predicted_score = .5 + .25 * nonempty.float() + .25 * valid.float()
                    diagnostic["maze_restricted_score_checks"] += len(x)
                    diagnostic["maze_restricted_score_failures"] += int((before != predicted_score).sum())
                    diagnostic["empty_path_candidates"] += int((~nonempty).sum())
                    event = labels[:, 0].bool()
                    solve_transition = (~valid) & next_valid
                    diagnostic["maze_nonempty_improvement_checks"] += int(nonempty.sum())
                    diagnostic["maze_nonempty_improvement_failures"] += int(((event != solve_transition) & nonempty).sum())
                    diagnostic["maze_nonempty_improvement_positives"] += int((event & nonempty).sum())
                pieces.append({"x": x, "y": child_y, "z": child_z, "labels": labels,
                               "example_index": torch.arange(start, start+len(x)),
                               "depth": torch.full((len(x),), depth),
                               "action": torch.full((len(x),), action)})
            y, z = spine
    result = {key: torch.cat([p[key] for p in pieces]) for key in pieces[0]}
    diagnostic["invalid_quality_histogram"] = {str(k): v for k, v in sorted(diagnostic["invalid_quality_histogram"].items())}
    return result, diagnostic


@torch.inference_mode()
def replay_rows(core_seed: int, example_ids: list[str], pool: dict,
                values: dict, spec: TaskSpec, surface: str) -> list[dict]:
    """Evaluate the exact frozen heads, never fitting or selecting a replacement."""
    if (type(core_seed) is not int or not example_ids
        or any(not isinstance(i, str) or not i for i in example_ids)
        or len(set(example_ids)) != len(example_ids)):
        raise ValueError("unique nonempty example IDs and an integer core seed are required")
    if surface not in {"development", "confirmation"}:
        raise ValueError("unknown retained surface")
    scores = {}
    for target, value in values.items():
        if value.training or any(p.requires_grad for p in value.parameters()):
            raise ValueError("replay value head must be frozen and in eval mode")
        chunks = []
        for start in range(0, len(pool["x"]), VALUE_BATCH):
            sl = slice(start, start+VALUE_BATCH)
            chunks.append(value.value_state(pool["x"][sl], pool["y"][sl], pool["z"][sl], width=spec.width))
        scores[target.value] = torch.cat(chunks).cpu().numpy()
    if set(scores) != set(TARGET_NAMES):
        raise ValueError("all three declared target heads are required")
    indices = pool["example_index"].numpy()
    if set(indices.tolist()) != set(range(len(example_ids))):
        raise ValueError("reconstructed pool has missing or unexpected examples")
    rows = []
    for i, eid in enumerate(example_ids):
        at = np.flatnonzero(indices == i)
        if (len(at) != 16 or pool["depth"][at].tolist() != [d for d in range(1, 5) for _ in range(4)]
            or pool["action"][at].tolist() != list(range(4))*4):
            raise ValueError("reconstructed pool violates the frozen candidate order")
        rows.append(score_pool({"core_seed": core_seed, "example_id": eid,
            "validity": pool["labels"][at, 1].bool().tolist(),
            "improvement_labels": pool["labels"][at, 0].tolist(),
            "structural_quality": pool["labels"][at, 2].tolist(),
            "depths": pool["depth"][at].tolist(), "actions": pool["action"][at].tolist(),
            "scores": {name: score[at].tolist() for name, score in scores.items()},
            "surface": surface, "task": spec.metadata(), "reference_target_used": False}))
    return rows


def compare_records(recorded: list[dict], actual: list[dict]) -> dict[str, Any]:
    """Fail closed on changed labels, metadata, rankings, or missing/extra pools."""
    def index(rows):
        result = {}
        for row in rows:
            seed, eid = row.get("core_seed"), row.get("example_id")
            if type(seed) is not int or not isinstance(eid, str) or not eid:
                raise ValueError("invalid pool identity")
            key = seed, eid
            if key in result:
                raise ValueError("duplicate pool identity")
            if (len(row.get("validity", [])) != 16
                or row.get("depths") != [d for d in range(1, 5) for _ in range(4)]
                or row.get("actions") != list(range(4))*4):
                raise ValueError("pool does not contain the frozen 16 ordered candidates")
            if row.get("reference_target_used") is not False:
                raise ValueError("replay must not use reference targets")
            result[key] = row
        return result
    expected, observed = index(recorded), index(actual)
    if not expected or set(expected) != set(observed):
        raise ValueError("missing or unexpected model-example pools")
    max_error, score_count, exact_scores = 0.0, 0, 0
    for key in sorted(expected):
        old, new = expected[key], observed[key]
        for row in (old, new):
            fresh = score_pool(row)
            for field in ("covered", "selected_indices", "successes"):
                if row.get(field) != fresh[field]:
                    raise ValueError(f"stored selection arithmetic mismatch: {key}/{field}")
        if set(old) != set(new):
            raise ValueError(f"pool field inventory mismatch: {key}")
        for field in old:
            if field != "scores" and old[field] != new[field]:
                raise ValueError(f"exact replay mismatch: {key}/{field}")
        if set(old["scores"]) != set(TARGET_NAMES) or set(new["scores"]) != set(TARGET_NAMES):
            raise ValueError("target head inventory mismatch")
        for name in TARGET_NAMES:
            a, b = np.asarray(old["scores"][name]), np.asarray(new["scores"][name])
            error = float(np.max(np.abs(a-b)))
            if error > SCORE_ABSOLUTE_TOLERANCE:
                raise ValueError(f"prediction replay mismatch: {key}/{name}, error={error}")
            score_count += len(a)
            exact_scores += int(np.count_nonzero(a == b))
            max_error = max(max_error, error)
    return {"status": "PASS", "model_example_pools": len(expected),
            "candidate_states": 16*len(expected), "head_scores_compared": score_count,
            "head_scores_exact": exact_scores, "max_score_absolute_error": max_error,
            "score_absolute_tolerance": SCORE_ABSOLUTE_TOLERANCE,
            "labels_and_choices_exact": True}
