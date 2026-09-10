"""Frozen M17 model fits and reference-free common candidate pools.

The pool builder accepts inputs, not dataset targets. Maze labels are generated
with the existing independent semantic and structural checkers. Candidate states
are shared by all three target fits; no task answer is fed into a value network.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from common.seed import set_seed
from data import maze
from data.ancestry import digest
from eval.checkable_tasks import TaskSpec, MAZE11, action_directions, require_core
from eval.fixed_pool import score_pool, summarize_pools
from eval.verified_search import ValueTarget
from model.task_value import architecture, save_task_value, load_task_value
from model.trm import TRM
from model.typed_value import TypedStateValue
from scripts.m16_evidence import write_json
from scripts.m17_sources import TARGETS

CORE_STEPS, CORE_BATCH = 1800, 32
VALUE_STEPS, VALUE_BATCH = 180, 64
WEIGHTS = [0.1, 0.2, 0.3, 0.4]
MAZE_SEEDS = [1701, 2702]


def freeze(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def core_architecture(spec: TaskSpec) -> dict:
    return {"dim": spec.dim, "num_tokens": spec.num_tokens, "seq_len": spec.height*spec.width,
            "n_layers": 1, "n": 1, "T": 1, "N_sup": 4, "heads": 4,
            "alpha_y": 0.1, "alpha_z": 0.1, "max_grid_size": spec.max_grid_size,
            "ternary": False, "act8": False}


def tensor_digest(tensors: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key, tensor in sorted(tensors.items()):
        t = tensor.detach().cpu().contiguous()
        h.update((key+":"+str(t.dtype)+":"+str(tuple(t.shape))+"\n").encode())
        h.update(memoryview(t.numpy()).cast("B"))
    return h.hexdigest()


def core_loss(model, x, target, spec=MAZE11):
    """Existing detached-supervision TRM graph, four weighted OPEN-cell losses."""
    mask = x == maze.OPEN
    if not bool(mask.any()):
        raise ValueError("maze training batch has no OPEN cells")
    _, steps = model(x, height=spec.height, width=spec.width)
    if len(steps) != 4:
        raise ValueError("M17 training requires four supervision steps")
    return sum(weight*F.cross_entropy(step["logits"][mask], target[mask])
               for weight, step in zip(WEIGHTS, steps))


@torch.inference_mode()
def validation_record(model, inputs, targets, spec):
    model.eval()
    valid, correct, count = 0, 0, 0
    for start in range(0, len(inputs), 32):
        x, target = inputs[start:start+32], targets[start:start+32]
        logits, _ = model(x, height=spec.height, width=spec.width)
        answer = spec.restore(x, logits.argmax(-1))
        valid += int(spec.correct(x, answer).sum())
        mask = x == maze.OPEN
        correct += int(((answer == target) & mask).sum())
        count += int(mask.sum())
    return {"examples": len(inputs), "strict_success": valid/max(1, len(inputs)),
            "open_cell_accuracy": correct/max(1, count), "used_for_selection": False}


def load_maze_core(path: Path, *, expected_sha: str, expected_seed: int, manifest_sha: str):
    raw = path.read_bytes()
    if digest(raw) != expected_sha:
        raise ValueError("maze core content hash mismatch")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if set(payload) != {"format", "task", "seed", "architecture", "training_manifest_sha256", "model_state"}:
        raise ValueError("maze core checkpoint inventory mismatch")
    if (payload["format"] != "spectra.m17_core.v1" or payload["task"] != MAZE11.metadata()
        or payload["seed"] != expected_seed or payload["architecture"] != core_architecture(MAZE11)
        or payload["training_manifest_sha256"] != manifest_sha):
        raise ValueError("maze core identity/architecture/ancestry mismatch")
    model = TRM(**core_architecture(MAZE11)).cpu()
    state, expected = payload["model_state"], model.state_dict()
    if set(state) != set(expected) or any(not isinstance(state[k], torch.Tensor) or
        state[k].layout != torch.strided or state[k].dtype != expected[k].dtype or
        state[k].shape != expected[k].shape or not bool(torch.isfinite(state[k]).all()) for k in expected):
        raise ValueError("maze core tensor contract mismatch")
    model.load_state_dict(state, strict=True)
    return freeze(model)


def train_maze_core(seed, datasets, manifest_sha: str, out: Path):
    set_seed(seed, deterministic=True)
    model = TRM(**core_architecture(MAZE11)).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    x = torch.from_numpy(datasets["train"].inputs).long()
    y = torch.from_numpy(datasets["train"].targets).long()
    vx = torch.from_numpy(datasets["validation"].inputs).long()
    vy = torch.from_numpy(datasets["validation"].targets).long()
    rng = np.random.default_rng(seed)
    curve, validation, sampled = [], [], hashlib.sha256()
    update_ns = validation_ns = 0
    start_all = time.perf_counter_ns()
    for step in range(1, CORE_STEPS+1):
        t0 = time.perf_counter_ns()
        model.train()
        ix = rng.integers(len(x), size=CORE_BATCH, dtype=np.int64)
        sampled.update(ix.tobytes())
        index = torch.from_numpy(ix)
        loss = core_loss(model, x[index], y[index])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite core loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not bool(torch.isfinite(grad)):
            raise RuntimeError("non-finite core gradient")
        optimizer.step()
        update_ns += time.perf_counter_ns()-t0
        if step == 1 or step % 60 == 0:
            curve.append({"step": step, "loss": float(loss.detach()), "grad_norm_preclip": float(grad)})
            print(f"maze core={seed} step={step} loss={float(loss.detach()):.6f}", flush=True)
        if step in (1, 600, 1200, 1800):
            t0 = time.perf_counter_ns()
            validation.append({"step": step, **validation_record(model, vx, vy, MAZE11)})
            validation_ns += time.perf_counter_ns()-t0
    path = out/f"maze_core_{seed}.pt"
    out.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        torch.save({"format": "spectra.m17_core.v1", "seed": seed, "task": MAZE11.metadata(),
                    "architecture": core_architecture(MAZE11), "model_state": model.state_dict(),
                    "training_manifest_sha256": manifest_sha}, f)
    sha = digest(path.read_bytes())
    model = load_maze_core(path, expected_sha=sha, expected_seed=seed, manifest_sha=manifest_sha)
    record = {"seed": seed, "path": str(path), "sha256": sha, "training_manifest_sha256": manifest_sha,
              "optimizer_steps": CORE_STEPS, "batch_size": CORE_BATCH, "lr": .001, "weight_decay": .01,
              "clip": 1., "supervision_weights": WEIGHTS, "sampled_index_sha256": sampled.hexdigest(),
              "final_checkpoint_only": True, "validation": validation, "curve": curve,
              "parameter_count": sum(p.numel() for p in model.parameters()), "device": "cpu",
              "train_block_example_equivalents": CORE_STEPS*CORE_BATCH*8,
              "update_ms": update_ns/1e6, "validation_ms": validation_ns/1e6,
              "whole_training_and_export_ms": (time.perf_counter_ns()-start_all)/1e6,
              "data_ancestry": {"consumed_manifests": [{"sha256": manifest_sha, "splits": ["train"], "role": "training"}]}}
    write_json(out/f"maze_core_{seed}_training.json", record)
    return model, sha, record


@torch.no_grad()
def candidate_pool(core: TRM, inputs: torch.Tensor, spec: TaskSpec, *, batch_size=32):
    """16 candidates per input; all labels are reference-free. Normal tensors
    (not inference tensors) are returned so auxiliary fits can save activations.
    """
    require_core(core, spec)
    spec.validate_input(inputs)
    if type(batch_size) is not int or batch_size < 1 or not len(inputs):
        raise ValueError("pool construction needs nonempty inputs and a positive batch")
    parts = []
    delta = action_directions(spec.dim)
    for start in range(0, len(inputs), batch_size):
        x = inputs[start:start+batch_size]
        xemb = core.token_embed(x)+core.encode_positions(x, spec.height, spec.width)
        y, z = torch.zeros_like(xemb), torch.zeros_like(xemb)
        for depth in range(1, 5):
            identity = None
            for action in range(4):
                yn, zn = core.recursive_cycle(xemb, y, z+delta[action])
                if action == 0:
                    identity = yn, zn
                logits = core.out_head(yn)
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError("non-finite pool logits")
                answer = spec.restore(x, logits.argmax(-1))
                valid, quality = spec.correct(x, answer).float(), spec.quality(x, answer)
                next_y, _ = core.recursive_cycle(xemb, yn, zn)
                next_logits = core.out_head(next_y)
                if not bool(torch.isfinite(next_logits).all()):
                    raise RuntimeError("non-finite continuation logits")
                after = spec.quality(x, spec.restore(x, next_logits.argmax(-1)))
                labels = torch.stack([(after > quality+1e-6).float(), valid, quality], 1)
                parts.append({"x": x, "y": yn, "z": zn, "labels": labels,
                              "example_index": torch.arange(start, start+len(x)),
                              "depth": torch.full((len(x),), depth), "action": torch.full((len(x),), action)})
            y, z = identity
    return {key: torch.cat([p[key] for p in parts], 0) for key in parts[0]}


def fit_maze_value(pool, target_index, core_seed, core_sha, manifest_sha, out: Path):
    seed = core_seed+17000
    set_seed(seed, deterministic=True)
    target = TARGETS[target_index]
    model = TypedStateValue(target, **architecture(MAZE11)).cpu()
    initial_sha = tensor_digest(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    rng = np.random.default_rng(seed)
    curve, sampled = [], hashlib.sha256()
    t0 = time.perf_counter_ns()
    model.train()
    for step in range(1, VALUE_STEPS+1):
        ix = rng.integers(len(pool["x"]), size=VALUE_BATCH, dtype=np.int64)
        sampled.update(ix.tobytes())
        index = torch.from_numpy(ix)
        logits = model.forward_logits(pool["x"][index], pool["y"][index], pool["z"][index], width=11)
        loss = F.binary_cross_entropy_with_logits(logits, pool["labels"][index, target_index])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite value loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not bool(torch.isfinite(grad)):
            raise RuntimeError("non-finite value gradient")
        optimizer.step()
        if step == 1 or step % 30 == 0:
            curve.append({"step": step, "loss": float(loss.detach()), "grad_norm_preclip": float(grad)})
    train_ms = (time.perf_counter_ns()-t0)/1e6
    freeze(model)
    path = out/f"maze_value_{core_seed}_{target.value}.pt"
    sha = save_task_value(path, model, spec=MAZE11, core_sha256=core_sha, training_manifest_sha256=manifest_sha)
    model, contract = load_task_value(path, expected_sha256=sha, expected_core_sha256=core_sha,
        expected_training_manifest_sha256=manifest_sha, spec=MAZE11,
        diagnostic_improvement=target is ValueTarget.IMPROVEMENT)
    record = {"target": target.value, "path": str(path), "sha256": sha, "seed": seed,
              "initial_state_sha256": initial_sha, "sampled_index_sha256": sampled.hexdigest(),
              "steps": VALUE_STEPS, "batch": VALUE_BATCH, "lr": .002, "weight_decay": .01, "clip": 1.,
              "train_ms": train_ms, "curve": curve, "contract": contract.metadata(),
              "reference_target_used": False, "training_manifest_sha256": manifest_sha}
    write_json(out/f"maze_value_{core_seed}_{target.value}_training.json", record)
    print(f"maze value core={core_seed} target={target.value} complete", flush=True)
    return model, contract, record


@torch.inference_mode()
def probabilities(model, pool, spec):
    values = []
    for start in range(0, len(pool["x"]), 64):
        sl = slice(start, start+64)
        values.append(model.value_state(pool["x"][sl], pool["y"][sl], pool["z"][sl], width=spec.width))
    result = torch.cat(values).cpu().numpy()
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise RuntimeError("invalid value predictions")
    return result


def fixed_surface(cores, values, dataset, spec, surface, out):
    import json
    raw_path = out/f"{surface}_pools.jsonl"
    rows, hashes = [], {}
    with raw_path.open("x") as stream:
        for seed, core in cores.items():
            pool = candidate_pool(core, torch.from_numpy(dataset.inputs).long(), spec)
            hashes[str(seed)] = tensor_digest(pool)
            probs = {t: probabilities(values[seed][t], pool, spec) for t in TARGETS}
            indices = pool["example_index"].numpy()
            for i, example_id in enumerate(dataset.ids):
                ii = np.flatnonzero(indices == i)
                if len(ii) != 16:
                    raise AssertionError("fixed pool is not the frozen 16-candidate inventory")
                record = score_pool({"core_seed": seed, "example_id": example_id,
                    "validity": [bool(v) for v in pool["labels"][ii, 1].tolist()],
                    "improvement_labels": pool["labels"][ii, 0].tolist(),
                    "structural_quality": pool["labels"][ii, 2].tolist(),
                    "depths": pool["depth"][ii].tolist(), "actions": pool["action"][ii].tolist(),
                    "scores": {t.value: probs[t][ii].tolist() for t in TARGETS},
                    "surface": surface, "task": spec.metadata(), "reference_target_used": False})
                rows.append(record)
                stream.write(json.dumps(record, allow_nan=False)+"\n")
            del pool, probs
    summary = summarize_pools(rows)
    summary.update(pool_tensor_sha256_by_core=hashes, raw_rows_sha256=digest(raw_path.read_bytes()))
    write_json(out/f"{surface}_fixed_summary.json", summary)
    return summary
