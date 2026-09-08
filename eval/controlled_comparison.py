"""M14 complete-solve and paired statistical contracts (no experiment side effects)."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import norm, t as student_t

from data import sudoku
from model.bitlinear import FakeBitLinear
from model.system1_student import System1Student
from model.trm import TRM


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def object_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def append_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def make_model(family: str, seed: int) -> torch.nn.Module:
    from common.seed import set_seed
    set_seed(seed, deterministic=True)
    if family in {"fp_recursive", "ternary_recursive"}:
        model = TRM(dim=48, num_tokens=10, seq_len=81, n_layers=1, n=1, T=1,
                    N_sup=4, heads=4, max_grid_size=16,
                    ternary=family == "ternary_recursive", act8=family == "ternary_recursive")
        for p in model.halt_head.parameters():
            p.requires_grad_(False)
        return model.cpu()
    if family == "single_pass":
        model = System1Student(dim=96, num_tokens=10, seq_len=81,
                               n_layers=2, heads=4, max_grid_size=16)
        for p in model.conf_head.parameters():
            p.requires_grad_(False)
        return model.cpu()
    raise ValueError(f"unsupported family: {family}")


def set_quant_strength(model: torch.nn.Module, strength: float) -> None:
    for module in model.modules():
        if isinstance(module, FakeBitLinear):
            module.quant_strength.fill_(strength)


def assert_hard_quantized(model: torch.nn.Module) -> None:
    strengths = [float(m.quant_strength) for m in model.modules() if isinstance(m, FakeBitLinear)]
    if isinstance(model, TRM) and model.ternary and (not strengths or any(s != 1.0 for s in strengths)):
        raise ValueError("ternary evaluation requires all quantization strengths exactly 1")


def convert_int8(model: System1Student) -> tuple[torch.nn.Module, dict]:
    if "x86" not in torch.backends.quantized.supported_engines:
        raise RuntimeError("required x86 dynamic-INT8 backend unavailable")
    torch.backends.quantized.engine = "x86"
    names = {name for name, layer in model.named_modules()
             if type(layer) is torch.nn.Linear and ".attn." not in name}
    total = sum(p.numel() for name, p in model.named_parameters() if name.endswith("weight"))
    quantized = sum(model.get_submodule(name).weight.numel() for name in names)
    result = torch.ao.quantization.quantize_dynamic(copy.deepcopy(model).eval(), names, dtype=torch.qint8)
    converted = [name for name, m in result.named_modules()
                 if isinstance(m, torch.ao.nn.quantized.dynamic.Linear)]
    if set(converted) != names or not converted:
        raise RuntimeError("dynamic-INT8 conversion did not cover the requested layers")
    return result, {"backend": "pytorch_x86_dynamic_int8", "converted_modules": sorted(converted),
                    "quantized_weight_elements": quantized, "all_weight_elements": total,
                    "quantized_weight_fraction": quantized / total,
                    "precision": "dynamic_INT8_Linear_with_FP32_attention_embeddings_norms"}


def answer_loss(model: torch.nn.Module, x: torch.Tensor, target: torch.Tensor,
                *, blank_only: bool) -> torch.Tensor:
    result, extra = model(x, height=9, width=9)
    logits = [s["logits"] for s in extra] if isinstance(model, TRM) else [result]
    mask = x == 0 if blank_only else torch.ones_like(x, dtype=torch.bool)
    if not bool(mask.any()):
        raise ValueError("no supervised cells")
    return torch.stack([torch.nn.functional.cross_entropy(z[mask], target[mask]) for z in logits]).mean()


def decode(logits: torch.Tensor, inputs: torch.Tensor, constrained: bool) -> torch.Tensor:
    if constrained:
        pred = logits[..., 1:].argmax(-1) + 1
        return torch.where(inputs != 0, inputs, pred)
    return logits.argmax(-1)


def semantic_valid(puzzle: np.ndarray, prediction: np.ndarray) -> bool:
    p = np.asarray(puzzle).reshape(9, 9)
    y = np.asarray(prediction).reshape(9, 9)
    return sudoku.is_solved(y, 3) and sudoku.respects_clues(p, y, 3)


def full_solve(model: Any, puzzle: np.ndarray, *, constrained: bool = False,
               symbolic: bool = False, native: bool = False) -> tuple[np.ndarray, bool, float]:
    """The timed function has no reference solution and includes input/output conversion."""
    start = time.perf_counter_ns()
    if symbolic:
        solution = sudoku.solve(np.asarray(puzzle).reshape(9, 9), 3)
        pred = np.zeros(81, dtype=np.int64) if solution is None else solution.reshape(-1)
    else:
        with torch.inference_mode():
            x = torch.from_numpy(np.asarray(puzzle, dtype=np.int64).copy()).long().unsqueeze(0)
            logits = model.forward(x).logits if native else model(x, height=9, width=9)[0]
            pred = decode(logits, x, constrained)[0].cpu().numpy().copy()
    success = semantic_valid(puzzle, pred)
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    return pred, bool(success), elapsed_ms


def outcome(puzzle: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> dict:
    blank = puzzle == 0
    clues = ~blank
    return {"success": int(semantic_valid(puzzle, prediction)),
            "exact": int(np.array_equal(target, prediction)),
            "blank_correct": int((target[blank] == prediction[blank]).sum()),
            "blank_count": int(blank.sum()),
            "clue_errors": int((puzzle[clues] != prediction[clues]).sum()),
            "invalid_digits": int(((prediction < 1) | (prediction > 9)).sum())}


def budget_label(ratio: float, tolerance: float) -> str:
    if not math.isfinite(ratio) or ratio <= 0 or not 0 <= tolerance < 1:
        raise ValueError("invalid latency ratio or tolerance")
    if 1 / (1 + tolerance) <= ratio <= 1 + tolerance:
        return "iso_latency"
    return "lower_cost_unmatched" if ratio < 1 else "higher_cost_unmatched"


def cost_gate(candidate_mean_ms: float, ratio_upper: float, frozen_cap_ms: float, tolerance: float) -> dict:
    values = [candidate_mean_ms, ratio_upper, frozen_cap_ms]
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("finite positive measured costs and frozen cap required")
    cap_pass = candidate_mean_ms <= frozen_cap_ms
    ratio_pass = ratio_upper <= 1 + tolerance
    return {"candidate_mean_latency_ms": candidate_mean_ms, "frozen_budget_cap_ms": frozen_cap_ms,
            "frozen_cap_pass": cap_pass, "latency_ratio_pass": ratio_pass,
            "cost_pass": cap_pass and ratio_pass}


def wilson(successes: int, n: int, alpha: float = .05) -> list[float]:
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("invalid binomial counts")
    z = float(norm.ppf(1 - alpha / 2)); p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0., center - half), min(1., center + half)]


def paired_bounds(a: np.ndarray, b: np.ndarray, ca: np.ndarray, cb: np.ndarray,
                  *, alpha: float, draws: int, seed: int) -> dict:
    """Two independent resampling dimensions: seeds and tasks, never timing rounds."""
    arrays = [np.asarray(v, dtype=float) for v in (a, b, ca, cb)]
    a, b, ca, cb = arrays
    if a.ndim != 2 or any(v.shape != a.shape for v in arrays):
        raise ValueError("paired arrays must share [training_seed, example] shape")
    if a.shape[0] < 3 or a.shape[1] < 2 or any(not np.isfinite(v).all() for v in arrays):
        raise ValueError("at least three seeds and finite observations required")
    if not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all() or np.any(ca <= 0) or np.any(cb <= 0):
        raise ValueError("invalid successes or costs")
    ns, ne = a.shape
    rng = np.random.default_rng(seed)
    differences, ratios = [], []
    for _ in range(0, draws, 200):
        k = min(200, draws - len(differences))
        si = rng.integers(ns, size=(k, ns, 1)); ei = rng.integers(ne, size=(k, 1, ne))
        differences.extend((a - b)[si, ei].mean(axis=(1, 2)).tolist())
        ratios.extend((ca[si, ei].mean(axis=(1, 2)) / cb[si, ei].mean(axis=(1, 2))).tolist())
    seed_diffs = (a - b).mean(axis=1)
    seed_log_ratios = np.log(ca.mean(axis=1) / cb.mean(axis=1))
    critical = float(student_t.ppf(1 - alpha, ns - 1))
    t_lower = float(seed_diffs.mean() - critical * seed_diffs.std(ddof=1) / math.sqrt(ns))
    t_upper_cost = float(np.exp(seed_log_ratios.mean() + critical * seed_log_ratios.std(ddof=1) / math.sqrt(ns)))
    boot_lower = float(np.quantile(differences, alpha))
    boot_upper_cost = float(np.quantile(ratios, 1 - alpha))
    return {"accuracy_difference": float((a - b).mean()),
            "accuracy_lower_bound": min(t_lower, boot_lower),
            "accuracy_t_lower": t_lower, "accuracy_bootstrap_lower": boot_lower,
            "latency_ratio": float(ca.mean() / cb.mean()),
            "latency_ratio_upper_bound": max(t_upper_cost, boot_upper_cost),
            "latency_t_upper": t_upper_cost, "latency_bootstrap_upper": boot_upper_cost,
            "seeds": ns, "paired_examples": ne, "bootstrap_draws": draws,
            "one_sided_alpha": alpha, "accuracy_seed_differences": seed_diffs.tolist(),
            "empirical_accuracy_interval_degenerate": bool(np.ptp(differences) == 0),
            "uncertainty_scope": "paired_training_seeds_and_examples_not_timing_repetitions"}


def load_partition(root: Path, name: str, purpose: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    allowed = {"train": {"training"}, "tuning": {"selection", "warmup"},
               "development": {"development_evaluation"}, "confirmation": {"confirmation_evaluation"}}
    if purpose not in allowed.get(name, set()):
        raise ValueError("split/purpose mismatch")
    if name == "confirmation":
        auth_path = root / "confirmation_authorization.json"
        if not auth_path.exists():
            raise RuntimeError("confirmation is sealed until a passing development gate")
        auth = json.loads(auth_path.read_text())
        gate_path = root / auth["development_gate_path"]
        gate = json.loads(gate_path.read_text())
        if not gate.get("passed") or digest(gate_path) != auth["development_gate_sha256"]:
            raise RuntimeError("invalid confirmation authorization")
        if (root / "confirmation_consumed.json").exists():
            raise RuntimeError("the sole confirmation evaluation has already been consumed")
        write_json(root / "confirmation_consumed.json", {"authorization_sha256": digest(auth_path), "purpose": purpose})
    manifest = json.loads((root / "data/manifest.json").read_text())
    path = root / f"data/{name}.npz"
    if digest(path) != manifest["partitions"][name]["array_sha256"]:
        raise ValueError("data artifact hash mismatch")
    with np.load(path, allow_pickle=False) as data:
        x, y = data["inputs"].copy(), data["targets"].copy()
    append_json(root / "split_access.jsonl", {"split": name, "purpose": purpose,
                                              "array_sha256": digest(path), "time_ns": time.time_ns()})
    return x, y, manifest["partitions"][name]["examples"]


def verify_confirmation_freeze(root: Path, phase: str) -> None:
    """A passing gate authorizes only its exact selected phase and configuration."""
    auth_path = root / "confirmation_authorization.json"
    if not auth_path.exists():
        raise RuntimeError("confirmation has no passing development authorization")
    auth = json.loads(auth_path.read_text())
    if auth["phase"] != phase:
        raise RuntimeError("confirmation phase differs from the authorized candidate")
    gate_path = root / auth["development_gate_path"]
    gate = json.loads(gate_path.read_text())
    freeze_path = root / phase / "development_freeze.json"
    if not gate["passed"] or digest(gate_path) != auth["development_gate_sha256"] or digest(freeze_path) != gate["freeze_sha256"]:
        raise RuntimeError("development gate or freeze changed")
    frozen = json.loads(freeze_path.read_text())
    if digest(root / phase / "selection.json") != frozen["selection_sha256"]:
        raise RuntimeError("selected configuration changed after development")
    if digest(root / phase / "native_selection.json") != frozen["native_selection_sha256"]:
        raise RuntimeError("selected native backend changed after development")
