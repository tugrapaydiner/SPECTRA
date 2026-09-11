"""Fixed-pool selection accounting and crossed seed/common-example inference.

Every selector sees exactly the same ordered candidates. This isolates selection
from exploration; it does not measure a complete-solve performance advantage.
Uniform-random results are analytic expectations, not fabricated random trials.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np

from eval.verified_search import ValueTarget

TARGET_NAMES = tuple(t.value for t in (ValueTarget.IMPROVEMENT, ValueTarget.TERMINAL, ValueTarget.QUALITY))
GATE = {"min_coverage": 0.20, "min_quality_minus_improvement": 0.05,
        "min_ci_lower_strict": 0.0, "bootstrap_replicates": 2000, "bootstrap_seed": 17091}


def score_pool(record: dict) -> dict:
    flags = record["validity"]
    depths, actions = record["depths"], record["actions"]
    n = len(flags)
    if not n or any(type(v) is not bool for v in flags) or len(depths) != n or len(actions) != n:
        raise ValueError("malformed fixed-pool candidate inventory")
    if any(type(d) is not int or d < 1 for d in depths) or any(type(a) is not int or a < 0 for a in actions):
        raise ValueError("invalid candidate depth/action identity")
    if len(set(zip(depths, actions))) != n:
        raise ValueError("duplicate fixed-pool candidate identity")
    selectors = {}
    for name in TARGET_NAMES:
        scores = np.asarray(record["scores"][name], dtype=np.float64)
        if scores.shape != (n,) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError("invalid probability/quality score vector")
        selectors[name] = int(np.argmax(scores))
    event = np.asarray(record["improvement_labels"], dtype=np.float64)
    if event.shape != (n,) or not np.isin(event, [0.0, 1.0]).all():
        raise ValueError("improvement oracle needs the binary declared event")
    selectors["oracle_improvement"] = int(np.argmax(event))
    selectors["exact_first_valid"] = flags.index(True) if any(flags) else 0
    for depth in (1, 4):
        indices = [i for i, (d, a) in enumerate(zip(depths, actions)) if d == depth and a == 0]
        if len(indices) != 1:
            raise ValueError("pool must contain each named identity-spine comparator exactly once")
        selectors[f"identity_depth{depth}"] = indices[0]
    successes = {name: float(flags[index]) for name, index in selectors.items()}
    successes["uniform_random_expectation"] = sum(flags)/n
    return {**record, "covered": any(flags), "selected_indices": selectors, "successes": successes}


def summarize_pools(rows: list[dict], *, replicates: int = 2000, seed: int = 17091) -> dict:
    if not rows or type(replicates) is not int or replicates < 1:
        raise ValueError("nonempty pools and positive bootstrap count required")
    groups = defaultdict(dict)
    for raw in rows:
        row = score_pool(raw)  # Recompute rather than trusting stored selections.
        core, example = row["core_seed"], row["example_id"]
        if type(core) is not int or not isinstance(example, str) or not example:
            raise ValueError("invalid core/example identity")
        if example in groups[core]:
            raise ValueError("duplicate model-example pool")
        groups[core][example] = row
    seeds = sorted(groups)
    examples = sorted(groups[seeds[0]])
    if any(set(groups[s]) != set(examples) for s in seeds):
        raise ValueError("crossed bootstrap requires common example identities across all seeds")
    summary = {}
    covered = sum(r["covered"] for g in groups.values() for r in g.values())
    n = len(rows)
    names = tuple(next(iter(groups[seeds[0]].values()))["successes"])
    for name in names:
        returned = sum(r["successes"][name] for g in groups.values() for r in g.values())
        summary[name] = {"model_example_pools": n, "covered": covered,
                         "returned_valid_or_expected": returned,
                         "selection_failures_or_expected": covered-returned,
                         "success": returned/n, "coverage": covered/n,
                         "conditional_selection_reliability": returned/covered if covered else None,
                         "analytic_expectation": name == "uniform_random_expectation"}
    quality, improvement = ValueTarget.QUALITY.value, ValueTarget.IMPROVEMENT.value
    differences = np.asarray([[groups[s][i]["successes"][quality] -
                               groups[s][i]["successes"][improvement]
                               for i in examples] for s in seeds], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        si = rng.integers(len(seeds), size=len(seeds))
        ii = rng.integers(len(examples), size=len(examples))
        draws.append(float(differences[np.ix_(si, ii)].mean()))
    lo, hi = np.quantile(draws, [.025, .975]).tolist()
    effect = float(differences.mean())
    passed = covered/n >= GATE["min_coverage"] and effect >= GATE["min_quality_minus_improvement"] and lo > 0.0
    return {"model_seeds": seeds, "unique_examples": len(examples), "model_example_pools": n,
            "selectors": summary, "quality_minus_improvement": effect, "ci95": [lo, hi],
            "coverage": covered/n, "gate_pass": passed, "gate": dict(GATE),
            "actual_bootstrap_replicates": replicates, "actual_bootstrap_seed": seed,
            "scope": "selection on common candidates, not closed-loop speed or cross-task weight transfer"}
