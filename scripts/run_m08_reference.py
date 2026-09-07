#!/usr/bin/env python3
"""Emit and verify the hand-checkable Milestone-08 reference tree."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Direct execution (`python scripts/run_m08_reference.py`) puts `scripts/` rather
# than the repository root on sys.path. Bind the exact repo root explicitly so the
# documented command works from a clean checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.mcts_reference import run_frozen_reference


def verify(out: dict[str, object]) -> None:
    rows = {tuple(r["path"]): r for r in out["nodes"]}
    expected = {
        (): (4, 2.8),
        (0,): (1, 0.2),
        (1,): (3, 2.6),
        (1, 0): (2, 1.8),
        (1, 1): (0, 0.0),
    }
    for path, (visits, value_sum) in expected.items():
        row = rows[path]
        if row["visits"] != visits or abs(float(row["value_sum"]) - value_sum) > 1e-9:
            raise SystemExit(f"reference mismatch at {path}: {row}")
    if out["best_path"] != [1, 0]:
        raise SystemExit(f"unexpected best path: {out['best_path']}")
    work = out["work"]
    for key, expected_value in {
        "completed_rollouts": 4,
        "expansion_calls": 3,
        "transition_calls": 6,
        "verifier_evaluations": 4,
    }.items():
        if work[key] != expected_value:
            raise SystemExit(f"reference work mismatch {key}: {work[key]} != {expected_value}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = run_frozen_reference()
    verify(result)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
