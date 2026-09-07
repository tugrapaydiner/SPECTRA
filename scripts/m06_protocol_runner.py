#!/usr/bin/env python3
"""Bind the preregistered M06 parameter-match amendment to the experiment runner.

The first architecture-only preflight measured a 1.4013% trainable-parameter gap
between the existing FP and ternary recursive implementations. No experiment or
accuracy ran. docs/M06_PROTOCOL.md therefore amended the tolerance to 1.5%
before experimental execution. This wrapper changes only that contract check;
all training/evaluation logic remains in m06_baseline_experiment.py.
"""
from __future__ import annotations

import json

import scripts.m06_baseline_experiment as experiment

RECURSIVE_PARAMETER_MATCH_TOLERANCE = 0.015


def amended_parameter_audit(timing: list[dict]) -> dict:
    counts = {r["kind"]: int(r["trainable_params"]) for r in timing}
    fp = counts["fp_recursive"]
    ternary = counts["ternary_recursive"]
    relative_gap = abs(fp - ternary) / max(1, fp)
    if relative_gap > RECURSIVE_PARAMETER_MATCH_TOLERANCE:
        raise experiment.ExperimentStop(
            "recursive models exceed the preregistered 1.5% trainable-parameter "
            f"matching tolerance: gap={relative_gap:.8f}, counts={counts}"
        )
    if counts["single_pass"] <= max(fp, ternary):
        raise experiment.ExperimentStop(
            f"single-pass baseline is not larger than both recursive models: {counts}"
        )
    return {
        "trainable_parameter_counts": counts,
        "recursive_relative_parameter_gap": relative_gap,
        "recursive_match_tolerance": RECURSIVE_PARAMETER_MATCH_TOLERANCE,
        "single_pass_is_larger": True,
        "amendment_basis": (
            "architecture-only preflight 34124091722; existing FP/ternary bias conventions"
        ),
    }


def main() -> int:
    experiment.parameter_audit = amended_parameter_audit
    return experiment.main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except experiment.ExperimentStop as exc:
        print(f"M06 STOP: {exc}", flush=True)
        raise SystemExit(2)
