"""Verify retained inputs/results and compute the frozen descriptive analysis."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import statistics
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.focused_evaluation.study import (
    ARMS, MEMORY_ARMS, ROUNDS, SEARCH_SEEDS, case_specs, check_freeze, generate,
    require, semantic, validate_result,
)

ARTIFACTS = {"LOCK.json", "cases.jsonl", "rows.jsonl", "memory.jsonl"}


def validate_rows(cases, rows):
    by_id = {case["id"]: case for case in cases}
    require(len(by_id) == len(cases), "duplicate case")
    expected = set(itertools.product(by_id, SEARCH_SEEDS, ARMS, range(ROUNDS)))
    seen, trajectories = set(), {}
    for row in rows:
        key = row["id"], row["seed"], row["arm"], row["round"]
        require(key in expected and key not in seen, "unexpected or duplicate observation")
        seen.add(key)
        case = by_id[row["id"]]
        require(row["sha256"] == case["sha256"], "row formula mismatch")
        require(all(type(row[k]) is int and row[k] > 0 for k in ("wall_ns", "cpu_ns")), "invalid timing")
        validate_result(case, row["arm"], row["result"])
        if row["arm"] != "glucose4":
            require(row["result"]["seed"] == row["seed"], "result seed mismatch")
        result = semantic(row["result"])
        require(key[:3] not in trajectories or trajectories[key[:3]] == result,
                "nonrepeatable trajectory")
        trajectories[key[:3]] = result
    require(seen == expected, f"incomplete observation inventory: {len(expected-seen)} missing")


def load(path):
    if path.is_dir():
        read = lambda name: (path/name).read_bytes()
        names = {p.name for p in path.iterdir() if p.is_file()}
        archive = None
    else:
        archive = zipfile.ZipFile(path)
        require(len(archive.namelist()) == len(set(archive.namelist())), "duplicate archive entry")
        names, read = set(archive.namelist()), archive.read
    try:
        require(names == ARTIFACTS | {"MANIFEST.json"}, "unexpected or incomplete artifact inventory")
        manifest = json.loads(read("MANIFEST.json"))
        require(manifest["status"] == "COMPLETE" and set(manifest["files"]) == ARTIFACTS,
                "incomplete artifact bindings")
        for name, identity in manifest["files"].items():
            data = read(name)
            require(len(data) == identity["bytes"] and hashlib.sha256(data).hexdigest() == identity["sha256"],
                    f"artifact identity mismatch: {name}")
        lock = json.loads(read("LOCK.json"))
        check_freeze(lock["freeze"])
        require(lock["python_sat"] == "1.9.dev15", "native version mismatch")
        cases, rows, memory = ([json.loads(line) for line in read(name).splitlines()]
                               for name in ("cases.jsonl", "rows.jsonl", "memory.jsonl"))
        specs = case_specs()
        require(len(cases) == len(specs), "incomplete case inventory")
        require(len({c["sha256"] for c in cases}) == len(cases), "duplicate formula")
        for spec, case in zip(specs, cases):
            require(all(case[k] == v for k, v in spec.items()), "case specification mismatch")
            require(case == generate(spec), "retained formula differs from declared generator")
            f = case["formula"]
            require(f["nvars"] == case["nvars"] and len(f["clauses"]) == f["nvars"]*42//10,
                    "formula dimensions mismatch")
            require(all(len(c) == 3 and len({abs(lit) for lit in c}) == 3
                        and all(type(lit) is int and 1 <= abs(lit) <= f["nvars"] for lit in c)
                        for c in f["clauses"]), "invalid 3-SAT formula")
            require(len({tuple(c) for c in f["clauses"]}) == len(f["clauses"]), "duplicate clause")
            payload = json.dumps([f["nvars"], f["clauses"]], separators=(",", ":")).encode("ascii")
            require(hashlib.sha256(b"spectra.cnf.v1\0"+payload).hexdigest() == case["sha256"], "formula digest")
        validate_rows(cases, rows)
        expected = {(c["id"], a) for c in cases if c["index"] < 2 for a in MEMORY_ARMS}
        require(len(memory) == len(expected) and {(m["id"], m["arm"]) for m in memory} == expected,
                "incomplete memory inventory")
        for m in memory:
            require(type(m["baseline_rss_kib"]) is int and m["peak_rss_kib"] >= m["baseline_rss_kib"] > 0,
                    "invalid memory observation")
            require((m["python_peak_bytes"] is None) if m["arm"] == "glucose4" else
                    (type(m["python_peak_bytes"]) is int and m["python_peak_bytes"] > 0), "invalid allocation")
            result = next(r["result"] for r in rows if r["id"] == m["id"] and r["arm"] == m["arm"]
                          and r["seed"] == SEARCH_SEEDS[0])
            require(m["status"] == result["status"], "memory/timing outcome mismatch")
        return lock, cases, rows, memory
    finally:
        if archive is not None:
            archive.close()


def p95(values):
    return sorted(values)[(95*len(values)+99)//100-1]


def metrics(rows):
    out = {}
    for arm in ARMS:
        group = [r for r in rows if r["arm"] == arm]
        unique = {(r["id"], r["seed"]): r["result"]["status"] for r in group}
        wall, cpu = [[r[k]/1e6 for r in group] for k in ("wall_ns", "cpu_ns")]
        out[arm] = {"pairs": len(unique), "sat_verified": sum(s == "SAT_VERIFIED" for s in unique.values()),
                    "unknown": sum(s == "UNKNOWN" for s in unique.values()),
                    "unsat_reported": sum(s == "UNSAT_REPORTED" for s in unique.values()),
                    "formulas_solved": len({k[0] for k, s in unique.items() if s == "SAT_VERIFIED"}),
                    "wall_mean_ms": statistics.mean(wall), "wall_p95_ms": p95(wall),
                    "cpu_mean_ms": statistics.mean(cpu), "cpu_p95_ms": p95(cpu),
                    "observed_deadline_solved": {str(ms): sum(
                        status == "SAT_VERIFIED" and statistics.median(r["wall_ns"] for r in group
                        if (r["id"], r["seed"]) == key) <= ms*1e6 for key, status in unique.items())
                        for ms in (5, 20, 100)}}
    return out


def paired_bootstrap(cases, rows):
    clusters = {}
    for case in cases:
        pair = {}
        for arm in ("indexed", "novelty_break"):
            group = [r for r in rows if r["id"] == case["id"] and r["arm"] == arm]
            pair[arm] = (sum(r["result"]["status"] == "SAT_VERIFIED" for r in group if r["round"] == 0),
                         statistics.mean(r["wall_ns"] for r in group))
        clusters.setdefault((case["family"], case["nvars"]), []).append(pair)
    rng = random.Random(20261005017)
    deltas, ratios = [], []
    for _ in range(2000):
        sample = [rng.choice(group) for group in clusters.values() for _ in group]
        deltas.append(sum(p["novelty_break"][0]-p["indexed"][0] for p in sample)/(2*len(sample)))
        ratios.append(sum(p["novelty_break"][1] for p in sample)/sum(p["indexed"][1] for p in sample))
    return {"unit": "formula, paired and stratified by family/size; seeds/rounds clustered",
            "resamples": 2000, "sat_rate_difference_95": [sorted(deltas)[49], sorted(deltas)[1949]],
            "wall_mean_ratio_95": [sorted(ratios)[49], sorted(ratios)[1949]]}


def summary(cases, rows, memory):
    overall = metrics(rows)
    cells = {f"{family}:{size}": metrics([r for r in rows if r["id"].startswith(f"{family}:{size}:")])
             for family, size in sorted({(c["family"], c["nvars"]) for c in cases})}
    paired = {}
    lookup = {(r["id"], r["seed"], r["arm"]): r["result"]["status"] == "SAT_VERIFIED"
              for r in rows if r["round"] == 0}
    for arm in ARMS[1:]:
        wins = sum(lookup[c["id"], s, arm] and not lookup[c["id"], s, "indexed"] for c in cases for s in SEARCH_SEEDS)
        losses = sum(lookup[c["id"], s, "indexed"] and not lookup[c["id"], s, arm] for c in cases for s in SEARCH_SEEDS)
        paired[arm] = {"wins": wins, "losses": losses}
    checks = {"additional_sat_at_least_five": overall["novelty_break"]["sat_verified"] >= overall["indexed"]["sat_verified"]+5,
              "mean_wall_no_higher": overall["novelty_break"]["wall_mean_ms"] <= overall["indexed"]["wall_mean_ms"],
              "no_cell_loses_more_than_two": all(v["novelty_break"]["sat_verified"] >= v["indexed"]["sat_verified"]-2 for v in cells.values()),
              "all_sat_witnesses_valid": True}
    memory_summary = {}
    for arm in MEMORY_ARMS:
        group = [m for m in memory if m["arm"] == arm]
        memory_summary[arm] = {"formulas": len(group),
            "rss_mean_mib": statistics.mean(m["peak_rss_kib"]/1024 for m in group),
            "rss_max_mib": max(m["peak_rss_kib"]/1024 for m in group),
            "python_peak_mean_bytes": None if arm == "glucose4" else statistics.mean(m["python_peak_bytes"] for m in group)}
    return {"status": "VERIFIED", "formulas": len(cases), "timing_rows": len(rows), "memory_rows": len(memory),
            "gate": "PASS" if all(checks.values()) else "FAIL", "gate_checks": checks,
            "overall": overall, "cells": cells, "paired_against_indexed": paired,
            "uncertainty": paired_bootstrap(cases, rows), "memory": memory_summary,
            "native_conflict_max": max(r["result"]["stats"]["conflicts"] for r in rows if r["arm"] == "glucose4"),
            "native_conflict_overshoot_rows": sum(r["result"]["stats"]["conflicts"] > 2000 for r in rows if r["arm"] == "glucose4"),
            "scope": "fresh synthetic panel, one CPU host; no UNSAT proofs or general-intelligence claims"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("evidence", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    lock, cases, rows, memory = load(args.evidence)
    result = summary(cases, rows, memory)
    result["frozen"] = lock["freeze"]
    payload = json.dumps(result, indent=2)+"\n"
    if args.out:
        with args.out.open("x") as stream:
            stream.write(payload)
    print(payload)


if __name__ == "__main__":
    main()
