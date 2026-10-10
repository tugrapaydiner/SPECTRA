"""Frozen experiment definition; no evaluation inputs are generated on import."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

from data.cnf import CNF, random_3sat
from spectra.cnf import solve_focused, solve_indexed

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("planted3", "uniform3")
SIZES = (128, 512, 1024)
COUNT = 16
SEED_BASE = 202610055000
SEARCH_SEEDS = (17, 73)
ROUNDS = 3
ARMS = ("indexed", "poly", "minbreak", "novelty_break", "glucose4")
MEMORY_ARMS = ("indexed", "novelty_break", "glucose4")
MAX_FLIPS = 2048
CONFLICTS = 2000
SOURCES = (
    "data/__init__.py", "data/cnf.py", "spectra/__init__.py", "spectra/cnf/dimacs.py",
    "eval/__init__.py", "eval/cnf_cached.py",
    "spectra/cnf/__init__.py", "spectra/cnf/state.py", "spectra/cnf/search.py",
    "spectra/cnf/ranked.py", "spectra/cnf/indexed.py", "spectra/cnf/deductive.py",
    "spectra/cnf/focused.py", "experiments/focused_evaluation/study.py",
    "experiments/focused_evaluation/run.py", "experiments/focused_evaluation/analyse.py",
    "experiments/focused_evaluation/PROTOCOL.md",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identities():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}


def check_freeze(freeze):
    require(set(freeze["sources"]) == set(SOURCES), "incomplete source freeze")
    require(freeze["sources"] == identities(), "changed frozen source")
    require(len(freeze["commit"]) == 40 and len(freeze["tree"]) == 40, "invalid frozen identity")


def case_specs():
    return [{"id": f"{family}:{size}:{i}", "family": family, "nvars": size,
             "index": i, "generator_seed": SEED_BASE+j}
            for j, (family, size, i) in enumerate(itertools.product(FAMILIES, SIZES, range(COUNT)))]


def generate(spec):
    problem, _ = random_3sat(spec["nvars"], spec["nvars"]*42//10,
                            spec["generator_seed"], planted=spec["family"] == "planted3")
    return {**spec, "formula": problem.record(), "sha256": problem.sha256()}


def original_violations(formula, witness):
    require(type(witness) in (list, tuple) and len(witness) == formula["nvars"]
            and all(type(v) is bool for v in witness), "malformed Boolean witness")
    return [i for i, clause in enumerate(formula["clauses"])
            if not any(witness[abs(lit)-1] == (lit > 0) for lit in clause)]


def validate_result(case, arm, result):
    require(result["status"] in ("SAT_VERIFIED", "UNKNOWN", "UNSAT_REPORTED"), "invalid status")
    if arm != "glucose4":
        require(result["status"] != "UNSAT_REPORTED", "local search cannot prove UNSAT")
        require(type(result["flips"]) is int and 0 <= result["flips"] <= MAX_FLIPS, "invalid flip count")
        require(result["max_flips"] == MAX_FLIPS, "changed flip budget")
        bad = original_violations(case["formula"], result["witness"])
        require(list(result["unsatisfied"]) == bad, "residual list differs from original clauses")
        require((result["status"] == "SAT_VERIFIED") == (not bad), "false result status")
        if arm != "indexed":
            require(result["policy"] == arm and result["restarts"] == 0, "changed focused policy")
    elif result["status"] == "SAT_VERIFIED":
        require(not original_violations(case["formula"], result["witness"]), "false native SAT")
    if arm == "glucose4":
        require(result["conflict_budget_requested"] == CONFLICTS, "changed native budget")
        require(type(result["stats"]["conflicts"]) is int and result["stats"]["conflicts"] >= 0,
                "invalid native counters")


def call(case, arm, seed):
    """Charge decoding, setup, solving, destruction and independent checking."""
    problem = CNF.from_record(case["formula"])
    if arm == "glucose4":
        from pysat.solvers import Solver
        with Solver(name="glucose4", bootstrap_with=[list(c) for c in problem.clauses]) as solver:
            solver.conf_budget(CONFLICTS)
            answer = solver.solve_limited()
            model = solver.get_model() if answer is True else None
            stats = solver.accum_stats()
        values = {abs(lit): lit > 0 for lit in model} if model is not None else None
        result = {"status": "SAT_VERIFIED" if answer is True else "UNSAT_REPORTED" if answer is False else "UNKNOWN",
                  "witness": [values.get(i+1, False) for i in range(problem.nvars)] if values is not None else None,
                  "stats": stats, "conflict_budget_requested": CONFLICTS}
    elif arm == "indexed":
        result = solve_indexed(problem, seed=seed, max_flips=MAX_FLIPS).record()
    else:
        result = solve_focused(problem, seed=seed, max_flips=MAX_FLIPS, policy=arm).record()
    validate_result(case, arm, result)
    del problem
    return result


def semantic(result):
    return {k: v for k, v in result.items() if k != "elapsed_ns"}


def append(stream, record):
    import os
    stream.write(json.dumps(record, separators=(",", ":"))+"\n")
    stream.flush()
    os.fsync(stream.fileno())
