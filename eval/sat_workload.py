"""Bounded classical SAT workload triage with independent witness validation.

PySAT is optional and imported only in isolated solver workers. Conflict budgets
are solver-specific work limits, not equal wall time or portable FLOP budgets.
UNSAT reports are never labeled independently certified without a checked proof.
"""
from __future__ import annotations

from collections import Counter
import importlib.metadata
import json
import math
import multiprocessing as mp
import statistics
import time
import traceback

from data.cnf import CNF, random_3sat

STATUSES = {"SAT_VERIFIED", "UNSAT_REPORTED", "UNKNOWN_BUDGET", "TIMEOUT", "ERROR"}
SOLVERS = {"cadical195", "glucose4"}
ORDER = "round then alternating solver order then all declared inputs"


def validate_protocol(p: dict) -> None:
    if type(p) is not dict or p.get("schema") != "spectra.sat_workload_protocol.v1":
        raise ValueError("unsupported workload protocol")
    for key in ("instances_per_tier", "rounds", "conflict_budget", "checker_repeats",
                "minimum_instances_per_tier", "minimum_hard_sat_cases",
                "clause_ratio_numerator", "clause_ratio_denominator"):
        if type(p.get(key)) is not int or p[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if (p.get("families") != ["uniform_3sat", "planted_3sat"]
            or type(p.get("solvers")) is not list or len(p["solvers"]) != 2
            or any(type(s) is not str for s in p["solvers"])
            or set(p["solvers"]) != SOLVERS or p.get("execution_order") != ORDER):
        raise ValueError("both distributions, distinct declared solvers and explicit order required")
    sizes = p.get("variable_counts")
    if (type(sizes) is not list or not sizes or any(type(n) is not int or not 3 <= n <= 2**64 for n in sizes)
            or len(set(sizes)) != len(sizes)):
        raise ValueError("distinct valid 3-SAT sizes required")
    if (type(p.get("seed_start")) is not int or p["seed_start"] < 0
            or p["seed_start"] + 2*len(sizes)*p["instances_per_tier"] > 2**64):
        raise ValueError("invalid generator seed range")
    for key in ("process_timeout_seconds", "minimum_hard_sat_fraction",
                "minimum_complete_solve_ms", "minimum_solve_to_check_ratio"):
        if (type(p.get(key)) not in (int, float) or not math.isfinite(p[key]) or p[key] <= 0):
            raise ValueError(f"{key} must be positive and finite")
    if (p["minimum_hard_sat_fraction"] > 1 or type(p.get("python_sat_version")) is not str
            or not p["python_sat_version"]):
        raise ValueError("invalid gate fraction or package version")


def generate_cases(protocol: dict) -> list[dict]:
    validate_protocol(protocol)
    cases, seen = [], set()
    for family in protocol["families"]:
        for n in protocol["variable_counts"]:
            m = n*protocol["clause_ratio_numerator"] // protocol["clause_ratio_denominator"]
            for index in range(protocol["instances_per_tier"]):
                seed = protocol["seed_start"] + len(cases)
                formula, planted = random_3sat(n, m, seed, planted=family == "planted_3sat")
                normalized = formula.sha256(normalize_order=True)
                if normalized in seen:
                    raise ValueError("duplicate formula: do not silently replace a declared input")
                seen.add(normalized)
                cases.append({"case_id": f"{family}:{n}:{index}", "family": family,
                    "seed": seed, "formula": formula.record(),
                    "ordered_sha256": formula.sha256(), "normalized_sha256": normalized,
                    "generation_witness": None if planted is None else list(planted)})
    return cases


def execution_schedule(protocol: dict, cases: list[dict]):
    """Deterministic full inventory; alternate solver order to expose time drift."""
    for round_id in range(protocol["rounds"]):
        solvers = protocol["solvers"] if round_id % 2 == 0 else protocol["solvers"][::-1]
        for solver in solvers:
            for case in cases:
                yield case, solver, round_id


def model_to_witness(problem: CNF, model: list[int]) -> tuple[bool, ...]:
    if type(model) is not list:
        raise ValueError("solver must return a literal-list model")
    values = {}
    for lit in model:
        if type(lit) is not int or not 1 <= abs(lit) <= problem.nvars or abs(lit) in values:
            raise ValueError("invalid, duplicated or out-of-range model literal")
        values[abs(lit)] = lit > 0
    used = {abs(lit) for clause in problem.clauses for lit in clause}
    if not used <= values.keys():
        raise ValueError("solver omitted a referenced variable")
    # Only genuinely unused variables can be completed arbitrarily.
    return tuple(values.get(i+1, False) for i in range(problem.nvars))


def independent_check(problem: CNF, witness: tuple[bool, ...]) -> bool:
    """Separate signed-literal-set semantics; no incremental state or solver."""
    problem.validate_witness(witness)
    true_literals = {i+1 if value else -i-1 for i, value in enumerate(witness)}
    return all(bool(set(clause) & true_literals) for clause in problem.clauses)


def _worker(connection, formula: dict, name: str, budget: int, repeats: int,
            expected_version: str) -> None:
    try:
        actual_version = importlib.metadata.version("python-sat")
        if actual_version != expected_version:
            raise RuntimeError(f"expected python-sat {expected_version}, got {actual_version}")
        from pysat.solvers import Solver
        problem = CNF.from_record(formula)
        start = time.perf_counter_ns()
        with Solver(name=name, bootstrap_with=[list(c) for c in problem.clauses]) as solver:
            solver.conf_budget(budget)
            setup_end = time.perf_counter_ns()
            result = solver.solve_limited()
            solve_end = time.perf_counter_ns()
            if result is not None and type(result) is not bool:
                raise RuntimeError("unexpected solver result type")
            witness = model_to_witness(problem, solver.get_model()) if result is True else None
            stats = solver.accum_stats()
        check_start = time.perf_counter_ns()
        if witness is not None and not problem.satisfied(witness):
            raise RuntimeError("SAT solver returned a false witness")
        complete_end = time.perf_counter_ns()
        checks = []
        if witness is not None:
            for _ in range(repeats):
                t = time.perf_counter_ns()
                valid = problem.satisfied(witness)
                checks.append(time.perf_counter_ns()-t)
                if not valid:
                    raise RuntimeError("nonrepeatable witness check")
        connection.send({"status": "SAT_VERIFIED" if result is True else
                         "UNSAT_REPORTED" if result is False else "UNKNOWN_BUDGET",
            "witness": None if witness is None else list(witness),
            "setup_ns": setup_end-start, "solver_call_ns": solve_end-setup_end,
            "complete_ns": complete_end-start, "first_check_ns": complete_end-check_start,
            "checker_ns": checks, "solver_stats": stats, "python_sat_version": actual_version})
    except Exception:
        connection.send({"status": "ERROR", "error": traceback.format_exc(),
                         "witness": None, "complete_ns": None, "checker_ns": []})
    finally:
        connection.close()


def run_case(case: dict, solver: str, round_id: int, protocol: dict) -> dict:
    """Contain each solve in a process; report supervision time independently.

    The deadline includes spawn/import/IPC, but is not a hard real-time guarantee:
    OS scheduling, process startup and termination can overrun the requested time.
    Timed-out runs have no invented in-process duration or successful outcome.
    Only the formula (never its planted solution) is passed to the worker.
    """
    validate_protocol(protocol)
    if solver not in protocol["solvers"] or type(round_id) is not int or not 0 <= round_id < protocol["rounds"]:
        raise ValueError("undeclared solver or round")
    problem = CNF.from_record(case["formula"])
    if problem.sha256() != case["ordered_sha256"]:
        raise ValueError("case input hash mismatch")
    ctx = mp.get_context("spawn")
    receive, send = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_worker, args=(send, problem.record(), solver,
        protocol["conflict_budget"], protocol["checker_repeats"], protocol["python_sat_version"]))
    start = time.perf_counter_ns()
    try:
        process.start()
        send.close()
        remaining = max(0.0, protocol["process_timeout_seconds"] -
                        (time.perf_counter_ns()-start)/1e9)
        if receive.poll(remaining):
            try:
                result = receive.recv()
            except EOFError:
                result = {"status": "ERROR", "error": "worker exited without a record",
                          "witness": None, "complete_ns": None, "checker_ns": []}
        else:
            result = {"status": "TIMEOUT", "witness": None, "complete_ns": None, "checker_ns": []}
    except Exception:
        result = {"status": "ERROR", "error": traceback.format_exc(),
                  "witness": None, "complete_ns": None, "checker_ns": []}
    finally:
        send.close()
        receive.close()
        if process.pid is not None:
            process.join(timeout=0.1)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()
    result.update(case_id=case["case_id"], solver=solver, round=round_id,
        ordered_sha256=case["ordered_sha256"], conflict_budget_requested=protocol["conflict_budget"],
        process_timeout_seconds=protocol["process_timeout_seconds"],
        supervised_wall_ns=time.perf_counter_ns()-start)
    return result


def _positive_ns(value) -> bool:
    return type(value) is int and value > 0


def summarize(protocol: dict, cases: list[dict], rows: list[dict]) -> dict:
    """Require every declared input/solver/round and validate every SAT witness."""
    validate_protocol(protocol)
    expected_cases = generate_cases(protocol)
    if json.dumps(cases, sort_keys=True, allow_nan=False) != json.dumps(expected_cases, sort_keys=True):
        raise ValueError("case inventory/seed/formula/witness differs from the declared generator")
    lookup = {c["case_id"]: c for c in cases}
    expected = {(key, solver, r) for key in lookup for solver in protocol["solvers"]
                for r in range(protocol["rounds"])}
    found = {}
    for row in rows:
        if type(row) is not dict:
            raise ValueError("observation must be an object")
        key = row.get("case_id"), row.get("solver"), row.get("round")
        if (type(key[0]) is not str or type(key[1]) is not str or type(key[2]) is not int
                or key not in expected or key in found):
            raise ValueError("duplicate or undeclared solver/round observation")
        case = lookup[key[0]]
        if row.get("ordered_sha256") != case["ordered_sha256"]:
            raise ValueError("observation input hash mismatch")
        if (type(row.get("conflict_budget_requested")) is not int
                or type(row.get("process_timeout_seconds")) not in (int, float)
                or row["conflict_budget_requested"] != protocol["conflict_budget"]
                or row["process_timeout_seconds"] != protocol["process_timeout_seconds"]):
            raise ValueError("observation budget mismatch")
        status = row.get("status")
        if type(status) is not str or status not in STATUSES or not _positive_ns(row.get("supervised_wall_ns")):
            raise ValueError("invalid status or supervising duration")
        if status == "ERROR":
            raise ValueError("solver error is failed evidence, not a negative scientific gate")
        if status == "TIMEOUT":
            if row.get("complete_ns") is not None or row.get("witness") is not None or row.get("checker_ns") != []:
                raise ValueError("timeout must not invent a solve duration or witness")
        else:
            if (not all(_positive_ns(row.get(k)) for k in
                        ("setup_ns", "solver_call_ns", "complete_ns", "first_check_ns"))
                    or row["setup_ns"]+row["solver_call_ns"]+row["first_check_ns"] > row["complete_ns"]
                    or row["complete_ns"] > row["supervised_wall_ns"]
                    or row.get("python_sat_version") != protocol["python_sat_version"]):
                raise ValueError("invalid or inconsistent full-cost timing/package record")
            stats = row.get("solver_stats")
            if (type(stats) is not dict or not {"conflicts", "decisions", "propagations", "restarts"} <= stats.keys()
                    or any(type(v) is not int or v < 0 for v in stats.values())):
                raise ValueError("invalid solver work counters")
            if status == "SAT_VERIFIED":
                witness, checks = row.get("witness"), row.get("checker_ns")
                if (type(witness) is not list or type(checks) is not list
                        or len(checks) != protocol["checker_repeats"] or not all(map(_positive_ns, checks))
                        or not independent_check(CNF.from_record(case["formula"]), tuple(witness))):
                    raise ValueError("invalid SAT witness or checker timing inventory")
            elif row.get("witness") is not None or row.get("checker_ns") != []:
                raise ValueError("non-SAT observation must not carry a witness or checker timing")
        found[key] = row
    if set(found) != expected:
        raise ValueError("incomplete observation inventory")
    actual_order = [(row["case_id"], row["solver"], row["round"]) for row in rows]
    intended_order = [(case["case_id"], solver, r)
                      for case, solver, r in execution_schedule(protocol, cases)]
    if actual_order != intended_order:
        raise ValueError("observation order differs from declared execution schedule")
    per_case = []
    for case in cases:
        observations = [found[(case["case_id"], s, r)] for s in protocol["solvers"] for r in range(protocol["rounds"])]
        statuses = {row["status"] for row in observations}
        if "UNSAT_REPORTED" in statuses and ("SAT_VERIFIED" in statuses or case["generation_witness"] is not None):
            raise ValueError("UNSAT report contradicts an independently valid SAT witness")
        all_sat = statuses == {"SAT_VERIFIED"}
        complete, checker, ratio = None, None, None
        if all_sat:
            complete = min(statistics.median(found[(case["case_id"], s, r)]["complete_ns"]
                           for r in range(protocol["rounds"])) for s in protocol["solvers"])
            checker = max(statistics.median(ns for r in range(protocol["rounds"])
                          for ns in found[(case["case_id"], s, r)]["checker_ns"]) for s in protocol["solvers"])
            ratio = complete/checker
        hard = bool(all_sat and complete >= protocol["minimum_complete_solve_ms"]*1e6
                    and ratio >= protocol["minimum_solve_to_check_ratio"])
        per_case.append({"case_id": case["case_id"], "family": case["family"],
            "nvars": case["formula"]["nvars"], "all_solver_rounds_sat_verified": all_sat,
            "status_inventory": dict(Counter(row["status"] for row in observations)),
            "fastest_solver_median_complete_ns": complete, "slowest_checker_median_ns": checker,
            "conservative_solve_to_check_ratio": ratio, "hard_witnessed_sat": hard})
    tiers = []
    for family in protocol["families"]:
        for n in protocol["variable_counts"]:
            subset = [c for c in per_case if c["family"] == family and c["nvars"] == n]
            hard = [c["case_id"] for c in subset if c["hard_witnessed_sat"]]
            passed = (len(subset) >= protocol["minimum_instances_per_tier"]
                      and len(hard) >= protocol["minimum_hard_sat_cases"]
                      and len(hard)/len(subset) >= protocol["minimum_hard_sat_fraction"])
            tiers.append({"family": family, "nvars": n, "instances": len(subset),
                "all_solver_rounds_sat_verified": sum(c["all_solver_rounds_sat_verified"] for c in subset),
                "hard_sat_cases": len(hard), "hard_sat_case_ids": hard,
                "hard_sat_fraction_of_all_generated": len(hard)/len(subset),
                "gate": "FEASIBILITY_LEAD_ONLY" if passed else "NO_FEASIBLE_TIER"})
    return {"schema": "spectra.sat_workload_summary.v1", "instances": len(cases),
        "observations": len(rows), "sat_witnesses_independently_checked": sum(r["status"] == "SAT_VERIFIED" for r in rows),
        "statuses": dict(sorted(Counter(r["status"] for r in rows).items())),
        "tiers": tiers, "per_case": per_case, "learned_model_evaluated": False,
        "independent_confirmation": False, "unsat_proofs_verified": False,
        "energy_joules": None, "novelty_established": False,
        "gate_scope": "default-solver triage; not a learned advantage, tuned portfolio or power analysis"}
