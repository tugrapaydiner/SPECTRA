"""Dependency-free audit and paired analysis of the trained FP timing matrix.

All timing repetitions remain observations. The bootstrap resamples observed
problems, retaining both fixed checkpoints; it is not a model-population interval.
"""
from __future__ import annotations
from collections import defaultdict
import math
import random
import statistics
from typing import Any

ARMS = ("python", "native", "prepared", "symbolic")
ROUNDS = 7
ORDER_SEED = 9161201
BOOTSTRAP_SEED = 9161202


def correct(puzzle: list[int], answer: list[int]) -> bool:
    if len(puzzle) != 16 or len(answer) != 16 or any(type(x) is not int for x in puzzle + answer):
        raise ValueError("expected integer 4x4 boards")
    if any(x < 0 or x > 4 for x in puzzle):
        raise ValueError("invalid clue symbol")
    if any(a < 1 or a > 4 or (p != 0 and p != a) for p, a in zip(puzzle, answer)):
        return False
    units = [answer[r*4:(r+1)*4] for r in range(4)]
    units += [answer[c::4] for c in range(4)]
    units += [[answer[r*4+c] for r in range(br, br+2) for c in range(bc, bc+2)]
              for br in (0, 2) for bc in (0, 2)]
    return all(set(unit) == {1, 2, 3, 4} for unit in units)


def quantile(values: list[float], q: float) -> float:
    data = sorted(values)
    if not data:
        raise ValueError("empty quantile")
    position = (len(data)-1)*q
    lo = math.floor(position); hi = math.ceil(position)
    return data[lo] + (data[hi]-data[lo])*(position-lo)


def analyze(cases: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {case["case_id"]: case for case in cases}
    if len(by_id) != len(cases) or not cases:
        raise ValueError("empty or duplicated case manifest")
    expected_rows = len(cases)*ROUNDS*len(ARMS)
    if len(rows) != expected_rows:
        raise ValueError("incomplete timing matrix")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    rng = random.Random(ORDER_SEED)
    cursor = 0
    for case in cases:
        for round_id in range(ROUNDS):
            order = list(ARMS); rng.shuffle(order)
            for position, arm in enumerate(order):
                row = rows[cursor]; cursor += 1
                if (row.get("case_id"), row.get("round"), row.get("order"), row.get("arm")) != (
                        case["case_id"], round_id, position, arm):
                    raise ValueError("case/arm/round/order mismatch")
                if type(row.get("elapsed_ns")) is not int or row["elapsed_ns"] <= 0:
                    raise ValueError("timing must be a positive integer nanosecond count")
                if type(row.get("valid")) is not bool or row["valid"] != correct(case["input"], row["answer"]):
                    raise ValueError("independent answer check disagrees")
                steps = row.get("steps")
                if type(steps) is not int or (arm == "symbolic" and steps != 0) or (
                        arm != "symbolic" and not 1 <= steps <= 4):
                    raise ValueError("invalid executed depth")
                work = row.get("work", {})
                if (work.get("executed_steps") != steps or work.get("final_semantic") is not row["valid"] or
                        work.get("target_used") is not False or work.get("semantic_checks") != max(steps, 1) or
                        work.get("block_applications") != steps*2*case["blocks"]):
                    raise ValueError("work accounting mismatch")
                if any(type(work.get(k)) is not int for k in ("executed_steps", "semantic_checks", "block_applications")):
                    raise ValueError("work counters must be integers, not booleans")
                if arm != "python" and (type(work.get("checker_constructions")) is not int or work.get("checker_constructions") != 1):
                    raise ValueError("each solve must construct a fresh checker")
                if arm != "symbolic" and work.get("stop_reason") != (
                        "semantic_valid" if row["valid"] else "budget_exhausted"):
                    raise ValueError("stop reason mismatch")
                groups[(case["case_id"], arm)].append(row)
    medians = {}; case_rows = []; strata: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        cid = case["case_id"]
        for arm in ARMS:
            rr = groups[(cid, arm)]
            if any((r["answer"], r["valid"], r["steps"], r["work"]) !=
                   (rr[0]["answer"], rr[0]["valid"], rr[0]["steps"], rr[0]["work"]) for r in rr):
                raise ValueError("nondeterministic repeated answer/work")
            medians[(cid, arm)] = statistics.median(r["elapsed_ns"] for r in rr)
        ref = groups[(cid, "native")][0]
        for arm in ("python", "prepared"):
            r = groups[(cid, arm)][0]
            if (r["answer"], r["valid"], r["steps"]) != (ref["answer"], ref["valid"], ref["steps"]):
                raise ValueError("neural answer/validity/depth fidelity failed")
        if groups[(cid, "prepared")][0]["work"] != ref["work"]:
            raise ValueError("prepared/reference work dictionaries differ")
        strata[f"{case['family']}:{case['seed']}"].append(cid)
        c = {"case_id": cid, "valid": ref["valid"], "steps": ref["steps"],
             "median_ns": {a: medians[(cid, a)] for a in ARMS}}
        c["prepared_native_ratio"] = medians[(cid, "prepared")]/medians[(cid, "native")]
        c["prepared_native_p95_ratio"] = quantile([r["elapsed_ns"] for r in groups[(cid,"prepared")]],.95)/quantile(
            [r["elapsed_ns"] for r in groups[(cid,"native")]],.95)
        case_rows.append(c)
    def ratio(ids):
        return sum(medians[(cid,"prepared")] for cid in ids)/sum(medians[(cid,"native")] for cid in ids)
    by_stratum = {}
    for key, ids in strata.items():
        by_stratum[key] = {"cases":len(ids), "valid":sum(groups[(cid,"native")][0]["valid"] for cid in ids),
                           "ratio":ratio(ids)}
    clusters: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for c in cases:
        clusters[c["family"]][c["problem_id"]].append(c["case_id"])
    bootstrap = random.Random(BOOTSTRAP_SEED); samples=[]
    totals = {f:[(sum(medians[(cid,"prepared")] for cid in ids),
                  sum(medians[(cid,"native")] for cid in ids)) for ids in problems.values()]
              for f, problems in clusters.items()}
    for _ in range(2000):
        candidate=reference=0
        for cluster in totals.values():
            for _ in cluster:
                a,b = cluster[bootstrap.randrange(len(cluster))];candidate+=a;reference+=b
        samples.append(candidate/reference)
    ci = [quantile(samples,.025),quantile(samples,.975)]
    primary_ratio=ratio(list(by_id))
    gate = (primary_ratio <= .80 and ci[1] < 1 and
            all(x["valid"] > 0 and x["ratio"] <= 1 for x in by_stratum.values()))
    arms = {}
    for arm in ARMS:
        rr=[r for r in rows if r["arm"]==arm]; valid=sum(r["valid"] for r in rr)
        arms[arm] = {"distinct_cases":len(cases),"distinct_valid":sum(groups[(c["case_id"],arm)][0]["valid"] for c in cases),
                     "observations":len(rr),"mean_ns":sum(r["elapsed_ns"] for r in rr)/len(rr),
                     "median_ns":statistics.median(r["elapsed_ns"] for r in rr),
                     "p95_ns_descriptive":quantile([r["elapsed_ns"] for r in rr],.95),
                     "charged_ns_per_verified_answer":sum(r["elapsed_ns"] for r in rr)/valid if valid else None,
                     "summed_case_median_ns":sum(medians[(c["case_id"],arm)] for c in cases)}
    return {"schema":"spectra.trained_fp_analysis.v1", "cases":len(cases),"observations":len(rows),
            "rounds":ROUNDS, "exact_answer_validity_work":True,
            "gate":"PASS" if gate else "FAIL", "primary_ratio":primary_ratio,
            "problem_bootstrap_ci95":ci,"bootstrap_replicates":2000,"strata":by_stratum,
            "case_median_regressions":sum(c["prepared_native_ratio"]>1 for c in case_rows),
            "case_p95_regressions_descriptive":sum(c["prepared_native_p95_ratio"]>1 for c in case_rows),
            "arms":arms,"case_results":case_rows,
            "scope":"previously observed development inputs; two fixed models; warm complete solve; no new learned or solver-superiority claim"}
