"""Standard-library auditing for the fixed four-arm compiler comparison.

Compiler numerical differences are observations, not errors to suppress. Timing
and task-preservation claims remain distinct from exact trajectory fidelity.
"""
from __future__ import annotations
from collections import defaultdict
import math
import random
import statistics
from typing import Any
from spectra.fp_evidence import correct, quantile

ARMS = ("eager", "prepared", "inductor_default", "inductor_frozen")
ROUNDS = 7
ORDER_SEED = 9161227
BOOTSTRAP_SEED = 9161228
WORK_KEYS = {"executed_steps", "block_applications", "semantic_checks", "checker_constructions",
             "final_semantic", "target_used", "checker", "stop_reason"}


def schedule(cases):
    rng = random.Random(ORDER_SEED)
    for c in cases:
        for r in range(ROUNDS):
            arms = list(ARMS)
            rng.shuffle(arms)
            for order, arm in enumerate(arms):
                yield c, r, order, arm


def audit_work(case, row):
    if type(row.get("valid")) is not bool or row["valid"] != correct(case["input"], row["answer"]):
        raise ValueError("original-clue answer validity mismatch")
    w = row.get("work", {})
    if set(w) != WORK_KEYS:
        raise ValueError("incomplete or unknown work fields")
    for key in ("executed_steps", "block_applications", "semantic_checks", "checker_constructions"):
        if type(w[key]) is not int:
            raise ValueError("work counters must be integers, not booleans")
    n = w["executed_steps"]
    if (not 1 <= n <= 4 or w["semantic_checks"] != n or w["checker_constructions"] != 1
            or w["block_applications"] != n * 2 * case["blocks"]
            or w["final_semantic"] is not row["valid"] or w["target_used"] is not False
            or w["checker"] != "native_exact_sudoku_v1"
            or w["stop_reason"] != ("semantic_valid" if row["valid"] else "budget_exhausted")
            or (not row["valid"] and n != 4)):
        raise ValueError("invalid work/exit contract")


def analyze(cases: list[dict[str, Any]], rows: list[dict[str, Any]],
            traces: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {c["case_id"]: c for c in cases}
    if not cases or len(by_id) != len(cases):
        raise ValueError("empty or duplicate cases")
    problems = {}
    for c in cases:
        correct(c["input"], c["input"])
        if type(c["seed"]) is not int or type(c["blocks"]) is not int or c["blocks"] < 1:
            raise ValueError("invalid model identity")
        key = (c["family"], c["problem_id"])
        if key in problems and problems[key] != c["input"]:
            raise ValueError("one problem identity has different inputs")
        problems[key] = c["input"]
    if len(rows) != len(cases)*ROUNDS*len(ARMS):
        raise ValueError("incomplete timing matrix")
    grouped = defaultdict(list)
    for row, (case, r, position, arm) in zip(rows, schedule(cases)):
        if (row.get("case_id"), row.get("round"), row.get("order"), row.get("arm")) != (
                case["case_id"], r, position, arm):
            raise ValueError("case/arm/round/order mismatch")
        if any(type(row.get(k)) is not int for k in ("round", "order", "elapsed_ns")) or row["elapsed_ns"] <= 0:
            raise ValueError("non-integer or nonpositive timing/index")
        audit_work(case, row)
        rr = grouped[(case["case_id"], arm)]
        if rr and (row["answer"], row["work"], row["valid"]) != (rr[0]["answer"], rr[0]["work"], rr[0]["valid"]):
            raise ValueError("answer or complete work record changes between rounds")
        rr.append(row)
    trajectory = {}
    if len(traces) != len(cases)*len(ARMS):
        raise ValueError("incomplete trajectory coverage")
    for t in traces:
        key = (t.get("case_id"), t.get("arm"))
        if key in trajectory or key[0] not in by_id or key[1] not in ARMS:
            raise ValueError("duplicate/unknown trajectory")
        if len(t.get("steps", [])) != 4 or type(t.get("embedding_bitwise")) is not bool:
            raise ValueError("incomplete full-budget trace")
        for h in [t.get("embedding_sha256")] + [s.get(k) for s in t["steps"] for k in ("y_sha256", "z_sha256", "logits_sha256")]:
            if not isinstance(h, str) or len(h) != 64 or any(ch not in "0123456789abcdef" for ch in h):
                raise ValueError("invalid tensor digest")
        for s in t["steps"]:
            if any(type(s.get(k)) is not bool for k in ("bitwise", "finite")):
                raise ValueError("invalid fidelity flag")
            for k in ("max_abs_y", "max_abs_z", "max_abs_logits"):
                v = s.get(k)
                if v is not None and (type(v) not in (int, float) or not math.isfinite(v) or v < 0):
                    raise ValueError("invalid numerical difference")
                if s["finite"] and v is None:
                    raise ValueError("finite trace lacks difference")
        trajectory[key] = t
    med = {k: statistics.median([r["elapsed_ns"] for r in rr]) for k, rr in grouped.items()}
    first = {k: rr[0] for k, rr in grouped.items()}
    for c in cases:
        cid = c["case_id"]
        a, b = first[(cid,"eager")], first[(cid,"prepared")]
        if (a["answer"], a["work"]) != (b["answer"], b["work"]):
            raise ValueError("existing exact reference fidelity failed")
        for arm in ("eager", "prepared"):
            tr = trajectory[(cid, arm)]
            if not tr["embedding_bitwise"] or not all(s["bitwise"] and s["finite"] for s in tr["steps"]):
                raise ValueError("existing exact trajectory fidelity failed")
        # Derived bitwise flags must agree with the stored tensor digests.
        ref = trajectory[(cid, "eager")]
        for arm in ARMS:
            tr = trajectory[(cid, arm)]
            if tr["embedding_bitwise"] != (tr["embedding_sha256"] == ref["embedding_sha256"]):
                raise ValueError("embedding bit flag differs from hashes")
            for aa, bb in zip(tr["steps"], ref["steps"]):
                same = all(aa[k] == bb[k] for k in ("y_sha256", "z_sha256", "logits_sha256"))
                if aa["bitwise"] != same:
                    raise ValueError("step bit flag differs from hashes")
    totals = {a: sum(med[(c["case_id"], a)] for c in cases) for a in ARMS}
    summary = {"scope": "fixed development cases; warm complete solve; not a generalization claim",
               "cases": len(cases), "unique_problems": len(problems), "observations": len(rows),
               "trace_executions": len(traces), "full_budget_steps": len(traces)*4, "arms": {}}
    strata = sorted({(c["family"], c["seed"]) for c in cases})
    for arm in ARMS:
        lost=gained=changed=depth=work=finite_bad=bits_bad=0
        max_diff=0.0
        for c in cases:
            cid=c["case_id"];a=first[(cid,arm)];ref=first[(cid,"eager")];tr=trajectory[(cid,arm)]
            lost += ref["valid"] and not a["valid"];gained += a["valid"] and not ref["valid"]
            changed += a["answer"] != ref["answer"];depth += a["work"]["executed_steps"] != ref["work"]["executed_steps"]
            work += a["work"] != ref["work"]
            finite_bad += not all(s["finite"] for s in tr["steps"])
            bits_bad += not (tr["embedding_bitwise"] and all(s["bitwise"] for s in tr["steps"]))
            for s in tr["steps"]:
                if s["max_abs_logits"] is not None:max_diff=max(max_diff,s["max_abs_logits"])
        # Conditional paired bootstrap: cluster by original problem, retain both fixed models.
        families=defaultdict(dict)
        for c in cases:
            v=families[c["family"]].setdefault(c["problem_id"],[0,0]);cid=c["case_id"]
            v[0]+=med[(cid,arm)];v[1]+=med[(cid,"prepared")]
        rng=random.Random(BOOTSTRAP_SEED);sample=[]
        for _ in range(2000):
            num=den=0
            for family in sorted(families):
                vals=list(families[family].values())
                for _ in vals:
                    n,d=vals[rng.randrange(len(vals))];num+=n;den+=d
            sample.append(num/den)
        valid=sum(first[(c["case_id"],arm)]["valid"] for c in cases)
        observations=[r for r in rows if r["arm"]==arm]
        valid_calls=sum(r["valid"] for r in observations)
        times=sum(r["elapsed_ns"] for r in observations)
        entry={"valid":valid,"lost_valid":lost,"gained_valid":gained,"changed_answers":changed,
               "changed_depths":depth,"changed_work":work,"nonfinite_cases":finite_bad,
               "nonbitwise_cases":bits_bad,"max_abs_logits":max_diff,
               "exact_track_pass":bits_bad==0 and changed==0 and work==0 and finite_bad==0,
               "bounded_task_track_pass":lost==0 and finite_bad==0,
               "summed_case_median_ns":totals[arm],"over_prepared_ratio":totals[arm]/totals['prepared'],
               "problem_cluster_ci95":[quantile(sample,.025),quantile(sample,.975)],
               "charged_ns_per_valid":times/valid_calls if valid_calls else None,
               "case_median_regressions_vs_prepared":sum(med[(c['case_id'],arm)]>med[(c['case_id'],'prepared')] for c in cases),
               "case_p95_regressions_vs_prepared":sum(quantile([r['elapsed_ns'] for r in grouped[(c['case_id'],arm)]],.95)>
                    quantile([r['elapsed_ns'] for r in grouped[(c['case_id'],'prepared')]],.95) for c in cases),
               "strata":{f'{f}:{s}':sum(med[(c['case_id'],arm)] for c in cases if (c['family'],c['seed'])==(f,s))/
                                      sum(med[(c['case_id'],'prepared')] for c in cases if (c['family'],c['seed'])==(f,s)) for f,s in strata}}
        summary["arms"][arm]=entry
    summary['case_median_ns']={c['case_id']:{a:med[(c['case_id'],a)] for a in ARMS} for c in cases}
    return summary
