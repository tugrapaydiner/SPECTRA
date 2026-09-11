"""Exact oracles and deliberately synthetic fixtures, not empirical SAT results."""
from __future__ import annotations

from copy import deepcopy
import itertools
import json
from pathlib import Path
import random
import sys
import types

import pytest

from data.cnf import CNF, SplitMix64, random_3sat
from eval.cnf_repair import CNFRepairState
from eval.sat_workload import (execution_schedule, generate_cases, independent_check,
    model_to_witness, run_case, summarize, validate_protocol, _worker)
from scripts import sat_workload_gate as cli

ROOT = Path(__file__).resolve().parents[1]


def protocol():
    p = json.loads((ROOT/"config/sat_workload_gate_v1.json").read_text())
    p.update(variable_counts=[8], instances_per_tier=2, rounds=3, checker_repeats=3,
             clause_ratio_numerator=1, clause_ratio_denominator=1,
             minimum_instances_per_tier=2, minimum_hard_sat_cases=1)
    return p


def evidence_fixture():
    """Invented nanosecond values ONLY for testing the statistical gate code."""
    p = protocol()
    cases = generate_cases(p)
    witnesses = {}
    for case in cases:
        cnf = CNF.from_record(case["formula"])
        witnesses[case["case_id"]] = list(next(w for w in itertools.product((False, True), repeat=cnf.nvars)
                                            if independent_check(cnf, w)))
    rows = []
    for c, solver, r in execution_schedule(p, cases):
        rows.append({"case_id": c["case_id"], "solver": solver, "round": r,
            "ordered_sha256": c["ordered_sha256"], "status": "SAT_VERIFIED",
            "witness": witnesses[c["case_id"]], "setup_ns": 1000,
            "solver_call_ns": 11_000_000, "first_check_ns": 1000,
            "complete_ns": 12_000_000, "supervised_wall_ns": 20_000_000,
            "checker_ns": [100_000]*p["checker_repeats"],
            "conflict_budget_requested": p["conflict_budget"],
            "process_timeout_seconds": p["process_timeout_seconds"],
            "python_sat_version": p["python_sat_version"],
            "solver_stats": {"conflicts": 1, "decisions": 1, "propagations": 1, "restarts": 0}})
    return p, cases, rows


@pytest.mark.parametrize("n,clauses", [(True, ()), (-1, ()), (1, ((0,),)), (1, ((2,),)),
    (1, ((True,),)), (1, ((1.0,),)), (1, [[1]]), (1, ([1],))])
def test_cnf_rejects_invalid_input(n, clauses):
    with pytest.raises((ValueError, TypeError)):
        CNF(n, clauses)


@pytest.mark.parametrize("witness", [[True], (1,), (True, False), (), (None,)])
def test_requires_complete_boolean_witness(witness):
    p = CNF(1, ((1,),))
    for checker in (p.satisfied, p.violated, lambda w: independent_check(p, w)):
        with pytest.raises(ValueError):
            checker(witness)


def test_empty_and_tautological_formulas():
    assert CNF(0, ()).satisfied(())
    assert not CNF(0, ((),)).satisfied(())
    assert CNF(1, ((1, -1, 1),)).satisfied((False,))
    assert CNF(1, ((1, -1, 1),)).satisfied((True,))
    assert not CNFRepairState(CNF(0, ((),)), ()).valid


def test_exact_repair_counts_exhaustive_small_formula_oracle():
    # Exhaust every assignment and every flip in each generated tiny formula.
    # Duplicate literals, empty clauses and tautologies occur naturally here.
    rng = random.Random(9911)
    for nvars in range(1, 5):
        for _ in range(40):
            clauses = tuple(tuple(rng.choice((-1, 1))*rng.randint(1, nvars)
                                  for _ in range(rng.randrange(6))) for _ in range(rng.randrange(8)))
            p = CNF(nvars, clauses)
            for w in itertools.product((False, True), repeat=nvars):
                direct = tuple(i for i, clause in enumerate(clauses)
                               if all((lit > 0) != w[abs(lit)-1] for lit in clause))
                state = CNFRepairState(p, w)
                assert state.unsatisfied == p.violated(w) == direct
                assert state.valid == independent_check(p, w) == p.satisfied(w)
                for v in range(nvars):
                    before = state.witness
                    made, broken = state.make_break(v)
                    assert state.witness == before
                    state.flip(v)
                    assert len(state.unsatisfied) == len(direct)-made+broken
                    assert state.unsatisfied == p.violated(state.witness)
                    state.flip(v)
                    assert state.witness == w and state.unsatisfied == direct
                assert state.flips == 2*nvars
                assert state.literal_updates == 2*sum(map(len, clauses))
                assert state.feature_literal_visits == sum(map(len, clauses))


def test_long_repair_trajectory_and_snapshot_ownership():
    p, w = random_3sat(12, 40, 102, planted=True)
    s = CNFRepairState(p, w)
    original = s.witness
    for v in [0, 11, 3, 1, 2, 6, 3, 4, 9]*13:
        made, broken = s.make_break(v)
        previous = len(s.unsatisfied)
        s.flip(v)
        assert s.unsatisfied == p.violated(s.witness)
        assert len(s.unsatisfied) == previous-made+broken
    assert original == w
    for invalid in (-1, 12, True, 0.0, "1"):
        with pytest.raises(ValueError):
            s.flip(invalid)


def test_generator_golden_inventory_and_order_identity():
    f, w = random_3sat(3, 8, 0)
    assert w is None
    assert f.sha256() == "c0a6b47088a0229adc32c0a7051eb1ecc9fb951808b80c19c7295ce3e7a0d9b1"
    assert not any(f.satisfied(a) for a in itertools.product((False, True), repeat=3))
    p, w = random_3sat(3, 7, 0, planted=True)
    assert w == (True, False, True) and p.satisfied(w)
    assert p.sha256() == "595d34441f2b9b8cc19d832113f639fc2535c22c646a00876888cb5a2d0b0fde"
    reordered = CNF(f.nvars, tuple(tuple(reversed(c)) for c in reversed(f.clauses)))
    assert reordered.sha256() != f.sha256()
    assert reordered.sha256(normalize_order=True) == f.sha256(normalize_order=True)
    assert CNF.from_record(f.record()) == f
    assert SplitMix64(0).below(2**64) == 0xE220A8397B1DCDAF


@pytest.mark.parametrize("args,kwargs", [((2, 1, 0), {}), ((3, 9, 0), {}),
    ((3, 8, 0), {"planted": True}), ((3, -1, 0), {}), ((3, 1, -1), {}),
    ((3, 1, 2**64), {}), ((3, True, 0), {}), ((3, 1, 0), {"planted": 1})])
def test_generator_rejects_invalid_contracts(args, kwargs):
    with pytest.raises(ValueError):
        random_3sat(*args, **kwargs)


@pytest.mark.parametrize("model", [[1, 1], [1, -1], [0], [True], [1.0], [3], [], None])
def test_model_conversion_rejects_unchecked_or_partial_models(model):
    with pytest.raises(ValueError):
        model_to_witness(CNF(2, ((1,),)), model)


def test_only_unused_variables_can_be_filled():
    assert model_to_witness(CNF(3, ((1,), (-2,))), [1, -2]) == (True, False, False)
    assert model_to_witness(CNF(0, ()), []) == ()


@pytest.mark.parametrize("key,value", [("rounds", 0), ("rounds", True), ("variable_counts", []),
    ("variable_counts", [8, 8]), ("variable_counts", [True]), ("solvers", ["glucose4", "glucose4"]),
    ("solvers", [[], "glucose4"]), ("process_timeout_seconds", float("nan")),
    ("minimum_solve_to_check_ratio", float("inf")), ("minimum_hard_sat_fraction", 1.1),
    ("seed_start", 2**64), ("checker_repeats", 0), ("python_sat_version", None),
    ("execution_order", "best first")])
def test_protocol_rejects_invalid_fields(key, value):
    p = protocol(); p[key] = value
    with pytest.raises(ValueError):
        validate_protocol(p)


def test_summary_has_complete_inventory_and_no_invented_scientific_claim():
    p, cases, rows = evidence_fixture()
    s = summarize(p, cases, rows)
    assert s["observations"] == 24 and s["instances"] == 4
    assert s["sat_witnesses_independently_checked"] == 24
    assert all(t["hard_sat_cases"] == 2 and t["gate"] == "FEASIBILITY_LEAD_ONLY" for t in s["tiers"])
    assert s["energy_joules"] is None and not s["learned_model_evaluated"]
    assert not s["independent_confirmation"] and not s["unsat_proofs_verified"]


@pytest.mark.parametrize("key,value", [("round", False), ("round", []), ("solver", "other"),
    ("case_id", []), ("ordered_sha256", "changed"), ("status", "PASS"),
    ("status", "ERROR"), ("status", []), ("conflict_budget_requested", True),
    ("process_timeout_seconds", float("nan")), ("complete_ns", 0), ("complete_ns", 3.2),
    ("complete_ns", 22_000_000), ("supervised_wall_ns", -1), ("setup_ns", 20_000_000),
    ("checker_ns", [100]), ("checker_ns", [True, 1, 1]), ("witness", [True]),
    ("python_sat_version", "different"), ("solver_stats", {"conflicts": -1})])
def test_summary_fails_closed_on_tampering(key, value):
    p, cases, rows = evidence_fixture(); rows[0][key] = value
    with pytest.raises(ValueError):
        summarize(p, cases, rows)


def test_summary_rejects_missing_duplicate_reordered_and_changed_input():
    p, cases, rows = evidence_fixture()
    for changed in (rows[:-1], rows+[rows[0]], list(reversed(rows))):
        with pytest.raises(ValueError):
            summarize(p, cases, changed)
    cases[0]["seed"] += 1
    with pytest.raises(ValueError):
        summarize(p, cases, rows)


def test_unknown_and_timeout_remain_in_denominator():
    p, cases, rows = evidence_fixture()
    rows[0].update(status="UNKNOWN_BUDGET", witness=None, checker_ns=[])
    rows[1] = {k: v for k, v in rows[1].items() if k not in
               ("setup_ns", "solver_call_ns", "first_check_ns", "solver_stats", "python_sat_version")}
    rows[1].update(status="TIMEOUT", witness=None, checker_ns=[], complete_ns=None)
    s = summarize(p, cases, rows)
    assert s["instances"] == 4 and s["observations"] == 24
    assert s["tiers"][0]["hard_sat_fraction_of_all_generated"] == 0
    assert s["tiers"][0]["gate"] == "NO_FEASIBLE_TIER"
    assert s["statuses"]["TIMEOUT"] == s["statuses"]["UNKNOWN_BUDGET"] == 1
    rows[1]["complete_ns"] = 3_000_000_000
    with pytest.raises(ValueError):
        summarize(p, cases, rows)


def test_unsat_contradiction_and_no_proof_promotion():
    p, cases, rows = evidence_fixture()
    rows[0].update(status="UNSAT_REPORTED", witness=None, checker_ns=[])
    with pytest.raises(ValueError, match="contradicts"):
        summarize(p, cases, rows)
    for row in rows:
        if row["case_id"] == cases[0]["case_id"]:
            row.update(status="UNSAT_REPORTED", witness=None, checker_ns=[])
    s = summarize(p, cases, rows)
    assert s["statuses"]["UNSAT_REPORTED"] == 6 and not s["unsat_proofs_verified"]
    for row in rows:
        if row["case_id"] == cases[-1]["case_id"]:
            row.update(status="UNSAT_REPORTED", witness=None, checker_ns=[])
    with pytest.raises(ValueError, match="contradicts"):
        summarize(p, cases, rows)


def test_gate_uses_fastest_solver_and_every_round_not_slowest_or_filtered():
    p, cases, rows = evidence_fixture()
    for row in rows:
        if row["solver"] == "cadical195":
            row.update(complete_ns=9_000_000, solver_call_ns=8_000_000)
    assert not any(c["hard_witnessed_sat"] for c in summarize(p, cases, rows)["per_case"])


@pytest.mark.parametrize("outcome,expected", [(True, "SAT_VERIFIED"), (False, "UNSAT_REPORTED"),
    (None, "UNKNOWN_BUDGET"), (1, "ERROR")])
def test_worker_states_with_explicit_solver_doubles(monkeypatch, outcome, expected):
    # API doubles test routing, not solver correctness or empirical performance.
    class SolverDouble:
        def __init__(self, **kwargs): self.kwargs = kwargs
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def conf_budget(self, n): assert n == 2000
        def solve_limited(self): return outcome
        def get_model(self): return [1]
        def accum_stats(self): return dict(conflicts=0, decisions=0, propagations=0, restarts=0)
    fake = types.ModuleType("pysat.solvers"); fake.Solver = SolverDouble
    monkeypatch.setitem(sys.modules, "pysat.solvers", fake)
    monkeypatch.setattr("importlib.metadata.version", lambda _: "test-version")
    class Pipe:
        def send(self, row): self.row = row
        def close(self): self.closed = True
    pipe = Pipe()
    _worker(pipe, CNF(1, ((1,),)).record(), "cadical195", 2000, 3, "test-version")
    assert pipe.closed and pipe.row["status"] == expected
    if outcome is True:
        assert len(pipe.row["checker_ns"]) == 3
        assert pipe.row["complete_ns"] >= pipe.row["setup_ns"]+pipe.row["solver_call_ns"]+pipe.row["first_check_ns"]


def test_worker_rejects_version_mismatch(monkeypatch):
    monkeypatch.setattr("importlib.metadata.version", lambda _: "wrong")
    class Pipe:
        def send(self, row): self.row = row
        def close(self): pass
    pipe = Pipe(); _worker(pipe, CNF(1, ()).record(), "cadical195", 2000, 3, "required")
    assert pipe.row["status"] == "ERROR" and "expected python-sat" in pipe.row["error"]


def test_supervisor_excludes_planted_witness_and_retains_startup_failure(monkeypatch):
    import eval.sat_workload as module
    p = protocol(); case = generate_cases(p)[-1]
    captured = {}
    class Pipe:
        def close(self): pass
    class ProcessDouble:
        pid = None
        def __init__(self, *, target, args): captured["args"] = args
        def start(self): raise OSError("synthetic spawn failure")
    class Context:
        def Pipe(self, **kwargs): return Pipe(), Pipe()
        Process = ProcessDouble
    monkeypatch.setattr(module.mp, "get_context", lambda _: Context())
    row = run_case(case, p["solvers"][0], 0, p)
    assert row["status"] == "ERROR" and row["complete_ns"] is None
    assert captured["args"][1] == case["formula"]
    assert "generation_witness" not in captured["args"][1]


def test_evidence_roundtrip_and_tampering(tmp_path):
    p, cases, rows = evidence_fixture()
    cli.write_json(tmp_path/"protocol.json", p)
    cli.write_json(tmp_path/"cases.json", cases)
    (tmp_path/"rows.jsonl").write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in rows))
    cli.write_json(tmp_path/"summary.json", summarize(p, cases, rows))
    cli.write_json(tmp_path/"environment.json", {"python_sat_version": p["python_sat_version"], "physical_energy_joules": None})
    cli.write_json(tmp_path/"source.json", {"files": cli.current_sources(),
        "protocol_sha256": cli.sha((tmp_path/"protocol.json").read_bytes()), "git_commit": None})
    cli.write_json(tmp_path/"sha256.json", {name: cli.sha((tmp_path/name).read_bytes()) for name in cli.EVIDENCE_FILES})
    assert cli.verify(tmp_path)["integrity"] == "PASS"
    (tmp_path/"failure.txt").write_text("retained failed attempt")
    with pytest.raises(ValueError): cli.verify(tmp_path)
    (tmp_path/"failure.txt").unlink()
    (tmp_path/"rows.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="digest"): cli.verify(tmp_path)


def test_cli_refuses_overwrite_and_ambiguous_json(tmp_path):
    with pytest.raises(FileExistsError):
        cli.run(ROOT/"config/sat_workload_gate_v1.json", tmp_path)
    path = tmp_path/"bad.json"
    for raw in ('{"x":1,"x":2}', '{"x":NaN}'):
        path.write_text(raw)
        with pytest.raises(ValueError): cli.read_json(path)


@pytest.mark.parametrize("mode, expected", [("timeout", "TIMEOUT"), ("eof", "ERROR")])
def test_supervisor_timeout_and_early_exit_cleanup(monkeypatch, mode, expected):
    import eval.sat_workload as module
    p = protocol(); case = generate_cases(p)[0]
    events = []
    class PipeDouble:
        def close(self): events.append("pipe_close")
        def poll(self, seconds):
            assert 0 <= seconds <= p["process_timeout_seconds"]
            return mode != "timeout"
        def recv(self): raise EOFError()
    class ProcessDouble:
        pid = 123
        live = True
        def __init__(self, **kwargs): pass
        def start(self): events.append("start")
        def join(self, timeout=None): events.append("join")
        def is_alive(self): return self.live
        def terminate(self): events.append("terminate")
        def kill(self):
            events.append("kill")
            self.live = False
        def close(self): events.append("process_close")
    class Context:
        def Pipe(self, **kwargs): return PipeDouble(), PipeDouble()
        Process = ProcessDouble
    monkeypatch.setattr(module.mp, "get_context", lambda _: Context())
    result = run_case(case, p["solvers"][0], 0, p)
    assert result["status"] == expected
    assert result["witness"] is None and result["complete_ns"] is None
    assert "kill" in events and "terminate" in events and "process_close" in events


def test_bounded_evidence_archive(tmp_path):
    import zipfile
    folder = tmp_path/"evidence"
    folder.mkdir()
    # Synthetic timings remain test fixtures, never empirical solver evidence.
    p, cases, rows = evidence_fixture()
    cli.write_json(folder/"protocol.json", p)
    cli.write_json(folder/"cases.json", cases)
    (folder/"rows.jsonl").write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in rows))
    cli.write_json(folder/"summary.json", summarize(p, cases, rows))
    cli.write_json(folder/"environment.json", {"python_sat_version": p["python_sat_version"], "physical_energy_joules": None})
    cli.write_json(folder/"source.json", {"files": cli.current_sources(),
        "protocol_sha256": cli.sha((folder/"protocol.json").read_bytes())})
    cli.write_json(folder/"sha256.json", {n: cli.sha((folder/n).read_bytes()) for n in cli.EVIDENCE_FILES})
    archive = tmp_path/"evidence.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(folder.iterdir()): z.write(path, path.name)
    assert cli.verify(archive) == cli.verify(folder)
    with zipfile.ZipFile(archive, "a") as z: z.writestr("../outside.txt", "rejected")
    with pytest.raises(ValueError, match="inventory"): cli.verify(archive)
    assert not (tmp_path/"outside.txt").exists()
    with zipfile.ZipFile(archive, "w") as z:
        for path in sorted(folder.iterdir()): z.write(path, path.name)
        with pytest.warns(UserWarning): z.writestr("summary.json", "{}")
    with pytest.raises(ValueError, match="inventory"): cli.verify(archive)
