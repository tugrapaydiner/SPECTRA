import dataclasses
import itertools
import random
import numpy as np
import pytest
from spectra_reliability.identity import input_fingerprint, pair_fingerprint, strict_json
from spectra_reliability.lineage import (ArtifactNode, ExposureIndex, LineageError, ManifestSource,
                                         ancestral_index, make_manifest_source)
from spectra_reliability.semantics import CheckedEvaluator, EvaluatorContract, SemanticMismatch, TargetKind
from spectra_reliability.budget import Budget, BudgetExhausted, WorkLedger
from spectra_reliability.search import Incumbent, ReliableSearch, Task
from spectra_reliability.sudoku import all_four_by_four, generate_unique, solutions, valid
D = "1" * 64


def manifest(name="ancestor", x=(0,2,3,4), y=(1,2,3,4), row_id="a", *, input_hash=True, group=None):
    fp = pair_fingerprint("test", x, y, 2, 2); row = dict(id=row_id, fingerprint=fp, group_id=group or fp)
    if input_hash: row["input_fingerprint"] = input_fingerprint("test", x, 2, 2)
    return make_manifest_source(name, {"splits": {"train": {"count": 1, "examples": [row]}}})


def contract(kind=TargetKind.CURRENT_VALIDITY, **kwargs):
    extra = dict(continuation_policy="frozen_cycle", budget_unit="recursive_cycles", horizon=1) if kind in (
        TargetKind.ONE_STEP_IMPROVEMENT, TargetKind.BUDGETED_SUCCESS) else {}
    return EvaluatorContract(kind, "test", D, "v1", "exact-v1", **(extra | kwargs))


def budget(n=100, **kw): return Budget(**(dict(transitions=n, checks=n, values=n, decodes=n, policies=n) | kw))


def test_historical_fingerprint_matches_upstream():
    from data.splits import example_fingerprint
    x = np.array([0,2,3,4]); y = np.array([1,2,3,4])
    assert pair_fingerprint("test", x, y, 2, 2) == example_fingerprint("test", x, y, 2, 2)


def test_new_id_does_not_hide_reuse():
    index = ExposureIndex([manifest(row_id="old")]); candidate = manifest("fresh", row_id="unrelated")
    report = index.audit(candidate)
    assert report["overlap_count"] == 1 and report["overlaps"][0]["ancestors"][0]["row_id"] == "old"
    with pytest.raises(LineageError): index.require_disjoint(candidate)


def test_new_target_does_not_hide_reused_input():
    assert ExposureIndex([manifest()]).audit(manifest("new", y=(4,3,2,1), row_id="new"))["overlap_count"] == 1


def test_group_identity_catches_changed_augmentation():
    ancestor = manifest(); group = ancestor.exposures()[0].group_id
    assert ExposureIndex([ancestor]).audit(manifest("new", x=(4,3,2,0), y=(4,3,2,1), row_id="new", group=group))["overlap_count"] == 1


def test_incomplete_input_only_coverage_not_promoted():
    index = ExposureIndex([manifest(input_hash=False)]); candidate = manifest("new", x=(0,0,3,4), row_id="new")
    r = index.audit(candidate)
    assert r["pass"] and not r["complete_input_only_coverage"] and not r["symmetry_disjointness_established"]
    with pytest.raises(LineageError): index.require_disjoint(candidate, require_input_coverage=True)


def test_hash_missing_file_and_missing_inventory_fail_closed(tmp_path):
    m = manifest()
    with pytest.raises(LineageError): ExposureIndex([ManifestSource(m.name, "0"*64, m.raw)])
    with pytest.raises(FileNotFoundError): ManifestSource.from_path("missing", tmp_path/"none.json", D)
    with pytest.raises(LineageError): ExposureIndex([])


def test_transitive_ancestry_and_cycles():
    src = manifest(); nodes = [ArtifactNode("core", D, (), ("data",)), ArtifactNode("verifier", D, ("core",), ())]
    assert len(ancestral_index(nodes, ["verifier"], {"data": src}).exposures) == 1
    with pytest.raises(LineageError, match="unresolved ancestor"): ancestral_index(nodes, ["unknown"], {"data": src})
    with pytest.raises(LineageError, match="unresolved manifest"): ancestral_index(nodes, ["core"], {})
    with pytest.raises(LineageError, match="cycle"):
        ancestral_index([ArtifactNode("a", D, ("b",), ()), ArtifactNode("b", D, ("a",), ())], ["a"], {})


@pytest.mark.parametrize("raw", ['{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}', '{"x": 1e999}'])
def test_strict_json(raw):
    with pytest.raises(ValueError): strict_json(raw)


def test_bad_manifest_count():
    m = make_manifest_source("bad", {"splits": {"train": {"count": 1, "examples": []}}})
    with pytest.raises(LineageError): m.exposures()


def test_improvement_is_not_absolute_value():
    c = contract(TargetKind.ONE_STEP_IMPROVEMENT)
    with pytest.raises(SemanticMismatch): CheckedEvaluator(c, contract(), lambda _: .8)
    with pytest.raises(SemanticMismatch): CheckedEvaluator(c, c, lambda _: .8)


@pytest.mark.parametrize("field,value", [("model_sha256", "2"*64), ("task", "other"), ("decoder", "v2"), ("checker", "other")])
def test_evaluator_identity_mismatch(field, value):
    c = contract()
    with pytest.raises(SemanticMismatch): CheckedEvaluator(c, dataclasses.replace(c, **{field:value}), lambda _: .5)


@pytest.mark.parametrize("value", [float('nan'), float('inf'), -.01, 1.01, True])
def test_invalid_value_output(value):
    c = contract(); evaluator = CheckedEvaluator(c, c, lambda _: value)
    with pytest.raises(SemanticMismatch): evaluator(None)


def test_horizon_and_policy_are_bound():
    c = contract(TargetKind.BUDGETED_SUCCESS)
    with pytest.raises(SemanticMismatch): c.require(contract(TargetKind.BUDGETED_SUCCESS, horizon=2))
    with pytest.raises(SemanticMismatch): c.require(contract(TargetKind.BUDGETED_SUCCESS, continuation_policy="new"))


def test_incumbent_is_immutable_and_task_bound():
    task = Task("test", "exact-v1", (0,)); inc = Incumbent(task, lambda x,y: y == (1,))
    mutable = [1]; inc.offer(mutable, (0,), 0.0); mutable[0] = 7
    for i in range(100): inc.offer((2,), (i,), 1e30+i)
    assert inc.best.answer == (1,) and inc.best.valid and inc.invalid_after_valid == 100
    assert inc.best.task_identity == task.identity != Task("test", "exact-v1", (2,)).identity


def test_checker_scores_are_not_certificates():
    inc = Incumbent(Task("test", "exact-v1", (0,)), lambda x,y: .999)
    with pytest.raises(TypeError): inc.offer([1], (), 0.)


def make_search(transition=None, predict=lambda s:.9):
    c = contract()
    return ReliableSearch(task=Task("test", "exact-v1", (0,)), checker=lambda x,y:y == (1,),
        evaluator=CheckedEvaluator(c, c, predict), actions=lambda s:[1,2,3], transition=transition or (lambda s,a:a),
        decode=lambda s:[s], max_depth=3)


def test_valid_answer_skips_proxy_and_remaining_work():
    r = make_search(predict=lambda s:(_ for _ in ()).throw(AssertionError())).run(0, budget())
    assert r.valid and r.answer == (1,) and r.status == "verified_search"
    assert r.stats["counts"] == dict(transition=1, decode=1, check=1, value=0, policy=1)


def test_baseline_cost_is_charged():
    r = make_search().run(0, budget(), baseline_actions=[1,2])
    assert r.valid and r.status == "verified_baseline"
    assert r.stats["baseline_steps"] == r.stats["counts"]["transition"] == 1


def test_actual_callback_failure_retains_incumbent():
    def bad(s,a):
        if a == 2: raise RuntimeError("adversarial transition")
        return a
    r = make_search(bad).run(0, budget(), baseline_actions=[1], stop_on_valid=False)
    assert r.valid and r.answer == (1,) and r.status == "callback_error"
    assert r.stats["callback_failures"]["transition"] == 1


def test_actual_budget_exhaustion_retains_incumbent():
    r = make_search().run(0, budget(2), baseline_actions=[1], stop_on_valid=False)
    assert r.valid and r.answer == (1,) and r.status == "budget_exhausted"
    assert r.stats["counts"]["transition"] == 2


def test_zero_budget_cannot_invent_answer():
    r = make_search().run(0, budget(0))
    assert r.answer is None and not r.valid and r.status == "budget_exhausted"
    assert all(v == 0 for v in r.stats["counts"].values())


def test_cooperative_deadline_retains_overshoot():
    now = [0]; ledger = WorkLedger(budget(deadline_ms=1), clock=lambda:now[0], cpu_clock=lambda:now[0])
    def work(): now[0] = 2_000_000; return 3
    assert ledger.call("transition", work) == 3 and ledger.snapshot()["deadline_overshoot_ns"] == 1_000_000
    with pytest.raises(BudgetExhausted): ledger.call("decode", lambda:1)
    assert ledger.counts["decode"] == 0


def test_nested_accounting_rejected():
    ledger = WorkLedger(budget())
    with pytest.raises(RuntimeError): ledger.call("transition", lambda:ledger.call("decode", lambda:1))
    assert ledger.counts["transition"] == 1 and ledger.counts["decode"] == 0


def test_exhaustive_valid_boards_and_single_cell_corruption():
    boards = all_four_by_four(); assert len(boards) == 288
    for y in boards:
        assert valid((0,)*16, y, 2)
        for index in range(16):
            for wrong in (-1,0,1,2,3,4,5):
                if wrong == y[index]: continue
                bad = list(y); bad[index] = wrong
                assert not valid((0,)*16, bad, 2)


def test_generated_unique_9x9_and_4x4():
    rng = random.Random(73)
    for box, clues in [(2,6), (2,4), (3,32)]:
        x,y = generate_unique(box,clues,rng); got,stats = solutions(x,box)
        assert stats.complete and got == [y] and valid(x,y,box) and sum(v != 0 for v in x) == clues


@pytest.mark.parametrize("x,y", [([0.]*16,[1]*16), ([False]*16,[1]*16), ([0]*15,[1]*16)])
def test_checker_malformed_input(x,y):
    with pytest.raises(ValueError): valid(x,y,2)
