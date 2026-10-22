from __future__ import annotations

import ctypes
from dataclasses import replace
import hashlib
import importlib
from itertools import combinations, product
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

import pytest

from experiments.general_competition.binary_core import canonical_cnf, model_satisfies
from experiments.general_competition.proof_backend import check_drat, write_dimacs
from experiments.general_competition.native_cdcl import (
    DRUPProof,
    NativeCDCLRuntime,
    build_native_cdcl,
    solve_cnf_native,
    solve_general_native,
    verify_direct_native_result,
    verify_drup_proof,
    verify_native_result,
    write_drup,
)


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    root = tmp_path_factory.mktemp("native-cdcl")
    return NativeCDCLRuntime(build_native_cdcl(root))


_BUILDERS = (
    ("spectra.exact", "build_exact_runtime"),
    ("experiments.compiled_query.exact", "build_exact_runtime"),
    ("experiments.general_competition.native_cdcl", "build_native_cdcl"),
    ("experiments.general_competition.native_binary_core", "build_native_binary_core"),
)


@pytest.mark.parametrize("module_name,builder_name", _BUILDERS)
@pytest.mark.parametrize("event", ("source-edit", "receipt-collision", "compiler-failure"))
def test_native_build_publication_and_source_identity(
        tmp_path, monkeypatch, module_name, builder_name, event):
    module = importlib.import_module(module_name)
    if module_name.endswith("native_binary_core") and sys.platform != "linux":
        pytest.skip("binary-core builder supports Linux only")
    source = tmp_path / module.SOURCE.name
    original = b"// original source bytes\n"
    source.write_bytes(original)
    destination = tmp_path / "build"
    sentinel = b"another writer's receipt\n"
    compiled = []

    def compiler(command, **_kwargs):
        sources = [Path(arg) for arg in command if arg.endswith(".cpp")]
        if sources:
            snapshot = sources[0]
            assert snapshot != source and snapshot.name == source.name
            source.write_bytes(b"// edited while the compiler runs\n")
            assert snapshot.read_bytes() == original
            compiled.append(snapshot)
            target = (
                Path(command[command.index("-o") + 1]) if "-o" in command
                else Path(next(arg[5:] for arg in command if arg.startswith("/OUT:")))
            )
            target.write_bytes(original)
            if event == "receipt-collision":
                (destination / "build.json").write_bytes(sentinel)
            if event == "compiler-failure":
                return subprocess.CompletedProcess(command, 1, "", "injected compiler failure")
        return subprocess.CompletedProcess(command, 0, "test compiler\n", "")

    monkeypatch.setattr(module, "SOURCE", source)
    monkeypatch.setattr(module.shutil, "which", lambda name: name)
    monkeypatch.setattr(module.subprocess, "run", compiler)
    builder = getattr(module, builder_name)
    if event == "source-edit":
        library = builder(destination)
        receipt = json.loads(library.with_name("build.json").read_text())
        assert receipt["source_sha256"] == hashlib.sha256(original).hexdigest()
        assert receipt["library_sha256"] == hashlib.sha256(library.read_bytes()).hexdigest()
        assert compiled and not compiled[0].exists()
    elif event == "receipt-collision":
        with pytest.raises(FileExistsError):
            builder(destination)
        assert (destination / "build.json").read_bytes() == sentinel
        assert list(destination.iterdir()) == [destination / "build.json"]
    else:
        with pytest.raises(RuntimeError, match="injected compiler failure"):
            builder(destination)
        assert not list(destination.iterdir())


@pytest.mark.skipif(os.name != "posix", reason="replacing a loaded library requires POSIX unlink")
def test_native_runtime_rejects_changed_bytes_at_a_loaded_path(runtime, tmp_path):
    library = tmp_path / runtime.path.name
    receipt = library.with_name("build.json")
    shutil.copy2(runtime.path, library)
    shutil.copy2(runtime.path.with_name("build.json"), receipt)
    NativeCDCLRuntime(library)
    NativeCDCLRuntime(library)  # Reusing exactly the same bytes is supported.
    changed = library.read_bytes() + b"\x00"
    library.unlink()
    library.write_bytes(changed)
    record = json.loads(receipt.read_text())
    record["library_sha256"] = hashlib.sha256(changed).hexdigest()
    receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="already loaded with different bytes"):
        NativeCDCLRuntime(library)


@pytest.mark.parametrize("nvars", [10_000_001, 1 << 30, (1 << 31) - 1, (1 << 32) - 1])
def test_raw_native_abi_rejects_excess_variables_before_indexing(runtime, nvars):
    # Call the exported ABI directly: Python's canonical_cnf guard must not be
    # the only protection against signed overflow in 2 * nvars / nvars + 1.
    offsets = (ctypes.c_uint64 * 1)(0)
    output = ctypes.c_void_p(1)
    size = ctypes.c_size_t(123)
    code = runtime._library.spectra_g_native_solve(
        nvars, None, 0, offsets, 0, 0, 0,
        ctypes.byref(output), ctypes.byref(size),
    )
    assert code == 1
    assert runtime._library.spectra_g_native_error() == b"too many variables"
    assert output.value is None
    assert size.value == 0


def test_activity_underflow_preserves_relevant_variables(tmp_path, monkeypatch):
    from experiments.general_competition import native_cdcl

    # Exercise the native rescaling invariant directly: a full search requires
    # tens of thousands of conflicts to reach the same four rescalings. Variables
    # 1 and 2 occur in an unsatisfied clause; variable 4 is genuinely absent.
    source = native_cdcl.SOURCE.resolve().as_posix()
    harness = tmp_path / "activity_underflow.cpp"
    harness.write_text(
        '#include "' + source + '"\n' + r"""
SPECTRA_EXPORT int spectra_g_test_activity_underflow() {
  Solver solver(4);
  const int clause[] = {1, 2};
  const int unit[] = {3};
  solver.add_initial(clause, clause + 2);
  solver.add_initial(unit, unit + 1);
  solver.initialize_heap();
  for (int step = 0; step < 4; ++step) {
    solver.var_increment = 1e101;
    solver.bump_variable(3);
  }
  if (solver.activity[1] != 0.0 || solver.activity[2] != 0.0) return 1;
  if (solver.solve(100, 10000) != 10) return 2;
  if (solver.literal_value(1) <= 0 && solver.literal_value(2) <= 0) return 3;
  if (solver.values[4] != 0) return 4;
  return 0;
}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(native_cdcl, "SOURCE", harness)
    library = ctypes.CDLL(str(build_native_cdcl(tmp_path / "underflow-build")))
    check = library.spectra_g_test_activity_underflow
    check.argtypes = []
    check.restype = ctypes.c_int
    assert check() == 0


def test_compact_watch_pairs_track_moves_rebuild_and_deletion(tmp_path, monkeypatch):
    from experiments.general_competition import native_cdcl

    source = native_cdcl.SOURCE.resolve().as_posix()
    harness = tmp_path / "watch_pairs.cpp"
    harness.write_text(
        '#include "' + source + '"\n' + r"""
bool valid_watch_pairs(const Solver &s) {
  if (s.watched_literals.size() != s.clauses.size()) return false;
  std::vector<int> counts(s.clauses.size(), 0);
  for (std::size_t slot = 0; slot < s.watches.size(); ++slot) {
    for (int id : s.watches[slot]) {
      const Clause &c = s.clauses[id];
      if (c.deleted || (slot != s.watch_index(c.lits[c.watch0]) &&
                        slot != s.watch_index(c.lits[c.watch1]))) return false;
      ++counts[id];
    }
  }
  for (std::size_t id = 0; id < s.clauses.size(); ++id) {
    const Clause &c = s.clauses[id];
    const auto pair = s.watched_literals[id];
    if (c.deleted) {
      if (pair.first || pair.second || counts[id]) return false;
    } else if (pair.first != c.lits[c.watch0] || pair.second != c.lits[c.watch1] ||
               counts[id] != (c.watch0 == c.watch1 ? 1 : 2)) return false;
  }
  return true;
}

SPECTRA_EXPORT int spectra_g_test_watch_pairs() {
  Solver s(4);
  const int row[] = {1, 2, 3};
  s.add_initial(row, row + 3);
  s.initialize_heap();
  if (!valid_watch_pairs(s)) return 1;
  s.trail_limits.push_back(0);
  if (!s.enqueue(-1, -1) || s.propagate() != -1) return 2;
  if (s.watched_literals[0] != std::make_pair(3, 2) || !valid_watch_pairs(s)) return 3;
  s.backtrack(0);
  s.trail_limits.push_back(0);
  if (!s.enqueue(1, -1) || s.propagate() != -1) return 4;
  s.trail_limits.push_back(static_cast<int>(s.trail.size()));
  if (!s.enqueue(-2, -1) || s.propagate() != -1) return 5;
  // Literal 1 satisfies the clause after backtracking. The second watch must
  // still move from 2 to 1, exactly as the original propagation algorithm does.
  if (s.watched_literals[0] != std::make_pair(3, 1) || s.values[3] || !valid_watch_pairs(s)) return 6;
  s.watched_literals[0] = {0, 0};
  s.rebuild_watches();
  if (!valid_watch_pairs(s)) return 7;
  s.backtrack(0);
  s.trail_limits.push_back(0);
  if (!s.enqueue(-1, -1) || s.propagate() != -1) return 8;
  s.trail_limits.push_back(static_cast<int>(s.trail.size()));
  if (!s.enqueue(-2, -1) || s.propagate() != -1) return 9;
  if (s.values[3] != 1 || s.reasons[3] != 0 || !valid_watch_pairs(s)) return 10;

  Solver unit(1);
  unit.add_clause({1}, false);
  unit.initialize_heap();
  if (unit.watched_literals[0] != std::make_pair(1, 1) || !valid_watch_pairs(unit)) return 11;
  unit.trail_limits.push_back(0);
  if (!unit.enqueue(-1, -1) || unit.propagate() != 0) return 12;
  Solver conflict(5);
  for (const auto &clause : std::vector<std::vector<int>>{{1, 2, 3}, {1, 2}, {1, 4}, {1, 5}})
    conflict.add_initial(clause.data(), clause.data() + clause.size());
  conflict.initialize_heap();
  conflict.trail_limits.push_back(0);
  if (!conflict.enqueue(-1, -1) || !conflict.enqueue(-2, -1) || conflict.propagate() != 1) return 13;
  if (conflict.watches[conflict.watch_index(1)] != std::vector<int>({1, 2, 3}) ||
      !valid_watch_pairs(conflict)) return 14;

  Solver reduction(4);
  reduction.add_initial(row, row + 3);
  for (int index = 0; index < 32; ++index) reduction.add_clause({1, 2, 3}, true, 3);
  reduction.reduce_database();
  if (reduction.stats.deleted_clauses != 16 || !valid_watch_pairs(reduction)) return 15;
  // Preserve the original stale-registration skip without dereferencing a
  // nonmatching compact pair as if it represented a current watch.
  reduction.watches[reduction.watch_index(4)].push_back(0);
  reduction.trail_limits.push_back(0);
  if (!reduction.enqueue(-4, -1) || reduction.propagate() != -1) return 16;
  return reduction.watches[reduction.watch_index(4)].empty() && valid_watch_pairs(reduction) ? 0 : 17;
}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(native_cdcl, "SOURCE", harness)
    library = ctypes.CDLL(str(build_native_cdcl(tmp_path / "watch-pair-build")))
    check = library.spectra_g_test_watch_pairs
    check.argtypes = []
    check.restype = ctypes.c_int
    assert check() == 0


def brute_sat(nvars, clauses):
    canonical = canonical_cnf(nvars, clauses)
    return any(
        model_satisfies(nvars, canonical, model)
        for model in product((False, True), repeat=nvars)
    )


def all_blocking_clauses(nvars):
    return tuple(
        tuple(variable if bit else -variable for variable, bit in enumerate(bits, 1))
        for bits in product((False, True), repeat=nvars)
    )


def test_direct_native_sat_and_unsat_are_checked_on_original_cnf(runtime, tmp_path):
    sat_clauses = ((1, 2, 3), (-1, 2), (-2, 3))
    sat = solve_cnf_native(runtime, 3, sat_clauses)
    assert sat.status == "SAT_VERIFIED"
    assert sat.reason == "model"
    assert sat.proof is None
    assert model_satisfies(3, sat_clauses, sat.model)
    assert verify_direct_native_result(3, sat_clauses, sat)

    unsat_clauses = all_blocking_clauses(4)
    unsat = solve_cnf_native(runtime, 4, unsat_clauses)
    assert unsat.status == "UNSAT_VERIFIED"
    assert unsat.reason == "drup"
    assert unsat.proof is not None
    assert verify_direct_native_result(4, unsat_clauses, unsat)
    proof_path = write_drup(tmp_path / "proof.drat", unsat.proof)
    assert proof_path.read_text(encoding="ascii").splitlines()[-1] == "0"
    with pytest.raises(FileExistsError):
        write_drup(proof_path, unsat.proof)


@pytest.mark.skipif(
    not os.environ.get("DRAT_TRIM"),
    reason="pinned external DRAT-trim checker is not configured",
)
def test_direct_native_proof_passes_drat_trim(runtime, tmp_path):
    clauses = all_blocking_clauses(5)
    result = solve_cnf_native(runtime, 5, clauses)
    assert result.status == "UNSAT_VERIFIED"
    assert result.proof is not None
    formula = write_dimacs(tmp_path / "formula.cnf", 5, clauses)
    proof = write_drup(tmp_path / "proof.drat", result.proof)
    checked = check_drat(os.environ["DRAT_TRIM"], formula, proof)
    assert checked.verified, checked.record()


def test_direct_native_budget_is_unknown(runtime):
    clauses = all_blocking_clauses(5)
    result = solve_cnf_native(runtime, 5, clauses, max_conflicts=0)
    assert result.status == "UNKNOWN"
    assert result.reason == "conflict_budget"
    assert result.model == ()
    assert result.proof is None
    assert verify_direct_native_result(5, clauses, result)


@pytest.fixture(params=["direct", "reduced"])
def result_api(request):
    if request.param == "direct":
        return solve_cnf_native, verify_direct_native_result
    return solve_general_native, verify_native_result


def test_native_replay_rejects_inconsistent_budget_receipts(runtime, result_api):
    solve, verify = result_api
    clauses = all_blocking_clauses(5)
    conflict = solve(runtime, 5, clauses, max_conflicts=0)
    assert conflict.status == "UNKNOWN"
    assert conflict.stats.conflicts == 1
    assert verify(5, clauses, conflict)
    assert not verify(5, clauses, replace(conflict, max_conflicts=1))
    assert not verify(5, clauses, replace(conflict, reason="learned_literal_budget"))
    assert not verify(5, clauses, replace(
        conflict, stats=replace(conflict.stats, conflicts=2)
    ))

    learned = solve(runtime, 5, clauses, max_learned_literals=0)
    assert learned.status == "UNKNOWN"
    assert learned.stats.conflicts > 0
    assert verify(5, clauses, learned)
    assert not verify(5, clauses, replace(
        learned, stats=replace(learned.stats, conflicts=0)
    ))


def test_native_replay_checks_solved_conflict_budgets(runtime, result_api):
    solve, verify = result_api
    clauses = all_blocking_clauses(3)
    # A final root conflict is checked before the conflict budget. The three
    # admitted learned clauses therefore allow UNSAT at conflict four.
    unsat = solve(runtime, 3, clauses, max_conflicts=3)
    assert unsat.status == "UNSAT_VERIFIED"
    assert unsat.stats.conflicts == unsat.max_conflicts + 1
    assert verify(3, clauses, unsat)
    assert not verify(3, clauses, replace(unsat, max_conflicts=2))

    root_clauses = ((1,), (-1,))
    root = solve(runtime, 1, root_clauses, max_conflicts=0, max_learned_literals=0)
    assert root.status == "UNSAT_VERIFIED"
    assert root.stats.conflicts == 0
    assert verify(1, root_clauses, root)

    sat_clauses = ((1, 2, 3),)
    sat = solve(runtime, 3, sat_clauses, max_conflicts=0)
    assert sat.status == "SAT_VERIFIED"
    assert verify(3, sat_clauses, sat)
    assert not verify(3, sat_clauses, replace(
        sat, stats=replace(sat.stats, conflicts=1)
    ))


@pytest.mark.parametrize("field", ["learned_clauses", "learned_literals"])
def test_native_replay_matches_unsat_proof_to_statistics(runtime, result_api, field):
    solve, verify = result_api
    clauses = all_blocking_clauses(3)
    result = solve(runtime, 3, clauses)
    assert result.status == "UNSAT_VERIFIED"
    assert getattr(result.stats, field) > 0
    assert not verify(3, clauses, replace(
        result, stats=replace(result.stats, **{field: 0})
    ))


def test_native_replay_rejects_impossible_learned_literal_budget(runtime, result_api):
    solve, verify = result_api
    clauses = all_blocking_clauses(3)
    result = solve(runtime, 3, clauses, max_learned_literals=0)
    assert result.reason == "learned_literal_budget"
    assert verify(3, clauses, result)
    # No learned clause can contain more than one literal per native variable.
    assert not verify(3, clauses, replace(
        result, max_learned_literals=result.stats.learned_literals + 3
    ))


def test_native_replay_uses_residual_variables_for_learned_budget(runtime):
    clauses = all_blocking_clauses(3) + ((4,), (5,))
    result = solve_general_native(runtime, 5, clauses, max_learned_literals=0)
    assert result.reason == "learned_literal_budget"
    assert result.residual_variables == 3
    assert verify_native_result(5, clauses, result)
    assert not verify_native_result(5, clauses, replace(
        result, max_learned_literals=result.stats.learned_literals + 3
    ))


@pytest.mark.parametrize("clauses", [(), ((1,), (-1,))])
@pytest.mark.parametrize("counter", [
    "decisions", "propagations", "conflicts", "restarts", "learned_clauses",
    "learned_literals", "reductions", "deleted_clauses",
])
def test_native_replay_requires_zero_shortcut_statistics(runtime, clauses, counter):
    result = solve_general_native(runtime, 1, clauses)
    assert verify_native_result(1, clauses, result)
    assert not verify_native_result(1, clauses, replace(
        result, stats=replace(result.stats, **{counter: 1})
    ))


@pytest.mark.parametrize("clauses,reason", [
    ((), "model"),
    (((1, 2, 3),), "binary_core_model"),
])
def test_native_replay_requires_reason_to_match_solver_path(runtime, clauses, reason):
    result = solve_general_native(runtime, 3, clauses)
    assert result.status == "SAT_VERIFIED"
    assert result.reason != reason
    assert verify_native_result(3, clauses, result)
    assert not verify_native_result(3, clauses, replace(result, reason=reason))


@pytest.mark.parametrize("field", ["max_conflicts", "max_learned_literals"])
def test_native_replay_rejects_out_of_abi_budget(runtime, result_api, field):
    solve, verify = result_api
    clauses = ((1, 2, 3),)
    result = solve(runtime, 3, clauses)
    assert verify(3, clauses, result)
    assert not verify(3, clauses, replace(result, **{field: 1 << 64}))


def test_native_replay_rejects_out_of_abi_statistics(runtime, result_api):
    solve, verify = result_api
    clauses = ((1, 2, 3),)
    result = solve(runtime, 3, clauses)
    assert not verify(3, clauses, replace(
        result, stats=replace(result.stats, decisions=1 << 64)
    ))


@pytest.mark.parametrize("model", [None, [True, True, True], (1, 1, 1), (True,)])
def test_native_replay_rejects_malformed_model(runtime, result_api, model):
    solve, verify = result_api
    clauses = ((1, 2, 3),)
    result = solve(runtime, 3, clauses)
    assert result.status == "SAT_VERIFIED"
    assert not verify(3, clauses, replace(result, model=model))


@pytest.mark.parametrize("proof", [
    object(), DRUPProof("wrong", ((),)),
    DRUPProof("spectra.general_drup.v1", None),
    DRUPProof("spectra.general_drup.v1", (None,)),
    DRUPProof("spectra.general_drup.v1", (1,)),
    DRUPProof("spectra.general_drup.v1", [()]),
])
def test_native_replay_rejects_malformed_proof(runtime, result_api, proof):
    solve, verify = result_api
    clauses = all_blocking_clauses(3)
    result = solve(runtime, 3, clauses)
    assert result.status == "UNSAT_VERIFIED"
    assert not verify(3, clauses, replace(
        result, proof=proof, proof_clauses=1, proof_literals=0
    ))


@pytest.mark.parametrize("field", ["variables", "clauses"])
@pytest.mark.parametrize("value", [True, 1.0])
def test_direct_native_replay_requires_exact_dimensions(runtime, field, value):
    result = solve_cnf_native(runtime, 1, ((1,),))
    assert not verify_direct_native_result(
        1, ((1,),), replace(result, **{field: value})
    )


@pytest.mark.parametrize("clauses", [(), ((1,), (-1,))])
def test_native_replay_rejects_unknown_for_binary_core_shortcut(runtime, clauses):
    result = solve_general_native(runtime, 1, clauses, max_conflicts=0)
    assert result.status != "UNKNOWN"
    forged = replace(result, status="UNKNOWN", model=(), reason="conflict_budget",
                     stats=replace(result.stats, conflicts=1))
    assert not verify_native_result(1, clauses, forged)


def test_native_sat_model_is_lifted_and_checked(runtime):
    clauses = (
        (1, -2),
        (-1, 2),
        (2, 3, 4),
        (-3, 4),
        (-4, 5, 6),
    )
    result = solve_general_native(runtime, 6, clauses)
    assert result.status == "SAT_VERIFIED"
    assert result.reason == "model"
    assert result.proof is None
    assert model_satisfies(6, clauses, result.model)
    assert verify_native_result(6, clauses, result)


def test_native_unsat_emits_replayable_drup(runtime):
    clauses = all_blocking_clauses(4)
    result = solve_general_native(runtime, 4, clauses)
    assert result.status == "UNSAT_VERIFIED"
    assert result.reason == "drup"
    assert result.proof is not None
    assert result.proof.additions[-1] == ()
    assert result.proof_clauses >= 1
    assert verify_drup_proof(
        result.core.reduced_nvars,
        result.core.residual_clauses,
        result.proof,
    )
    assert verify_native_result(4, clauses, result)


def test_binary_core_shortcuts_keep_native_solver_out_of_the_way(runtime):
    unsat = solve_general_native(runtime, 2, ((), (1, 2)), max_conflicts=0)
    assert unsat.status == "UNSAT_VERIFIED"
    assert unsat.reason == "binary_core"
    assert unsat.stats.conflicts == 0
    assert verify_native_result(2, ((), (1, 2)), unsat)

    sat = solve_general_native(runtime, 4, (), max_conflicts=0)
    assert sat.status == "SAT_VERIFIED"
    assert sat.reason == "binary_core_model"
    assert sat.model == (False,) * 4
    assert verify_native_result(4, (), sat)


def test_conflict_budget_is_unknown_not_unsat(runtime):
    # This formula needs search and reaches a conflict before it can complete.
    clauses = all_blocking_clauses(5)
    result = solve_general_native(runtime, 5, clauses, max_conflicts=0)
    assert result.status == "UNKNOWN"
    assert result.reason == "conflict_budget"
    assert result.proof is None
    assert verify_native_result(5, clauses, result)


def test_learned_literal_budget_is_unknown_not_unsat(runtime):
    clauses = all_blocking_clauses(5)
    result = solve_general_native(
        runtime,
        5,
        clauses,
        max_conflicts=1_000_000,
        max_learned_literals=0,
    )
    assert result.status == "UNKNOWN"
    assert result.reason == "learned_literal_budget"
    assert result.stats.learned_literals == 0
    assert result.proof is None
    assert verify_native_result(5, clauses, result)


def test_drup_replay_rejects_changed_or_missing_additions(runtime):
    clauses = all_blocking_clauses(4)
    result = solve_general_native(runtime, 4, clauses)
    assert result.proof is not None
    proof = result.proof
    assert not verify_drup_proof(4, clauses, DRUPProof(proof.schema, proof.additions[:-1]))
    changed = ((1,),) + proof.additions[1:]
    assert not verify_drup_proof(4, clauses, DRUPProof(proof.schema, changed))
    assert not verify_drup_proof(4, clauses, replace(proof, schema="wrong"))


def test_receipt_rejects_changed_source_and_library_hashes(runtime, tmp_path):
    copied = tmp_path / runtime.path.name
    receipt = tmp_path / "build.json"
    shutil.copy2(runtime.path, copied)
    shutil.copy2(runtime.path.with_name("build.json"), receipt)

    record = json.loads(receipt.read_text(encoding="utf-8"))
    record["source_sha256"] = "0" * 64
    receipt.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="source hash"):
        NativeCDCLRuntime(copied)

    shutil.copy2(runtime.path.with_name("build.json"), receipt)
    with copied.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="library hash"):
        NativeCDCLRuntime(copied)


def test_learned_clause_reduction_preserves_the_retained_proof(runtime):
    clauses = all_blocking_clauses(11)
    status, model, proof, stats, canonical = runtime.solve_residual(
        11,
        clauses,
        max_conflicts=10_000_000,
        max_learned_literals=100_000_000,
    )
    assert status == 20
    assert model == ()
    assert proof is not None
    assert stats.reductions >= 1
    assert stats.deleted_clauses >= 1
    assert verify_drup_proof(11, canonical, proof)


def test_exhaustive_small_formula_space(runtime):
    literals = (-3, -2, -1, 1, 2, 3)
    pool = {()}
    for width in (1, 2, 3):
        for row in combinations(literals, width):
            if any(-literal in row for literal in row):
                continue
            pool.add(tuple(sorted(row, key=lambda value: (abs(value), value))))
    ordered = tuple(sorted(pool, key=lambda row: (len(row), row)))

    checked = 0
    for count in range(4):
        for formula in combinations(ordered, count):
            expected = brute_sat(3, formula)
            result = solve_general_native(runtime, 3, formula, max_conflicts=100_000)
            assert result.status != "UNKNOWN"
            assert verify_native_result(3, formula, result)
            assert (result.status == "SAT_VERIFIED") == expected
            checked += 1
    assert checked > 3000


def test_random_formulas_match_truth_tables_and_pysat(runtime):
    rng = random.Random(0xC0D1)
    pysat = pytest.importorskip("pysat.solvers")
    for nvars in range(1, 9):
        for _ in range(80):
            rows = []
            for _ in range(rng.randrange(0, 32)):
                width = rng.randrange(1, min(6, nvars + 1))
                variables = rng.sample(range(1, nvars + 1), width)
                rows.append(tuple(
                    variable if rng.randrange(2) else -variable
                    for variable in variables
                ))
            expected = brute_sat(nvars, rows) if nvars <= 6 else None
            with pysat.Solver(name="g4", bootstrap_with=rows) as competitor:
                pysat_sat = competitor.solve()
            if expected is not None:
                assert expected == pysat_sat
            result = solve_general_native(runtime, nvars, rows, max_conflicts=1_000_000)
            assert result.status != "UNKNOWN"
            assert verify_native_result(nvars, rows, result)
            assert (result.status == "SAT_VERIFIED") == pysat_sat


def test_moderate_random_3sat_differential(runtime):
    pysat = pytest.importorskip("pysat.solvers")
    rng = random.Random(0x51A7)
    for nvars in (20, 35, 50):
        for _ in range(12):
            rows = []
            for _ in range(round(4.25 * nvars)):
                variables = rng.sample(range(1, nvars + 1), 3)
                rows.append(tuple(
                    variable if rng.randrange(2) else -variable
                    for variable in variables
                ))
            with pysat.Solver(name="g4", bootstrap_with=rows) as competitor:
                expected = competitor.solve()
            result = solve_general_native(runtime, nvars, rows, max_conflicts=2_000_000)
            assert result.status != "UNKNOWN"
            assert verify_native_result(nvars, rows, result)
            assert (result.status == "SAT_VERIFIED") == expected


@pytest.mark.skipif(os.name != "posix", reason="GCC/Clang sanitizer contract")
@pytest.mark.parametrize("sanitizer", ("undefined", "address,undefined"))
def test_native_sanitizer_builds_execute_checked_unsat(tmp_path, sanitizer):
    library = build_native_cdcl(tmp_path / sanitizer.replace(",", "-"), sanitizer=sanitizer)
    program = r"""
from itertools import product
import sys
from experiments.general_competition.native_cdcl import NativeCDCLRuntime, solve_general_native, verify_native_result
runtime = NativeCDCLRuntime(sys.argv[1])
clauses = tuple(tuple(v if b else -v for v, b in enumerate(bits, 1)) for bits in product((False, True), repeat=5))
result = solve_general_native(runtime, 5, clauses, max_conflicts=1000000)
assert result.status == 'UNSAT_VERIFIED'
assert verify_native_result(5, clauses, result)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.getcwd()
    environment["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
    if sanitizer.startswith("address"):
        compiler = shutil.which("g++")
        assert compiler is not None
        resolved = subprocess.run(
            [compiler, "-print-file-name=libasan.so"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        environment["LD_PRELOAD"] = resolved
        environment["ASAN_OPTIONS"] = "detect_leaks=0:halt_on_error=1"
    subprocess.run(
        [sys.executable, "-c", program, str(library)],
        cwd=os.getcwd(),
        env=environment,
        check=True,
        timeout=120,
    )


def test_native_proof_ids_preserve_deleted_rows_and_relocation(tmp_path, monkeypatch):
    from experiments.general_competition import native_cdcl

    source = native_cdcl.SOURCE.resolve().as_posix()
    harness = tmp_path / "proof_ids.cpp"
    harness.write_text(
        '#include "' + source + '"\n' + r"""
SPECTRA_EXPORT int spectra_g_test_proof_ids() {
  Solver solver(11);
  // These 2,048 blocking clauses form a real UNSAT formula whose search learns
  // and deletes clauses. The ordinary reduction test independently checks RUP.
  for (unsigned mask = 0; mask < (1u << 11); ++mask) {
    std::vector<int> row;
    for (int var = 1; var <= 11; ++var)
      row.push_back(mask & (1u << (var - 1)) ? var : -var);
    solver.add_initial(row.data(), row.data() + row.size());
  }
  solver.initialize_heap();
  solver.clauses.reserve(solver.clauses.capacity() + 1);
  if (solver.solve(10000000, 100000000) != 20) return 1;
  if (!solver.stats.reductions || !solver.stats.deleted_clauses) return 2;

  std::size_t proof_index = 0;
  std::size_t deleted_rows = 0;
  for (std::size_t id = 0; id < solver.clauses.size(); ++id) {
    const Clause &clause = solver.clauses[id];
    if (!clause.learned) continue;
    if (proof_index >= solver.proof_clause_ids.size() ||
        solver.proof_clause_ids[proof_index++] != static_cast<int>(id)) return 3;
    if (clause.lits.empty()) return 4;
    if (clause.deleted) ++deleted_rows;
  }
  if (deleted_rows != solver.stats.deleted_clauses ||
      proof_index != solver.stats.learned_clauses ||
      solver.proof_clause_ids.size() != proof_index + 1 ||
      solver.proof_clause_ids.back() != -1) return 5;

  const auto encoded = encode_result(solver, 20);
  // Construct the proof payload independently from the append-only clause
  // arena, including deleted rows, to catch dropped or reordered additions.
  if (encoded.size() < 76) return 6;
  std::vector<std::uint8_t> expected(encoded.begin(), encoded.begin() + 76);
  append_u32(expected, static_cast<std::uint32_t>(proof_index + 1));
  for (const Clause &clause : solver.clauses) {
    if (!clause.learned) continue;
    append_u32(expected, static_cast<std::uint32_t>(clause.lits.size()));
    for (int literal : clause.lits) append_i32(expected, literal);
  }
  append_u32(expected, 0);
  if (encoded != expected) return 7;
  // Force relocation explicitly instead of depending on vector growth policy.
  solver.clauses.reserve(solver.clauses.capacity() + 1);
  if (encode_result(solver, 20) != encoded) return 8;

  Solver root(1);
  root.add_clause({}, false);
  if (root.solve(0, 0) != 20 ||
      root.proof_clause_ids != std::vector<int>({-1})) return 9;
  // SGC1 header, UNSAT status, one variable, zero counters, one empty addition.
  std::vector<std::uint8_t> root_expected(84, 0);
  root_expected[0] = 'S'; root_expected[1] = 'G';
  root_expected[2] = 'C'; root_expected[3] = '1';
  root_expected[4] = 20; root_expected[8] = 1; root_expected[76] = 1;
  return encode_result(root, 20) == root_expected ? 0 : 10;
}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(native_cdcl, "SOURCE", harness)
    library = ctypes.CDLL(str(build_native_cdcl(tmp_path / "proof-id-build")))
    check = library.spectra_g_test_proof_ids
    check.argtypes = []
    check.restype = ctypes.c_int
    assert check() == 0


def test_indexed_heap_matches_linear_selection_and_preserves_membership(tmp_path, monkeypatch):
    from experiments.general_competition import native_cdcl

    # Check decision priority independently of the heap implementation, including
    # rounded-to-zero ties, lazy assigned entries, and backtrack reinsertion.
    source = native_cdcl.SOURCE.resolve().as_posix()
    harness = tmp_path / "indexed_heap.cpp"
    harness.write_text(
        '#include "' + source + '"\n' + r"""
static void require(bool result, const char *message) {
  if (!result) throw std::runtime_error(message);
}

static void valid_heap(const Solver &s) {
  std::vector<int> count(static_cast<std::size_t>(s.nvars + 1), 0);
  std::size_t relevant = 0;
  for (int var = 1; var <= s.nvars; ++var) relevant += s.occurs[var] != 0;
  require(s.variable_heap.size() <= relevant, "heap exceeds relevant variable count");
  for (std::size_t index = 0; index < s.variable_heap.size(); ++index) {
    const int var = s.variable_heap[index];
    require(var >= 1 && var <= s.nvars, "heap variable range");
    require(s.occurs[var], "unused variable inserted");
    require(++count[var] == 1, "duplicate heap variable");
    require(s.heap_positions[var] == static_cast<int>(index), "reverse position mismatch");
    if (index) {
      const int parent = s.variable_heap[(index - 1) / 2];
      require(s.activity[parent] > s.activity[var] ||
          (s.activity[parent] == s.activity[var] && parent < var), "heap order mismatch");
    }
  }
  for (int var = 1; var <= s.nvars; ++var) {
    require((s.heap_positions[var] >= 0) == (count[var] == 1), "absent position mismatch");
    if (s.occurs[var] && !s.values[var]) require(count[var] == 1, "missing relevant variable");
  }
}

static int reference(const Solver &s) {
  int best = 0;
  for (int var = 1; var <= s.nvars; ++var) {
    if (!s.occurs[var] || s.values[var]) continue;
    if (!best || s.activity[var] > s.activity[best] ||
        (s.activity[var] == s.activity[best] && var < best)) best = var;
  }
  return best;
}

static int choose_and_assign(Solver &s) {
  const int expected = reference(s);
  const int actual = s.choose_variable();
  require(actual == expected, "decision differs from linear reference");
  if (actual) {
    s.trail_limits.push_back(static_cast<int>(s.trail.size()));
    require(s.enqueue(actual, -1), "decision assignment rejected");
  }
  valid_heap(s);
  return actual;
}

static void test_permutations_and_ties() {
  std::vector<int> priorities{0, 1, 2, 3, 4, 5};
  do {
    for (unsigned mask = 0; mask < 64; ++mask) {
      Solver s(7);
      for (int var = 1; var <= 6; ++var) {
        s.occurs[var] = 1;
        s.activity[var] = (mask & (1u << (var - 1))) ? 0.0 : priorities[var - 1];
      }
      s.activity[7] = 1e99;  // Relevance, not activity, controls membership.
      s.initialize_heap();
      valid_heap(s);
      for (int index = 0; index < 6; ++index) require(choose_and_assign(s) != 0, "early exhaustion");
      require(choose_and_assign(s) == 0, "unused variable selected");
      s.backtrack(0);
      valid_heap(s);
      s.rebuild_heap();
      valid_heap(s);
      for (int index = 0; index < 6; ++index) choose_and_assign(s);
    }
  } while (std::next_permutation(priorities.begin(), priorities.end()));
}

static void test_updates_backtracks_and_rebuilds() {
  Solver s(33);
  for (int var = 1; var <= 32; ++var) s.occurs[var] = 1;
  s.initialize_heap();
  s.var_increment = 0.25;
  std::uint32_t random = 0x417cf091u;
  for (int step = 0; step < 20000; ++step) {
    random = random * 1664525u + 1013904223u;
    const int var = 1 + static_cast<int>((random >> 8) % 32);
    switch (random % 5) {
      case 0:
      case 1:
        s.bump_variable(var);  // Includes assigned variables still in the heap.
        s.push_heap(var);      // Repeated insertion must not create duplicates.
        break;
      case 2:
        if (!choose_and_assign(s)) s.backtrack(0);
        break;
      case 3:
        if (s.decision_level()) s.backtrack(static_cast<int>((random >> 16) % s.decision_level()));
        break;
      case 4:
        s.rebuild_heap();
        break;
    }
    valid_heap(s);
  }
  s.backtrack(0);
  while (choose_and_assign(s)) {}
}

static void test_rescale_and_zero_activity() {
  Solver s(4);
  s.occurs[1] = s.occurs[2] = s.occurs[3] = 1;
  s.activity[1] = 1e-250;
  s.activity[2] = 2e-250;
  s.activity[4] = 1e99;
  s.initialize_heap();
  require(s.variable_heap.front() == 2, "initial tiny activity ordering");
  s.trail_limits.push_back(0);
  require(s.enqueue(3, -1), "rescale assignment rejected");
  s.var_increment = 1e101;
  s.bump_variable(3);
  require(s.activity[1] == 0.0 && s.activity[2] == 0.0, "underflow not exercised");
  require(s.heap_positions[3] == -1, "rescale retained assigned membership");
  valid_heap(s);
  require(choose_and_assign(s) == 1, "underflow tie break changed");
  require(choose_and_assign(s) == 2, "zero activity variable lost");
  require(choose_and_assign(s) == 0, "unused variable included");
  s.backtrack(0);
  valid_heap(s);
  require(choose_and_assign(s) == 3, "assigned rescaled variable not reinserted");
  require(choose_and_assign(s) == 1 && choose_and_assign(s) == 2, "reinserted zero ties changed");
}

static void test_assigned_entry_lifetimes() {
  Solver s(4);
  for (int var = 1; var <= 3; ++var) {
    s.occurs[var] = 1;
    s.activity[var] = 4 - var;
  }
  s.initialize_heap();
  s.trail_limits.push_back(0);
  require(s.enqueue(1, -1), "lazy assignment rejected");
  require(s.heap_positions[1] >= 0, "assignment eagerly removed heap entry");
  s.backtrack(0);
  valid_heap(s);
  require(s.variable_heap.size() == 3, "present entry duplicated on backtrack");
  s.trail_limits.push_back(0);
  require(s.enqueue(1, -1), "second lazy assignment rejected");
  require(choose_and_assign(s) == 2, "assigned root was selected");
  require(s.heap_positions[1] == -1, "assigned root was not discarded");
  s.backtrack(0);
  valid_heap(s);
  require(s.variable_heap.size() == 3, "discarded entry not restored once");
  require(choose_and_assign(s) == 1 && choose_and_assign(s) == 2 &&
      choose_and_assign(s) == 3 && choose_and_assign(s) == 0, "pop/reinsert ordering changed");
}

SPECTRA_EXPORT int spectra_g_test_indexed_heap() {
  try {
    Solver empty(0);
    empty.initialize_heap();
    empty.rebuild_heap();
    require(empty.choose_variable() == 0, "empty solver selection");
    test_permutations_and_ties();
    test_updates_backtracks_and_rebuilds();
    test_rescale_and_zero_activity();
    test_assigned_entry_lifetimes();
    return 0;
  } catch (const std::exception &error) {
    last_error = error.what();
    return 1;
  }
}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(native_cdcl, "SOURCE", harness)
    library = ctypes.CDLL(str(build_native_cdcl(tmp_path / "indexed-heap-build")))
    check = library.spectra_g_test_indexed_heap
    check.argtypes = []
    check.restype = ctypes.c_int
    error = library.spectra_g_native_error
    error.argtypes = []
    error.restype = ctypes.c_char_p
    assert check() == 0, error().decode("utf-8")
