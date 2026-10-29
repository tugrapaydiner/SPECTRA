from __future__ import annotations

import ctypes
from dataclasses import replace
import gc
import hashlib
import importlib
from itertools import combinations, product
import json
import os
from pathlib import Path
import random
import shutil
import struct
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


@pytest.mark.parametrize("nvars,rows,canonical,literals,offsets", [
    (0, (), (), (), (0,)),
    (0, ((),), ((),), (), (0, 0)),
    (3, ((3, -2, 1, 3), (2, -2), (-3,), (-3,)),
     ((-3,), (1, -2, 3)), (-3, 1, -2, 3), (0, 1, 4)),
])
def test_native_input_buffers_keep_their_storage(
        nvars, rows, canonical, literals, offsets):
    from experiments.general_competition.native_cdcl import _pack_cnf

    packed, literal_buffer, offset_buffer = _pack_cnf(
        nvars, (iter(row) for row in rows))
    _pack_cnf(3, ((-1, 2), (3,)))
    gc.collect()
    assert packed == canonical
    assert bytes(literal_buffer) == struct.pack(f"={len(literals)}i", *literals)
    assert bytes(offset_buffer) == struct.pack(f"={len(offsets)}Q", *offsets)


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


@pytest.mark.parametrize("one_pass", (False, True))
def test_direct_native_sat_and_unsat_are_checked_on_original_cnf(runtime, tmp_path, one_pass):
    def inputs(clauses):
        if one_pass:
            return (iter(row) for row in (*clauses, clauses[0], (1, -1)))
        return clauses

    sat_clauses = ((1, 2, 3), (-1, 2), (-2, 3))
    sat = solve_cnf_native(runtime, 3, inputs(sat_clauses))
    assert sat.status == "SAT_VERIFIED"
    assert sat.clauses == len(sat_clauses)
    assert sat.reason == "model"
    assert sat.proof is None
    assert model_satisfies(3, sat_clauses, sat.model)
    assert verify_direct_native_result(3, sat_clauses, sat)

    unsat_clauses = all_blocking_clauses(4)
    unsat = solve_cnf_native(runtime, 4, inputs(unsat_clauses))
    assert unsat.status == "UNSAT_VERIFIED"
    assert unsat.clauses == len(unsat_clauses)
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


def test_drup_replay_matches_clause_simplification(runtime):
    def conflict(clauses, denied):
        pending = [set(row) for row in clauses] + [{-lit} for lit in denied]
        while True:
            if any(not row for row in pending):
                return True
            units = {next(iter(row)) for row in pending if len(row) == 1}
            if not units:
                return False
            if any(-lit in units for lit in units):
                return True
            falsified = {-lit for lit in units}
            pending = [row - falsified for row in pending if row.isdisjoint(units)]

    rng = random.Random(20261029)
    for nvars in (2, 3):
        pool = tuple(
            tuple((index + 1) * sign for index, sign in enumerate(signs) if sign)
            for signs in product((-1, 0, 1), repeat=nvars)
        )
        if nvars == 2:
            formulas = [tuple(row for i, row in enumerate(pool) if mask & (1 << i))
                        for mask in range(1 << len(pool))]
        else:
            nonunits = tuple(row for row in pool if len(row) >= 2)
            formulas = [tuple(row for row in pool if len(row) == 3)]
            formulas.extend(tuple(rng.sample(nonunits, rng.randint(3, 10)))
                            for _ in range(64))
        nonempty = tuple(row for row in pool if row)
        prefixes = [()] + [(row,) for row in nonempty]
        prefixes.extend(tuple(rng.sample(nonempty, 3)) for _ in range(16))
        for clauses in formulas:
            for prefix in prefixes:
                additions = prefix + ((),)
                database = list(clauses)
                expected = True
                for row in additions:
                    if not conflict(database, row):
                        expected = False
                        break
                    database.append(row)
                proof = DRUPProof("spectra.general_drup.v1", additions)
                reversed_proof = replace(
                    proof, additions=tuple(tuple(reversed(row)) for row in additions),
                )
                for checker in (verify_drup_proof, runtime._verify_drup_proof):
                    assert checker(nvars, clauses, proof) == expected
                    assert checker(nvars, clauses, reversed_proof) == expected


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
import ctypes
from itertools import product
import sys
from experiments.general_competition.native_cdcl import (
    NativeCDCLRuntime, solve_cnf_native, solve_general_native,
    verify_direct_native_result, verify_native_result,
)
from experiments.general_competition.test_native_cdcl import (
    test_native_drup_follows_connected_implication_chains,
    test_native_drup_rejects_invalid_addition_after_watch_table_growth,
)
runtime = NativeCDCLRuntime(sys.argv[1])
clauses = tuple(tuple(v if b else -v for v, b in enumerate(bits, 1)) for bits in product((False, True), repeat=5))
for solve, verify in ((solve_general_native, verify_native_result),
                      (solve_cnf_native, verify_direct_native_result)):
    result = solve(runtime, 5, clauses, max_conflicts=1000000)
    assert result.status == 'UNSAT_VERIFIED'
    assert verify(5, clauses, result)
assert runtime._verify_drup is not None
invalid = (ctypes.c_int32 * 1)(-2147483648)
original_offsets = (ctypes.c_uint64 * 2)(0, 1)
proof_offsets = (ctypes.c_uint64 * 2)(0, 0)
assert runtime._verify_drup(1, invalid, 1, original_offsets, 1, None, 0, proof_offsets, 1) == -1
test_native_drup_follows_connected_implication_chains(runtime, 257, False)
test_native_drup_follows_connected_implication_chains(runtime, 257, True)
test_native_drup_rejects_invalid_addition_after_watch_table_growth(runtime)
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
        cxx_runtime = subprocess.run(
            [compiler, "-print-file-name=libstdc++.so"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        # ASan must resolve C++ exception handlers before ctypes loads the kernel.
        environment["LD_PRELOAD"] = os.pathsep.join((resolved, cxx_runtime))
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
    switch (random % 6) {
      case 0:
      case 1:
        s.bump_variable(var);  // Includes assigned variables still in the heap.
        s.push_heap(var);      // Repeated insertion must not create duplicates.
        break;
      case 5:
        s.bump_variable(var);
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
  s.var_increment = 8.0;
  s.bump_variable(1);
  s.bump_variable(2);
  s.bump_variable(2);
  valid_heap(s);
  s.backtrack(0);
  valid_heap(s);
  require(s.variable_heap.size() == 3, "discarded entry not restored once");
  require(choose_and_assign(s) == 2 && choose_and_assign(s) == 1 &&
      choose_and_assign(s) == 3 && choose_and_assign(s) == 0, "backtrack lost updated priorities");
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

@pytest.fixture(scope="module")
def native_proof_core(tmp_path_factory):
    from experiments.general_competition.native_binary_core import NativeBinaryCore, build_native_binary_core

    if sys.platform != "linux" or shutil.which("g++") is None:
        pytest.skip("native binary-core builder currently requires Linux and g++")
    return NativeBinaryCore(build_native_binary_core(
        tmp_path_factory.mktemp("native-original-rup-core")))


def _equivalence_created_units():
    return ((-1, 2), (1, -2), (1, 2), (-3, 4), (3, -4), (3, 4))


def test_original_rup_requires_full_substitution_before_residual(native_proof_core):
    from experiments.general_competition.native_cdcl import _rup_conflict

    original = _equivalence_created_units() + ((-1, -3, 5, 6),)
    reduction = native_proof_core.reduce(6, original)
    assert reduction.class_literals == (1, 1, 2, 2, 3, 4)
    assert reduction.mapping == (0, 0, 0, 0, 1, 2)
    assert reduction.residual_clauses == ((1, 2),)
    canonical = canonical_cnf(6, original)
    assert not _rup_conflict(canonical, (5, 6), 6)
    substituted = ((1,), (3,), (-1, -3, 5, 6))
    assert all(_rup_conflict(canonical, row, 6) for row in substituted)
    assert _rup_conflict(canonical + substituted, (5, 6), 6)


def test_original_rup_composes_native_backend_in_original_namespace(native_proof_core, runtime):
    from experiments.general_competition.native_cdcl import (
        _compose_native_original_rup, _OriginalProofBudgetExceeded,
    )

    original = _equivalence_created_units() + (
        (-1, -3, 5, 6), (-1, -3, 5, -6),
        (-1, -3, -5, 6), (-1, -3, -5, -6),
    )
    reduction = native_proof_core.reduce(6, original)
    assert not reduction.contradiction
    code, model, proof, stats, residual = runtime.solve_residual(
        reduction.reduced_nvars, reduction.residual_clauses)
    assert code == 20 and model == () and proof is not None
    composed = _compose_native_original_rup(6, original, reduction, proof)
    assert verify_drup_proof(6, original, composed.proof)
    assert composed.prefix_clauses == 6
    assert composed.prefix_literals == 10
    assert composed.backend_clauses == stats.learned_clauses + 1
    assert composed.backend_literals == stats.learned_literals
    assert composed.frontend_terminal_clauses == 0
    assert len(composed.proof.additions) == composed.prefix_clauses + composed.backend_clauses
    expected_suffix = tuple(tuple(
        (abs(lit) + 4) if lit > 0 else -(abs(lit) + 4) for lit in row
    ) for row in proof.additions)
    assert composed.proof.additions[composed.prefix_clauses:] == expected_suffix
    assert composed.proof.additions.count(()) == 1
    assert not verify_drup_proof(6, original, DRUPProof(
        proof.schema, composed.proof.additions[2:]))  # Remove new S units, keep R.
    serialized = "".join("".join(f"{lit} " for lit in row) + "0\n"
                         for row in composed.proof.additions)
    assert composed.proof_bytes == len(serialized.encode("ascii"))
    # A cap sufficient for either prefix or backend alone is not a combined cap.
    with pytest.raises(_OriginalProofBudgetExceeded, match="proof_literal_budget"):
        _compose_native_original_rup(
            6, original, reduction, proof,
            max_proof_literals=max(composed.prefix_literals, composed.backend_literals))


@pytest.mark.parametrize("nvars,original,kind,expected", (
    (2, ((1, 2), (1, -2), (-1, 2), (-1, -2)), "binary", ((1,), ())),
    (2, ((1,), (1, 2), (1, -2), (-1, 2), (-1, -2)), "binary", ((),)),
    (1, ((1,), (-1,)), "binary", ((),)),
    (5, _equivalence_created_units() + ((-5,), (-1, -3, 5)), "unit", None),
    (0, ((),), "unit", ((),)),
))
def test_original_rup_frontend_contradiction_branches(
        native_proof_core, nvars, original, kind, expected):
    from experiments.general_competition.native_cdcl import _compose_native_original_rup

    reduction = native_proof_core.reduce(nvars, original)
    assert reduction.contradiction and reduction.conflict_kind == kind
    composed = _compose_native_original_rup(nvars, original, reduction)
    assert composed.backend_clauses == composed.backend_literals == 0
    assert composed.frontend_terminal_clauses == 1
    assert composed.prefix_clauses == len(composed.proof.additions) - 1
    assert composed.prefix_literals == sum(map(len, composed.proof.additions))
    assert verify_drup_proof(nvars, original, composed.proof)
    if expected is not None:
        assert composed.proof.additions == expected
    if () not in original and expected != ((),):
        assert not verify_drup_proof(nvars, original, DRUPProof(composed.proof.schema, ((),)))


def test_original_rup_exact_certificate_budgets(native_proof_core):
    from experiments.general_competition.native_cdcl import _compose_native_original_rup, _OriginalProofBudgetExceeded

    original = ((1, 2), (1, -2), (-1, 2), (-1, -2))
    reduction = native_proof_core.reduce(2, original)
    exact = dict(max_proof_clauses=2, max_proof_literals=1, max_proof_bytes=6)
    composed = _compose_native_original_rup(2, original, reduction, **exact)
    assert composed.proof_bytes == 6
    for name, reason in (
        ("max_proof_clauses", "proof_clause_budget"),
        ("max_proof_literals", "proof_literal_budget"),
        ("max_proof_bytes", "proof_byte_budget"),
    ):
        limited = {**exact, name: exact[name] - 1}
        with pytest.raises(_OriginalProofBudgetExceeded) as error:
            _compose_native_original_rup(2, original, reduction, **limited)
        assert error.value.reason == reason
    empty = native_proof_core.reduce(0, ((),))
    assert _compose_native_original_rup(
        0, ((),), empty, max_proof_clauses=1, max_proof_literals=0,
        max_proof_bytes=2).proof.additions == ((),)
    with pytest.raises(ValueError, match="exact uint64"):
        _compose_native_original_rup(2, original, reduction, max_proof_clauses=True)


def test_original_rup_rejects_malformed_composition_evidence(native_proof_core, runtime):
    from experiments.general_competition.native_cdcl import _compose_native_original_rup

    original = _equivalence_created_units() + tuple(
        (-1, -3) + tuple(5 + index if bit else -(5 + index)
                        for index, bit in enumerate(bits))
        for bits in product((False, True), repeat=2)
    )
    reduction = native_proof_core.reduce(6, original)
    code, _, proof, _, _ = runtime.solve_residual(
        reduction.reduced_nvars, reduction.residual_clauses)
    assert code == 20 and proof is not None
    for bad in (
        replace(reduction, mapping=(0, 0, 0, 0, 1, 1)),
        replace(reduction, class_literals=(-1, -1, 2, 2, 3, 4)),
        replace(reduction, fixed=(True, False, True, True, False, False)),
    ):
        with pytest.raises(ValueError):
            _compose_native_original_rup(6, original, bad, proof)
    for bad_rows in (((1, 1), ()), ((1, -1), ()), ((0,), ()), ((3,), ()), ((), ())):
        with pytest.raises(ValueError):
            _compose_native_original_rup(
                6, original, reduction, DRUPProof(proof.schema, bad_rows))


def test_original_rup_does_not_mistake_rat_only_addition_for_rup():
    from experiments.general_competition.native_cdcl import PROOF_SCHEMA

    original = ((1, 2), (1, -2), (-1, 2), (-1, -2))
    # Pure unused variable 3 makes (3) vacuously RAT, but not RUP. This stream
    # must not pass merely because a DRAT checker can validate the refutation.
    assert not verify_drup_proof(3, original, DRUPProof(PROOF_SCHEMA, ((3,), (1,), ())))


@pytest.mark.parametrize("fixed_class", (False, True))
def test_original_rup_signed_multihop_free_and_fixed_classes(
        native_proof_core, runtime, fixed_class):
    from experiments.general_competition.native_cdcl import _compose_native_original_rup

    # 1 = -2 = 3, with 1 as the positive original representative.
    equivalences = ((1, 2), (-1, -2), (2, 3), (-2, -3))
    if fixed_class:
        original = equivalences + ((3,),) + tuple(
            (2,) + tuple(abs(lit) + 3 if lit > 0 else -(abs(lit) + 3) for lit in row)
            for row in all_blocking_clauses(2)
        )
        expected_mapping = (0, 0, 0, 1, 2)
        expected_fixed = (True, False, True, False, False)
        representatives = (4, 5)
        expected_substitution = ((1,),) + tuple(
            (-1,) + tuple(abs(lit) + 3 if lit > 0 else -(abs(lit) + 3) for lit in row)
            for row in all_blocking_clauses(2)
        )
    else:
        def original_literal(literal):
            target = (-2, 4, 5)[abs(literal) - 1]
            return target if literal > 0 else -target
        original = equivalences + tuple(
            tuple(map(original_literal, row)) for row in all_blocking_clauses(3))
        expected_mapping = (1, -1, 1, 2, 3)
        expected_fixed = (False,) * 5
        representatives = (1, 4, 5)
        expected_substitution = tuple(tuple(
            representatives[abs(lit) - 1] if lit > 0 else -representatives[abs(lit) - 1]
            for lit in row) for row in all_blocking_clauses(3))
    reduction = native_proof_core.reduce(5, original)
    assert not reduction.contradiction
    assert reduction.class_literals == (1, -1, 1, 2, 3)
    assert reduction.mapping == expected_mapping
    assert reduction.fixed == expected_fixed
    code, _, proof, stats, residual = runtime.solve_residual(
        reduction.reduced_nvars, reduction.residual_clauses)
    assert code == 20 and proof is not None
    composed = _compose_native_original_rup(5, original, reduction, proof)
    substituted = canonical_cnf(5, expected_substitution)
    assert composed.proof.additions[:len(substituted)] == substituted
    expected_suffix = tuple(tuple(
        representatives[abs(lit) - 1] if lit > 0 else -representatives[abs(lit) - 1]
        for lit in row) for row in proof.additions)
    assert composed.proof.additions[composed.prefix_clauses:] == expected_suffix
    assert composed.backend_literals == stats.learned_literals
    # In the free case the embedded residual is already entirely present in S.
    assert composed.prefix_clauses == (9 if fixed_class else 8)
    assert composed.prefix_literals == (21 if fixed_class else 24)
    assert verify_drup_proof(5, original, composed.proof)


def test_native_original_signed_lifting_keeps_unused_header_variables(native_proof_core, runtime):
    original = ((1, 2), (-1, -2), (2, 3), (-2, -3), (3,), (2, 4, 5))
    reduction = native_proof_core.reduce(9, original)
    assert not reduction.contradiction
    code, model, proof, _, _ = runtime.solve_residual(
        reduction.reduced_nvars, reduction.residual_clauses)
    assert code == 10 and proof is None
    lifted = reduction.lift(model)
    assert len(lifted) == 9 and all(type(value) is bool for value in lifted)
    assert lifted[:3] == (True, False, True)
    assert model_satisfies(9, original, lifted)
    assert all(any(lifted[abs(lit) - 1] == (lit > 0) for lit in row) for row in original)


def test_original_rup_large_original_ids_have_exact_byte_limits(native_proof_core, runtime, tmp_path):
    from experiments.general_competition.native_cdcl import (
        _compose_native_original_rup, _OriginalProofBudgetExceeded,
    )

    # Dense backend variables 1,2,3 map to original IDs 98,99,100.
    original = tuple((var,) for var in range(1, 98)) + tuple(tuple(
        abs(lit) + 97 if lit > 0 else -(abs(lit) + 97) for lit in row
    ) for row in all_blocking_clauses(3))
    reduction = native_proof_core.reduce(100, original)
    assert not reduction.contradiction and reduction.reduced_nvars == 3
    assert reduction.mapping == (0,) * 97 + (1, 2, 3)
    code, _, proof, stats, _ = runtime.solve_residual(3, reduction.residual_clauses)
    assert code == 20 and proof is not None and stats.learned_literals > 0
    composed = _compose_native_original_rup(100, original, reduction, proof)
    assert composed.prefix_clauses == composed.prefix_literals == 0
    path = write_drup(tmp_path / "large-ids.drat", composed.proof)
    exact_bytes = len(path.read_bytes())
    assert composed.proof_bytes == exact_bytes
    suffix = composed.proof.additions[composed.prefix_clauses:]
    raw_bytes = sum(sum(len(str(lit)) + 1 for lit in row) + 2 for row in proof.additions)
    mapped_bytes = sum(sum(len(str(lit)) + 1 for lit in row) + 2 for row in suffix)
    assert mapped_bytes > raw_bytes
    assert _compose_native_original_rup(
        100, original, reduction, proof, max_proof_bytes=exact_bytes
    ) == composed
    with pytest.raises(_OriginalProofBudgetExceeded, match="proof_byte_budget"):
        _compose_native_original_rup(
            100, original, reduction, proof, max_proof_bytes=exact_bytes - 1)
    assert _compose_native_original_rup(
        100, original, reduction, proof,
        max_proof_clauses=composed.backend_clauses,
        max_proof_literals=composed.backend_literals,
        max_proof_bytes=exact_bytes) == composed
    assert verify_drup_proof(100, original, composed.proof)


def test_original_rup_rejects_bad_evidence_before_tiny_budget(native_proof_core, runtime):
    from experiments.general_competition.native_cdcl import _compose_native_original_rup

    original = _equivalence_created_units() + tuple(
        (-1, -3) + tuple(abs(lit) + 4 if lit > 0 else -(abs(lit) + 4) for lit in row)
        for row in all_blocking_clauses(2))
    reduction = native_proof_core.reduce(6, original)
    code, _, proof, _, _ = runtime.solve_residual(2, reduction.residual_clauses)
    assert code == 20 and proof is not None
    tiny = dict(max_proof_clauses=0, max_proof_literals=0, max_proof_bytes=0)
    with pytest.raises(TypeError, match="NativeReduction required"):
        _compose_native_original_rup(6, original, object(), proof, **tiny)
    for bad in (
        replace(reduction, mapping=(0, 0, 0, 0, 1, 1)),
        replace(reduction, mapping=[0, 0, 0, 0, 1, 2]),
        replace(reduction, fixed=(True, False, True, True, False, False)),
        replace(reduction, class_literals=(True, 1, 2, 2, 3, 4)),
        replace(reduction, residual_clauses=((3,),)),
    ):
        with pytest.raises(ValueError):
            _compose_native_original_rup(6, original, bad, proof, **tiny)
    for bad in (
        None, object(), DRUPProof("wrong", ((),)),
        DRUPProof(proof.schema, ((1,),)),
        DRUPProof(proof.schema, ([1], ())),
        DRUPProof(proof.schema, ((True,), ())),
        DRUPProof(proof.schema, ((1, 1), ())),
        DRUPProof(proof.schema, ((1, -1), ())),
        DRUPProof(proof.schema, ((), ())),
    ):
        with pytest.raises(ValueError):
            _compose_native_original_rup(6, original, reduction, bad, **tiny)
    quotient_original = _equivalence_created_units() + ((-5,), (-1, -3, 5))
    quotient = native_proof_core.reduce(5, quotient_original)
    assert quotient.conflict_kind == "unit"
    with pytest.raises(ValueError, match="quotient conflict variable"):
        _compose_native_original_rup(
            5, quotient_original, replace(quotient, conflict_variable=1), **tiny)


def test_original_empty_precedes_unrelated_binary_conflict(native_proof_core):
    from experiments.general_competition.native_cdcl import _compose_native_original_rup

    original = ((), (1, 2), (1, -2), (-1, 2), (-1, -2))
    reduction = native_proof_core.reduce(2, original)
    assert reduction.conflict_kind == "binary"
    composed = _compose_native_original_rup(2, original, reduction)
    assert composed.proof.additions == ((),)
    assert composed.prefix_clauses == composed.prefix_literals == 0
    assert composed.frontend_terminal_clauses == 1 and composed.proof_bytes == 2
    assert verify_drup_proof(2, original, composed.proof)


def test_original_rup_well_shaped_false_reduction_is_not_certified(native_proof_core):
    from experiments.general_competition.native_cdcl import _compose_native_original_rup

    original = _equivalence_created_units() + ((-1, -3, 5, 6),)
    reduction = native_proof_core.reduce(6, original)
    false_reduction = replace(
        reduction, contradiction=True, conflict_kind="unit", conflict_variable=0,
        reduced_nvars=0, residual_clauses=((),), mapping=(0,) * 6, fixed=(False,) * 6)
    # Shape validation is not logical validation. The helper must not be exposed
    # as a verified solver result: this well-shaped false claim still fails RUP.
    composed = _compose_native_original_rup(6, original, false_reduction)
    assert not verify_drup_proof(6, original, composed.proof)


@pytest.mark.skipif(
    os.name != "posix" or not os.environ.get("DRAT_TRIM"),
    reason="bounded POSIX execution and pinned RUP-only checker are required",
)
@pytest.mark.parametrize("case", (
    "identity", "identity-repeated-backend", "mixed", "mixed-permuted", "false-unit",
))
def test_original_rup_deduplicated_prefix_external_verdict(
        native_proof_core, runtime, tmp_path, case):
    from experiments.general_competition.campaign import run_process
    from experiments.general_competition.native_cdcl import _compose_native_original_rup

    if case.startswith("identity"):
        nvars, original = 3, all_blocking_clauses(3)
        back = (1, 2, 3)
        expected_prefix = ()
    elif case.startswith("mixed"):
        # Half of S is already original; all of the embedded R is in original
        # or S. A signed class makes both kinds of overlap consequential.
        nvars = 5
        equivalences = ((1, 2), (-1, -2), (2, 3), (-2, -3))  # 1 = -2 = 3.
        blocks = all_blocking_clauses(3)
        original = equivalences + tuple(
            tuple(target if lit > 0 else -target
                  for lit, target in zip(row, (1 if index < 4 else -2, 4, 5)))
            for index, row in enumerate(blocks))
        expected_prefix = canonical_cnf(5, tuple(
            (1,) + tuple(target if lit > 0 else -target
                         for lit, target in zip(row[1:], (4, 5)))
            for row in blocks[4:]))
        back = (5, -1, 4) if case == "mixed-permuted" else (1, 4, 5)
    else:
        nvars = 6
        original = _equivalence_created_units() + ((-1, -3, 5, 6),)
        expected_prefix = ((1,), (3,))

    reduction = native_proof_core.reduce(nvars, original)
    assert not reduction.contradiction
    if case == "false-unit":
        # A validly shaped but false claim about this SAT formula must still
        # fail independent checking after duplicate clauses are omitted.
        reduction = replace(
            reduction, contradiction=True, conflict_kind="unit", conflict_variable=0,
            reduced_nvars=0, residual_clauses=((),), mapping=(0,) * nvars,
            fixed=(False,) * nvars)
        proof = None
        expected_suffix = ((),)
    else:
        if case == "mixed-permuted":
            # All blocking clauses are invariant under this signed permutation,
            # but embedded row order differs from canonical original row order.
            reduction = replace(reduction, mapping=(-2, 2, -2, 3, 1))
        code, _, proof, _, _ = runtime.solve_residual(
            reduction.reduced_nvars, reduction.residual_clauses)
        assert code == 20 and proof is not None
        if case == "identity-repeated-backend":
            repeated = reduction.residual_clauses[0]
            proof = DRUPProof(proof.schema, (repeated, repeated) + proof.additions)
        expected_suffix = tuple(tuple(
            back[abs(lit) - 1] if lit > 0 else -back[abs(lit) - 1] for lit in row
        ) for row in proof.additions)

    composed = _compose_native_original_rup(nvars, original, reduction, proof)
    assert composed.proof.additions == expected_prefix + expected_suffix
    assert composed.prefix_clauses == len(expected_prefix)
    assert composed.prefix_literals == sum(map(len, expected_prefix))
    assert composed.backend_clauses == (0 if proof is None else len(proof.additions))
    assert composed.backend_literals == (0 if proof is None else sum(map(len, proof.additions)))
    assert composed.frontend_terminal_clauses == (1 if proof is None else 0)
    assert composed.proof.additions.count(()) == 1
    formula = write_dimacs(tmp_path / "original.cnf", nvars, original)
    proof_path = write_drup(tmp_path / "original.drat", composed.proof)
    assert proof_path.stat().st_size == composed.proof_bytes
    prefix_bytes = sum(map(len, proof_path.read_bytes().splitlines(keepends=True)[
        :composed.prefix_clauses]))
    assert prefix_bytes == (0 if case.startswith("identity") else
                            36 if case.startswith("mixed") else 8)
    assert _compose_native_original_rup(
        nvars, original, reduction, proof,
        max_proof_clauses=len(composed.proof.additions),
        max_proof_literals=composed.prefix_literals + composed.backend_literals,
        max_proof_bytes=composed.proof_bytes) == composed
    checked = run_process(
        [str(Path(os.environ["DRAT_TRIM"]).resolve()), str(formula), str(proof_path),
         "-f", "-U", "-I"], tmp_path / "check", timeout=30,
        memory_mib=512, output_mib=8)
    assert not checked["timed_out"] and checked["spawn_error"] is None, checked
    lines = {
        line.strip() for line in (tmp_path / "check" / "stdout.txt").read_text().splitlines()
    }
    if case == "false-unit":
        assert checked["returncode"] != 0 and "s VERIFIED" not in lines, checked
    else:
        assert checked["returncode"] == 0 and "s VERIFIED" in lines, checked


@pytest.mark.skipif(
    os.name != "posix" or not os.environ.get("DRAT_TRIM"),
    reason="bounded POSIX execution and pinned RUP-only checker are required",
)
def test_original_rup_genuine_prefix_retains_deleted_backend_additions(
        native_proof_core, runtime, tmp_path):
    from experiments.general_competition.campaign import run_process

    # Keep both the stress solve/composition and independent check supervised.
    # Do not run the simple Python replay on this large certificate in pytest.
    program = r"""
import json
from itertools import product
from pathlib import Path
import sys
from experiments.general_competition.binary_core import canonical_cnf
from experiments.general_competition.native_binary_core import NativeBinaryCore
from experiments.general_competition.native_cdcl import (
    NativeCDCLRuntime, _compose_native_original_rup, _rup_conflict, write_drup,
)
from experiments.general_competition.proof_backend import write_dimacs
core = NativeBinaryCore(sys.argv[1])
backend = NativeCDCLRuntime(sys.argv[2])
out = Path(sys.argv[3])
blocks = tuple(tuple(v if bit else -v for v, bit in enumerate(bits, 1))
               for bits in product((False, True), repeat=11))
equivalences = ((-1,2), (1,-2), (1,2), (-3,4), (3,-4), (3,4))
original = equivalences + tuple((-1,-3) + tuple(
    abs(lit) + 4 if lit > 0 else -(abs(lit) + 4) for lit in row) for row in blocks)
reduction = core.reduce(15, original)
assert not reduction.contradiction and reduction.reduced_nvars == 11
assert reduction.residual_clauses == canonical_cnf(11, blocks)
code, model, proof, stats, residual = backend.solve_residual(
    11, reduction.residual_clauses, max_conflicts=10_000_000,
    max_learned_literals=100_000_000)
assert code == 20 and model == () and proof is not None
assert stats.reductions >= 1 and stats.deleted_clauses >= 1
composed = _compose_native_original_rup(15, original, reduction, proof)
assert composed.prefix_clauses == 2050 and composed.prefix_literals == 22530
assert composed.backend_clauses == stats.learned_clauses + 1
assert composed.backend_literals == stats.learned_literals
expected = tuple(tuple(abs(lit) + 4 if lit > 0 else -(abs(lit) + 4)
                       for lit in row) for row in proof.additions)
assert composed.proof.additions[composed.prefix_clauses:] == expected
assert composed.proof.additions.count(()) == 1
assert not _rup_conflict(canonical_cnf(15, original), composed.proof.additions[2], 15)
write_dimacs(out / 'original.cnf', 15, original)
proof_path = write_drup(out / 'original.drat', composed.proof)
assert proof_path.stat().st_size == composed.proof_bytes
print(json.dumps({'reductions': stats.reductions, 'deleted': stats.deleted_clauses,
                  'prefix_clauses': composed.prefix_clauses,
                  'backend_clauses': composed.backend_clauses}))
"""
    solved = run_process(
        [sys.executable, "-c", program, str(native_proof_core.path), str(runtime.path), str(tmp_path)],
        tmp_path / "stress-solve", timeout=180, memory_mib=4096, output_mib=128)
    assert not solved["timed_out"] and solved["spawn_error"] is None, solved
    assert solved["returncode"] == 0, (tmp_path / "stress-solve" / "stderr.txt").read_text()
    observed = json.loads((tmp_path / "stress-solve" / "stdout.txt").read_text())
    assert observed["reductions"] >= 1 and observed["deleted"] >= 1
    assert observed["prefix_clauses"] == 2050 and observed["backend_clauses"] > 1
    checked = run_process(
        [str(Path(os.environ["DRAT_TRIM"]).resolve()), str(tmp_path / "original.cnf"),
         str(tmp_path / "original.drat"), "-f", "-U", "-I"],
        tmp_path / "stress-check", timeout=120, memory_mib=4096, output_mib=128)
    assert not checked["timed_out"] and checked["spawn_error"] is None, checked
    assert checked["returncode"] == 0, checked
    assert "s VERIFIED" in {
        line.strip() for line in (tmp_path / "stress-check" / "stdout.txt").read_text().splitlines()
    }


@pytest.mark.parametrize("additions,limits", [
    ((), {}), (((), ()), {}), (((1,),), {}),
    (((True,), ()), {}), (((0,), ()), {}), (((3,), ()), {}),
    (((-3,), ()), {}), (((2**63,), ()), {}), (((-(2**63),), ()), {}),
    (((1, 1), ()), {}), (((1, -1), ()), {}), (([1], ()), {}),
    (((),), {"max_clauses": 0}), (((),), {"max_clauses": True}),
    (((),), {"max_literals": -1}), (((),), {"max_literals": False}),
    (((1,), ()), {"max_literals": 0}),
])
def test_native_drup_rejects_malformed_proofs(runtime, additions, limits):
    # Contradictory originals must not make a malformed proof acceptable.
    proof = DRUPProof("spectra.general_drup.v1", additions)
    for checker in (verify_drup_proof, runtime._verify_drup_proof):
        assert not checker(2, ((1,), (-1,)), proof, **limits)


@pytest.mark.parametrize("nvars,original,original_offsets,proof,proof_offsets", [
    (10_000_001, [], [0], [], [0, 0]),
    (1, [0], [0, 1], [], [0, 0]),
    (1, [-2147483648], [0, 1], [], [0, 0]),
    (1, [2], [0, 1], [], [0, 0]),
    (1, [1, -1], [0, 2], [], [0, 0]),
    (2, [2, 1], [0, 2], [], [0, 0]),
    (1, [1], [1, 1], [], [0, 0]),
    (1, [1], [0, 0], [], [0, 0]),
    (1, [1], [0, 2, 1], [], [0, 0]),
    (1, [], [0, 0], [1, 1], [0, 2, 2]),
    (1, [], [0, 0], [0], [0, 1, 1]),
    (1, [], [0, 0], [], [0, 1, 0]),
])
def test_native_drup_abi_rejects_malformed_buffers(
        runtime, nvars, original, original_offsets, proof, proof_offsets):
    def buffer(kind, values):
        return (kind * len(values))(*values)

    code = runtime._verify_drup(
        nvars, buffer(ctypes.c_int32, original), len(original),
        buffer(ctypes.c_uint64, original_offsets), len(original_offsets) - 1,
        buffer(ctypes.c_int32, proof), len(proof),
        buffer(ctypes.c_uint64, proof_offsets), len(proof_offsets) - 1,
    )
    assert code == -1
    assert runtime._library.spectra_g_native_error()


def test_native_drup_abi_empty_and_null_buffers(runtime):
    offsets = (ctypes.c_uint64 * 2)(0, 0)
    verify = runtime._verify_drup
    assert verify(0, None, 0, offsets, 1, None, 0, offsets, 1) == 1
    assert verify(0, None, 0, offsets, 0, None, 0, offsets, 1) == 0
    assert verify(0, None, 0, offsets, 1, None, 0, offsets, 0) == 0
    early_empty = (ctypes.c_uint64 * 3)(0, 0, 0)
    assert verify(0, None, 0, offsets, 1, None, 0, early_empty, 2) == 0
    nonempty = (ctypes.c_int32 * 1)(1)
    nonempty_offsets = (ctypes.c_uint64 * 2)(0, 1)
    assert verify(1, None, 0, offsets, 1, nonempty, 1, nonempty_offsets, 1) == 0
    assert verify(0, None, 0, None, 0, None, 0, offsets, 1) == -1
    assert verify(1, None, 1, offsets, 1, None, 0, offsets, 1) == -1
    assert verify(0, None, 0, offsets, 1, None, 0, None, 1) == -1
    assert verify(1, None, 0, offsets, 1, None, 1, offsets, 1) == -1
    too_many = ctypes.c_size_t(-1).value
    assert verify(0, None, 0, offsets, too_many, None, 0, offsets, 1) == -1
    assert verify(0, None, 0, offsets, 1, None, 0, offsets, too_many) == -1
    assert verify(0, None, 0, offsets, 1, None, 0, offsets, 1) == 1
    assert runtime._library.spectra_g_native_error() == b""


def test_direct_native_replay_checks_failure_and_fallback(runtime, monkeypatch):
    clauses = ((1,), (-1,))
    monkeypatch.setattr(runtime, "_verify_drup", lambda *_: 0)
    with pytest.raises(AssertionError, match="independent replay"):
        solve_cnf_native(runtime, 1, clauses)
    monkeypatch.setattr(runtime, "_verify_drup", lambda *_: -1)
    with pytest.raises(RuntimeError, match="native RUP failure"):
        solve_cnf_native(runtime, 1, clauses)
    monkeypatch.setattr(runtime, "_verify_drup", None)
    result = solve_cnf_native(runtime, 1, clauses)
    assert result.status == "UNSAT_VERIFIED"
    assert verify_direct_native_result(1, clauses, result)


@pytest.mark.parametrize("nvars", (8, 257))
@pytest.mark.parametrize("reverse", (False, True))
def test_native_drup_follows_connected_implication_chains(runtime, nvars, reverse):
    def label(literal):
        if not reverse:
            return literal
        variable = nvars + 1 - abs(literal)
        return variable if literal > 0 else -variable

    for disconnected in (False, True):
        clauses = [(1, 2), (-1, 2), (1, -2), (-1, -2)]
        for variable in range(2, nvars):
            if disconnected and variable == nvars // 2:
                continue
            clauses.extend(((-variable, variable + 1), (variable, -variable - 1)))
        original = tuple(tuple(map(label, row)) for row in clauses)
        proof = DRUPProof("spectra.general_drup.v1", ((label(nvars),), ()))
        # An UNSAT core does not justify a disconnected, non-RUP addition.
        for checker in (verify_drup_proof, runtime._verify_drup_proof):
            assert checker(nvars, original, proof) == (not disconnected)


def test_native_drup_rejects_invalid_addition_after_watch_table_growth(runtime):
    original = tuple((1, 2, variable) for variable in range(3, 130))
    proof = DRUPProof("spectra.general_drup.v1", ((1, 2, 3), (1, 2), ()))
    # Denying 1 and 2 moves watches onto 127 previously unindexed literals.
    for checker in (verify_drup_proof, runtime._verify_drup_proof):
        assert not checker(129, original, proof)
