from __future__ import annotations

import ctypes
from dataclasses import replace
from itertools import combinations, product
import json
import os
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
