from __future__ import annotations

import shutil
import sys

import pytest

pytest.importorskip("pysat")

from experiments.general_competition.matched_backend import (
    PreparedFormula,
    compare_prepared_backend,
    compare_same_backend,
    prepare_matched_formula,
)
from experiments.general_competition.native_binary_core import (
    NativeBinaryCore,
    build_native_binary_core,
)


def test_same_backend_sat_models_are_checked_before_timing_acceptance():
    result = compare_same_backend(
        4,
        ((1, -2), (-1, 2), (2, 3, 4), (-3, 4)),
    )
    assert result.answers_agree
    assert result.accepted_for_timing
    assert result.plain.status == result.hybrid.status == "SAT_CHECKED"
    assert result.plain.model_verified and result.hybrid.model_verified
    assert result.residual_variables <= result.original_variables
    assert result.residual_clauses <= result.original_clauses
    assert (
        result.empty_input_clauses
        + result.unit_input_clauses
        + result.binary_input_clauses
        + result.long_input_clauses
        == result.original_clauses
    )


def test_prepared_formula_matches_the_direct_path():
    clauses = ((1, -2), (-1, 2), (2, 3, 4), (-3, 4))
    prepared = prepare_matched_formula(4, clauses)
    direct = compare_same_backend(4, clauses)
    repeated = compare_prepared_backend(prepared)
    assert prepared == PreparedFormula(
        4,
        prepared.clauses,
        prepared.empty_clauses,
        prepared.unit_clauses,
        prepared.binary_clauses,
        prepared.long_clauses,
    )
    assert direct.answers_agree == repeated.answers_agree
    assert direct.accepted_for_timing == repeated.accepted_for_timing
    assert direct.plain.status == repeated.plain.status == "SAT_CHECKED"
    assert direct.hybrid.status == repeated.hybrid.status == "SAT_CHECKED"
    assert direct.original_variables == repeated.original_variables
    assert direct.original_clauses == repeated.original_clauses
    assert direct.residual_variables == repeated.residual_variables
    assert direct.residual_clauses == repeated.residual_clauses


def test_residual_unsat_is_not_promoted_without_a_standard_proof():
    clauses = tuple(
        tuple(variable if mask & (1 << (variable - 1)) else -variable
              for variable in range(1, 4))
        for mask in range(8)
    )
    result = compare_same_backend(3, clauses, order="hybrid-first")
    assert result.answers_agree
    assert result.plain.status == result.hybrid.status == "UNSAT_REPORTED"
    assert not result.accepted_for_timing


def test_invalid_execution_order_is_rejected():
    with pytest.raises(ValueError, match="order"):
        compare_same_backend(1, ((1,),), order="random")
    with pytest.raises(ValueError, match="prepared"):
        compare_prepared_backend(object())


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("g++") is None,
    reason="native binary-core builder requires Linux and g++",
)
def test_native_frontend_reuses_one_packed_formula_and_checks_models(tmp_path):
    runtime = NativeBinaryCore(build_native_binary_core(tmp_path / "native-core"))
    prepared = prepare_matched_formula(
        4,
        ((1, -2), (-1, 2), (2, 3, 4), (-3, 4)),
    )
    native_prepared = runtime.prepare_canonical(prepared.nvars, prepared.clauses)
    result = compare_prepared_backend(
        prepared,
        native_core=runtime,
        native_prepared=native_prepared,
    )
    assert result.frontend == "native-binary-core-prepared-v1"
    assert result.accepted_for_timing
    assert result.plain.model_verified and result.hybrid.model_verified

    other = runtime.prepare_canonical(1, ((1,),))
    with pytest.raises(ValueError, match="differs"):
        compare_prepared_backend(
            prepared,
            native_core=runtime,
            native_prepared=other,
        )
