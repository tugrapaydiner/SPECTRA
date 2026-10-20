from __future__ import annotations

import shutil
import sys

import pytest

pytest.importorskip("pysat")

from experiments.general_competition.matched_backend import compare_same_backend
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


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("g++") is None,
    reason="native binary-core builder requires Linux and g++",
)
def test_native_frontend_keeps_original_model_checking(tmp_path):
    runtime = NativeBinaryCore(build_native_binary_core(tmp_path / "native-core"))
    result = compare_same_backend(
        4,
        ((1, -2), (-1, 2), (2, 3, 4), (-3, 4)),
        native_core=runtime,
    )
    assert result.frontend == "native-binary-core-v1"
    assert result.accepted_for_timing
    assert result.plain.model_verified and result.hybrid.model_verified
