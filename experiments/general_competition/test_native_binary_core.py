from __future__ import annotations

from itertools import product
import json
import random
import shutil
import sys

import pytest

from experiments.general_competition.binary_core import (
    build_binary_core,
    model_satisfies,
    verify_binary_core,
)
from experiments.general_competition.native_binary_core import (
    NativeBinaryCore,
    assert_matches_python,
    build_native_binary_core,
)


def cube_clauses():
    return tuple(
        tuple(
            variable if signs[variable - 1] else -variable
            for variable in range(1, 4)
        )
        for signs in product((False, True), repeat=3)
    )


@pytest.fixture(scope="session")
def native_runtime(tmp_path_factory):
    if sys.platform != "linux" or shutil.which("g++") is None:
        pytest.skip("the native general-CNF core currently requires Linux and g++")
    library = build_native_binary_core(
        tmp_path_factory.mktemp("native-general-core")
    )
    return NativeBinaryCore(library)


def check_equal(runtime, nvars, clauses):
    frozen = tuple(tuple(row) for row in clauses)
    reference = build_binary_core(nvars, frozen)
    assert verify_binary_core(nvars, frozen, reference)
    native = runtime.reduce(nvars, frozen)
    assert_matches_python(nvars, frozen, native, reference)

    if not reference.contradiction and reference.reduced_nvars <= 6:
        for model in product((False, True), repeat=reference.reduced_nvars):
            assert native.lift(model) == reference.lift(model)
            if model_satisfies(
                reference.reduced_nvars,
                reference.residual_clauses,
                model,
            ):
                assert model_satisfies(nvars, frozen, native.lift(model))
    return native, reference


def test_all_three_variable_long_clause_subsets_match(native_runtime):
    pool = cube_clauses()
    for mask in range(1 << len(pool)):
        clauses = tuple(
            row for index, row in enumerate(pool) if mask & (1 << index)
        )
        check_equal(native_runtime, 3, clauses)


def test_random_units_binary_long_duplicates_and_tautologies_match(native_runtime):
    for seed in range(256):
        rng = random.Random(seed)
        nvars = rng.randint(1, 8)
        clauses = []
        for _ in range(rng.randint(0, 32)):
            row = []
            for _ in range(rng.randint(0, 7)):
                variable = rng.randint(1, nvars)
                row.append(variable if rng.getrandbits(1) else -variable)
            if row and rng.randrange(5) == 0:
                row.append(row[-1])
            if row and rng.randrange(7) == 0:
                row.append(-row[0])
            clauses.append(tuple(row))
        check_equal(native_runtime, nvars, clauses)


def test_zero_variable_and_direct_conflict_cases_match(native_runtime):
    for clauses in ((), ((),)):
        check_equal(native_runtime, 0, clauses)
    check_equal(native_runtime, 1, ((1,), (-1,)))
    check_equal(native_runtime, 4, ((1, -2), (-1, 2), (2,), (-4, 3), (-3, 4)))


def test_wrapper_rejects_non_exact_and_out_of_range_inputs(native_runtime):
    with pytest.raises(ValueError, match="nvars"):
        native_runtime.reduce(True, ())
    with pytest.raises(ValueError, match="literal"):
        native_runtime.reduce(1, ((True,),))
    with pytest.raises(ValueError, match="literal"):
        native_runtime.reduce(1, ((2,),))
    with pytest.raises(ValueError, match="iterable"):
        native_runtime.reduce(1, 4)
    with pytest.raises(ValueError, match="iterable"):
        native_runtime.reduce(1, (4,))


def test_build_is_receipt_bound_and_refuses_replacement(native_runtime, tmp_path):
    with pytest.raises(FileExistsError):
        build_native_binary_core(native_runtime.path.parent)

    copied = tmp_path / "copied"
    copied.mkdir()
    library = copied / native_runtime.path.name
    receipt = copied / "build.json"
    shutil.copy2(native_runtime.path, library)
    shutil.copy2(native_runtime.path.with_name("build.json"), receipt)
    record = json.loads(receipt.read_text(encoding="utf-8"))
    record["source_sha256"] = "0" * 64
    receipt.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source hash"):
        NativeBinaryCore(library)


def test_lift_contract_is_strict(native_runtime):
    reduction = native_runtime.reduce(2, ((1, 2),))
    with pytest.raises(ValueError, match="complete tuple"):
        reduction.lift([False] * reduction.reduced_nvars)
    with pytest.raises(ValueError, match="complete tuple"):
        reduction.lift((False,) * (reduction.reduced_nvars + 1))
