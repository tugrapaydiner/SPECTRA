from __future__ import annotations

from decimal import Decimal
import hashlib
import json

import pytest

from experiments.general_competition.official_score import (
    ScoreRow,
    main,
    parse_scores,
    score_bytes,
    score_field,
)


HEADER = b"solverid,instanceid,runtime,status,score,vresult\n"


def field_bytes() -> bytes:
    return HEADER + b"".join(
        (
            b"alpha[main],sat-a,1.00,sat-verified,1.00,sat\n",
            b"alpha[main],unsat-a,2.00,unsat-verified,2.00,unsat\n",
            b"alpha[main],sat-b,5000.05,solver-timeout,10000.00,sat\n",
            b"alpha[main],unknown-a,5000.01,solver-timeout,10000.00,\n",
            b"beta[main],sat-a,3.00,sat-verified,3.00,sat\n",
            b"beta[main],unsat-a,5000.03,checker-timeout,10000.00,unsat\n",
            b"beta[main],sat-b,4.00,sat-verified,4.00,sat\n",
            b"beta[main],unknown-a,5000.02,solver-timeout,10000.00,\n",
            b"gamma[main-ai],sat-a,0.50,sat-verified,0.50,sat\n",
            b"gamma[main-ai],unsat-a,0.60,unsat-verified,0.60,unsat\n",
            b"gamma[main-ai],sat-b,0.70,sat-verified,0.70,sat\n",
            b"gamma[main-ai],unknown-a,5000.00,unknown,10000.00,\n",
        )
    )


def test_main_field_reproduces_overall_sat_and_unsat_par2():
    data = field_bytes()
    receipt = score_bytes(data)
    assert receipt.schema == "spectra.general_sat_score_receipt.v1"
    assert receipt.source_sha256 == hashlib.sha256(data).hexdigest()
    assert receipt.source_rows == 12
    assert receipt.selected_suffix == "[main]"
    assert receipt.timeout_seconds == Decimal("5000")
    assert receipt.penalty_seconds == Decimal("10000")
    assert receipt.solver_count == 2
    assert receipt.instance_count == 4
    assert receipt.sat_instances == 2
    assert receipt.unsat_instances == 1
    assert receipt.unknown_instances == 1

    alpha, beta = receipt.standings
    assert (alpha.rank, alpha.solver_id) == (1, "alpha[main]")
    assert alpha.overall_par2 == Decimal("20003") / 4
    assert alpha.sat_par2 == Decimal("10001") / 2
    assert alpha.unsat_par2 == Decimal("2")
    assert (alpha.solved, alpha.sat_solved, alpha.unsat_solved) == (2, 1, 1)
    assert not alpha.disqualified

    assert (beta.rank, beta.solver_id) == (2, "beta[main]")
    assert beta.overall_par2 == Decimal("20007") / 4
    assert beta.sat_par2 == Decimal("7") / 2
    assert beta.unsat_par2 == Decimal("10000")
    assert (beta.solved, beta.sat_solved, beta.unsat_solved) == (2, 2, 0)


def test_explicit_field_can_compare_named_main_and_ai_entries():
    data = field_bytes()
    receipt = score_bytes(
        data,
        solver_ids=("alpha[main]", "gamma[main-ai]"),
    )
    assert receipt.selected_suffix is None
    assert tuple(item.solver_id for item in receipt.standings) == (
        "gamma[main-ai]",
        "alpha[main]",
    )
    assert receipt.standings[0].overall_par2 == Decimal("10001.80") / 4


def test_invalid_answer_is_disqualified_even_with_a_low_mean():
    data = HEADER + b"".join(
        (
            b"bad[main],one,1.00,sat-verified,1.00,sat\n",
            b"bad[main],two,1.00,wrong-answer,10000.00,unsat\n",
            b"sound[main],one,5000.00,solver-timeout,10000.00,sat\n",
            b"sound[main],two,5000.00,solver-timeout,10000.00,unsat\n",
        )
    )
    receipt = score_bytes(data)
    assert receipt.standings[0].solver_id == "sound[main]"
    assert not receipt.standings[0].disqualified
    assert receipt.standings[1].solver_id == "bad[main]"
    assert receipt.standings[1].disqualified
    assert receipt.standings[1].invalid_cells == 1


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"solver,instance\n", "expected score header"),
        (HEADER, "no result rows"),
        (
            HEADER
            + b"a[main],x,1,sat-verified,1,sat\n"
            + b"a[main],x,1,sat-verified,1,sat\n",
            "duplicate",
        ),
        (
            HEADER + b"a[main],x,1,sat-verified,2,sat\n",
            "verified score must equal runtime",
        ),
        (
            HEADER + b"a[main],x,1,sat-verified,1,unsat\n",
            "contradicts verdict",
        ),
        (
            HEADER + b"a[main],x,5000,solver-timeout,9999,sat\n",
            "PAR-2 penalty",
        ),
        (
            HEADER + b"a[main],x,-1,sat-verified,-1,sat\n",
            "non-negative",
        ),
        (
            HEADER + b"a[main],x,NaN,solver-timeout,10000,sat\n",
            "finite",
        ),
        (
            HEADER + b"a[main],x,1,sat-verified,1,maybe\n",
            "benchmark verdict",
        ),
        (
            HEADER + b"a[main],x,1,sat-verified,1\n",
            "six score fields",
        ),
    ],
)
def test_malformed_score_tables_fail_closed(data, message):
    with pytest.raises(ValueError, match=message):
        parse_scores(data)


def test_non_rectangular_and_conflicting_fields_are_rejected():
    non_rectangular = parse_scores(
        HEADER
        + b"a[main],x,1,sat-verified,1,sat\n"
        + b"a[main],y,1,unsat-verified,1,unsat\n"
        + b"b[main],x,1,sat-verified,1,sat\n"
    )
    with pytest.raises(ValueError, match="not rectangular"):
        score_field(
            non_rectangular,
            source_sha256="0" * 64,
            source_rows=len(non_rectangular),
        )

    conflicting = parse_scores(
        HEADER
        + b"a[main],x,5000,solver-timeout,10000,sat\n"
        + b"b[main],x,5000,solver-timeout,10000,unsat\n"
    )
    with pytest.raises(ValueError, match="verdict differs"):
        score_field(
            conflicting,
            source_sha256="1" * 64,
            source_rows=len(conflicting),
        )


def test_selection_and_metadata_validation_fail_closed():
    rows = parse_scores(field_bytes())
    with pytest.raises(ValueError, match="absent"):
        score_field(
            rows,
            source_sha256="2" * 64,
            source_rows=len(rows),
            solver_ids=("missing[main]",),
        )
    with pytest.raises(ValueError, match="solver_suffix"):
        score_field(
            rows,
            source_sha256="2" * 64,
            source_rows=len(rows),
            solver_suffix="",
        )
    with pytest.raises(ValueError, match="source_sha256"):
        score_field(rows, source_sha256="ABC", source_rows=len(rows))
    with pytest.raises(ValueError, match="source_rows"):
        score_field(rows, source_sha256="3" * 64, source_rows=1)
    with pytest.raises(ValueError, match="positive finite"):
        score_field(
            rows,
            source_sha256="3" * 64,
            source_rows=len(rows),
            timeout_seconds=Decimal("0"),
        )
    with pytest.raises(ValueError, match="exact bytes"):
        parse_scores("not bytes")


def test_json_and_cli_are_deterministic_and_refuse_overwrite(tmp_path, capsys):
    data = field_bytes()
    path = tmp_path / "scores.csv"
    path.write_bytes(data)
    output = tmp_path / "receipt.json"

    main((str(path), "--out", str(output)))
    printed = json.loads(capsys.readouterr().out)
    retained = json.loads(output.read_text(encoding="utf-8"))
    assert printed == retained
    assert printed["standings"][0]["solver_id"] == "alpha[main]"
    assert printed["standings"][0]["overall_par2"] == "5000.75"
    with pytest.raises(FileExistsError):
        main((str(path), "--out", str(output)))


def test_score_field_rejects_non_rows():
    with pytest.raises(ValueError, match="ScoreRow"):
        score_field(
            (ScoreRow("a[main]", "x", Decimal(1), "sat-verified", Decimal(1), "sat"), object()),
            source_sha256="4" * 64,
            source_rows=2,
        )


def test_declared_unknown_strata_preserve_means_and_solved_membership():
    data = HEADER + b"".join((
        b"a[main],known,4,sat-verified,4,sat\n",
        b"a[main],unknown-sat,1,sat-verified,1,\n",
        b"a[main],unknown-unsat,2,unsat-verified,2,\n",
        b"b[main],known,4,sat-verified,4,sat\n",
        b"b[main],unknown-sat,5000,solver-timeout,10000,\n",
        b"b[main],unknown-unsat,2,unsat-verified,2,\n",
    ))
    legacy = score_bytes(data)
    assert legacy == score_bytes(data, infer_unknown_verdicts=True)
    assert (legacy.sat_instances, legacy.unsat_instances, legacy.unknown_instances) == (2, 1, 0)
    declared = score_bytes(data, infer_unknown_verdicts=False)
    assert (declared.sat_instances, declared.unsat_instances, declared.unknown_instances) == (1, 0, 2)
    old = {row.solver_id: row for row in legacy.standings}
    for row in declared.standings:
        assert row.overall_par2 == old[row.solver_id].overall_par2
        assert row.solved == old[row.solver_id].solved
        assert row.sat_par2 == Decimal(4) and row.unsat_par2 is None
        assert row.sat_solved == 1 and row.unsat_solved == 0


@pytest.mark.parametrize("infer_unknown_verdicts", [False, True])
def test_unknown_stratum_policy_never_hides_opposite_checked_answers(infer_unknown_verdicts):
    data = HEADER + b"a[main],unknown,1,sat-verified,1,\nb[main],unknown,1,unsat-verified,1,\n"
    with pytest.raises(ValueError, match="verdict differs"):
        score_bytes(data, infer_unknown_verdicts=infer_unknown_verdicts)


@pytest.mark.parametrize("value", [None, 0, 1, "false"])
def test_unknown_stratum_policy_requires_boolean(value):
    with pytest.raises(ValueError, match="infer_unknown_verdicts must be Boolean"):
        score_bytes(field_bytes(), infer_unknown_verdicts=value)
