"""Strict reproduction of official-style SAT Competition PAR-2 standings.

The input format matches the public 2026 ``scores.csv`` columns. This module does
not run a solver and does not change a benchmark verdict. It validates a complete
rectangular solver-by-instance field, preserves checker outcomes, and recomputes
overall, SAT and UNSAT PAR-2 means from the retained per-cell score.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path
from typing import Iterable

_HEADER = ("solverid", "instanceid", "runtime", "status", "score", "vresult")
_VERIFIED = {"sat-verified": "sat", "unsat-verified": "unsat"}


@dataclass(frozen=True)
class ScoreRow:
    solver_id: str
    instance_id: str
    runtime: Decimal
    status: str
    score: Decimal
    declared_result: str


@dataclass(frozen=True)
class Standing:
    rank: int
    solver_id: str
    disqualified: bool
    overall_par2: Decimal
    sat_par2: Decimal | None
    unsat_par2: Decimal | None
    solved: int
    sat_solved: int
    unsat_solved: int
    instances: int
    sat_instances: int
    unsat_instances: int
    unknown_instances: int
    invalid_cells: int


@dataclass(frozen=True)
class ScoreReceipt:
    schema: str
    source_sha256: str
    source_rows: int
    selected_suffix: str | None
    timeout_seconds: Decimal
    penalty_seconds: Decimal
    solver_count: int
    instance_count: int
    sat_instances: int
    unsat_instances: int
    unknown_instances: int
    standings: tuple[Standing, ...]

    def to_record(self) -> dict:
        def decimal(value: Decimal | None) -> str | None:
            return None if value is None else format(value, "f")

        return {
            "schema": self.schema,
            "source_sha256": self.source_sha256,
            "source_rows": self.source_rows,
            "selected_suffix": self.selected_suffix,
            "timeout_seconds": decimal(self.timeout_seconds),
            "penalty_seconds": decimal(self.penalty_seconds),
            "solver_count": self.solver_count,
            "instance_count": self.instance_count,
            "sat_instances": self.sat_instances,
            "unsat_instances": self.unsat_instances,
            "unknown_instances": self.unknown_instances,
            "standings": [
                {
                    **asdict(item),
                    "overall_par2": decimal(item.overall_par2),
                    "sat_par2": decimal(item.sat_par2),
                    "unsat_par2": decimal(item.unsat_par2),
                }
                for item in self.standings
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_record(), indent=2, sort_keys=True) + "\n"


def _positive_decimal(value: Decimal, label: str) -> Decimal:
    if type(value) is not Decimal or not value.is_finite() or value <= 0:
        raise ValueError(f"{label} must be a positive finite Decimal")
    return value


def _decimal(text: str, label: str, line_number: int) -> Decimal:
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"line {line_number}: invalid {label}") from exc
    if not value.is_finite() or value < 0:
        raise ValueError(f"line {line_number}: {label} must be finite and non-negative")
    return value


def parse_scores(
    data: bytes, *, timeout_seconds: Decimal = Decimal("5000")
) -> tuple[ScoreRow, ...]:
    """Parse and validate a public score table without aggregating it."""
    if type(data) is not bytes:
        raise ValueError("score input must be exact bytes")
    timeout_seconds = _positive_decimal(timeout_seconds, "timeout_seconds")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("score input must be UTF-8") from exc
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = tuple(next(reader))
    except StopIteration as exc:
        raise ValueError("score input is empty") from exc
    if header != _HEADER:
        raise ValueError(f"expected score header {_HEADER!r}")

    penalty = timeout_seconds * 2
    rows: list[ScoreRow] = []
    seen: set[tuple[str, str]] = set()
    for line_number, fields in enumerate(reader, 2):
        if len(fields) != len(_HEADER):
            raise ValueError(f"line {line_number}: expected six score fields")
        solver_id, instance_id, runtime_text, status, score_text, declared = fields
        if not solver_id or not instance_id or not status:
            raise ValueError(f"line {line_number}: empty score identity or status")
        if declared not in ("", "sat", "unsat"):
            raise ValueError(f"line {line_number}: invalid benchmark verdict")
        key = (solver_id, instance_id)
        if key in seen:
            raise ValueError(f"line {line_number}: duplicate solver/instance cell")
        seen.add(key)

        runtime = _decimal(runtime_text, "runtime", line_number)
        score = _decimal(score_text, "score", line_number)
        verified = _VERIFIED.get(status)
        if verified is not None:
            if declared and declared != verified:
                raise ValueError(f"line {line_number}: checker result contradicts verdict")
            if score != runtime:
                raise ValueError(f"line {line_number}: verified score must equal runtime")
        elif score != penalty:
            raise ValueError(f"line {line_number}: unsolved score must equal PAR-2 penalty")

        rows.append(
            ScoreRow(
                solver_id=solver_id,
                instance_id=instance_id,
                runtime=runtime,
                status=status,
                score=score,
                declared_result=declared,
            )
        )
    if not rows:
        raise ValueError("score input has no result rows")
    return tuple(rows)


def _mean(values: Iterable[Decimal]) -> Decimal | None:
    retained = tuple(values)
    return None if not retained else sum(retained, Decimal(0)) / len(retained)


def _is_invalid_status(status: str) -> bool:
    lowered = status.lower()
    return any(token in lowered for token in ("wrong", "invalid", "unsound"))


def score_field(
    rows: tuple[ScoreRow, ...],
    *,
    source_sha256: str,
    source_rows: int,
    timeout_seconds: Decimal = Decimal("5000"),
    solver_suffix: str | None = "[main]",
    solver_ids: Iterable[str] | None = None,
) -> ScoreReceipt:
    """Recompute a complete rectangular solver field and deterministic ranking."""
    timeout_seconds = _positive_decimal(timeout_seconds, "timeout_seconds")
    if type(rows) is not tuple or any(type(row) is not ScoreRow for row in rows):
        raise ValueError("rows must be a tuple of ScoreRow values")
    if (
        type(source_sha256) is not str
        or len(source_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in source_sha256)
    ):
        raise ValueError("source_sha256 must be a lowercase SHA-256")
    if type(source_rows) is not int or source_rows < len(rows):
        raise ValueError("source_rows cannot be smaller than the selected rows")

    requested = None if solver_ids is None else frozenset(solver_ids)
    if requested is not None:
        if not requested or any(type(value) is not str or not value for value in requested):
            raise ValueError("solver_ids must name at least one non-empty solver")
        selected = tuple(row for row in rows if row.solver_id in requested)
        present = {row.solver_id for row in selected}
        if present != set(requested):
            raise ValueError("one or more requested solvers are absent")
        selected_suffix = None
    else:
        if type(solver_suffix) is not str or not solver_suffix:
            raise ValueError("solver_suffix must be a non-empty string")
        selected = tuple(row for row in rows if row.solver_id.endswith(solver_suffix))
        selected_suffix = solver_suffix
    if not selected:
        raise ValueError("no score rows match the selected solver field")

    by_solver: dict[str, dict[str, ScoreRow]] = {}
    verdicts: dict[str, str] = {}
    for row in selected:
        cells = by_solver.setdefault(row.solver_id, {})
        if row.instance_id in cells:
            raise ValueError("duplicate selected solver/instance cell")
        cells[row.instance_id] = row
        candidates = set()
        if row.declared_result:
            candidates.add(row.declared_result)
        if row.status in _VERIFIED:
            candidates.add(_VERIFIED[row.status])
        if len(candidates) > 1:
            raise ValueError("one score cell has conflicting verdict evidence")
        if candidates:
            verdict = next(iter(candidates))
            previous = verdicts.get(row.instance_id)
            if previous is not None and previous != verdict:
                raise ValueError("benchmark verdict differs across solvers")
            verdicts[row.instance_id] = verdict

    instance_sets = {
        solver: frozenset(cells) for solver, cells in by_solver.items()
    }
    reference = next(iter(instance_sets.values()))
    if not reference:
        raise ValueError("selected field contains no benchmark instances")
    if any(instances != reference for instances in instance_sets.values()):
        raise ValueError("selected solver field is not rectangular")

    sat_ids = frozenset(
        instance for instance in reference if verdicts.get(instance) == "sat"
    )
    unsat_ids = frozenset(
        instance for instance in reference if verdicts.get(instance) == "unsat"
    )
    unknown_ids = reference - sat_ids - unsat_ids
    provisional: list[Standing] = []
    for solver_id, cells in sorted(by_solver.items()):
        ordered = tuple(cells[instance] for instance in sorted(reference))
        solved_rows = tuple(row for row in ordered if row.status in _VERIFIED)
        invalid_cells = sum(_is_invalid_status(row.status) for row in ordered)
        provisional.append(
            Standing(
                rank=0,
                solver_id=solver_id,
                disqualified=invalid_cells > 0,
                overall_par2=_mean(row.score for row in ordered) or Decimal(0),
                sat_par2=_mean(cells[instance].score for instance in sat_ids),
                unsat_par2=_mean(cells[instance].score for instance in unsat_ids),
                solved=len(solved_rows),
                sat_solved=sum(row.status == "sat-verified" for row in solved_rows),
                unsat_solved=sum(
                    row.status == "unsat-verified" for row in solved_rows
                ),
                instances=len(reference),
                sat_instances=len(sat_ids),
                unsat_instances=len(unsat_ids),
                unknown_instances=len(unknown_ids),
                invalid_cells=invalid_cells,
            )
        )

    ordered_standings = sorted(
        provisional,
        key=lambda item: (
            item.disqualified,
            item.overall_par2,
            -item.solved,
            item.solver_id,
        ),
    )
    standings = tuple(
        replace(item, rank=rank)
        for rank, item in enumerate(ordered_standings, 1)
    )
    return ScoreReceipt(
        schema="spectra.general_sat_score_receipt.v1",
        source_sha256=source_sha256,
        source_rows=source_rows,
        selected_suffix=selected_suffix,
        timeout_seconds=timeout_seconds,
        penalty_seconds=timeout_seconds * 2,
        solver_count=len(standings),
        instance_count=len(reference),
        sat_instances=len(sat_ids),
        unsat_instances=len(unsat_ids),
        unknown_instances=len(unknown_ids),
        standings=standings,
    )


def score_bytes(
    data: bytes,
    *,
    timeout_seconds: Decimal = Decimal("5000"),
    solver_suffix: str | None = "[main]",
    solver_ids: Iterable[str] | None = None,
) -> ScoreReceipt:
    rows = parse_scores(data, timeout_seconds=timeout_seconds)
    return score_field(
        rows,
        source_sha256=hashlib.sha256(data).hexdigest(),
        source_rows=len(rows),
        timeout_seconds=timeout_seconds,
        solver_suffix=solver_suffix,
        solver_ids=solver_ids,
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("--timeout", type=Decimal, default=Decimal("5000"))
    parser.add_argument("--suffix", default="[main]")
    parser.add_argument("--solver", action="append", dest="solvers")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    data = args.scores.read_bytes()
    receipt = score_bytes(
        data,
        timeout_seconds=args.timeout,
        solver_suffix=args.suffix,
        solver_ids=args.solvers,
    )
    output = receipt.to_json()
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write(output)
    print(output, end="")


if __name__ == "__main__":
    main()
