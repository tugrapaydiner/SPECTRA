"""Strict, solver-free admission for arbitrary DIMACS CNF inputs.

This module establishes source identity and structural facts before any solver,
selector, timing run, or benchmark acceptance decision. It deliberately does
not simplify clauses, repair malformed files, solve formulas, or decide whether
an instance is favourable to SPECTRA.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import BinaryIO, Iterable

DEFAULT_MAX_SOURCE_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_MAX_DIMACS_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_MAX_VARIABLE_BITMAP_BYTES = 512 * 1024 * 1024
_READ_CHUNK = 1 << 20


@dataclass(frozen=True)
class CNFAdmission:
    schema: str
    source_name: str
    compression: str
    source_bytes: int
    dimacs_bytes: int
    source_sha256: str
    dimacs_sha256: str
    semantic_sha256: str
    declared_variables: int
    declared_clauses: int
    used_variables: int
    literal_occurrences: int
    empty_clauses: int
    unit_clauses: int
    binary_clauses: int
    long_clauses: int
    maximum_clause_length: int
    clauses_with_duplicate_literals: int
    tautological_clauses: int
    horn_clauses: int
    dual_horn_clauses: int
    problem_class: str
    spectra_role: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


class _BoundedReader:
    """Count and hash bytes while enforcing a decompressed-byte bound."""

    def __init__(self, stream: BinaryIO, limit: int) -> None:
        self._stream = stream
        self._limit = limit
        self.count = 0
        self.hash = hashlib.sha256()

    def readline(self) -> bytes:
        # Bound the read itself, not just the returned length: a gzip member may
        # expand to one enormous physical line. One extra byte distinguishes EOF
        # exactly at the limit from an over-limit stream without requesting the
        # rest. The decoder can still buffer a limited amount of read-ahead.
        line = self._stream.readline(min(self._limit - self.count + 1, sys.maxsize))
        self.count += len(line)
        if self.count > self._limit:
            raise MemoryError("DIMACS input exceeds max_dimacs_bytes")
        self.hash.update(line)
        return line


def _positive_int(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _source_identity(path: Path, max_source_bytes: int) -> tuple[int, str]:
    total = 0
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_READ_CHUNK):
            total += len(chunk)
            if total > max_source_bytes:
                raise MemoryError("source file exceeds max_source_bytes")
            digest.update(chunk)
    return total, digest.hexdigest()


def _open_dimacs(path: Path) -> tuple[BinaryIO, str]:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rb"), "gzip"
    return path.open("rb"), "none"


def _parse_int(token: bytes, *, line_number: int) -> int:
    try:
        text = token.decode("ascii")
        value = int(text, 10)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"line {line_number}: non-integer DIMACS token") from exc
    if any(byte not in b"+-0123456789" for byte in token):
        raise ValueError(f"line {line_number}: malformed integer token")
    return value


def admit_cnf(
    path: str | Path,
    *,
    max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
    max_dimacs_bytes: int = DEFAULT_MAX_DIMACS_BYTES,
    max_variable_bitmap_bytes: int = DEFAULT_MAX_VARIABLE_BITMAP_BYTES,
) -> CNFAdmission:
    """Parse one DIMACS CNF without solving or changing it.

    Clauses may span physical lines. Full-line comments are accepted before or
    after the header. Duplicate literals and tautologies are retained and
    counted rather than silently normalized.
    """
    max_source_bytes = _positive_int(max_source_bytes, "max_source_bytes")
    max_dimacs_bytes = _positive_int(max_dimacs_bytes, "max_dimacs_bytes")
    max_variable_bitmap_bytes = _positive_int(
        max_variable_bitmap_bytes, "max_variable_bitmap_bytes"
    )

    source = Path(path).resolve(strict=True)
    if not source.is_file():
        raise ValueError("CNF source must be a regular file")
    source_bytes, source_sha256 = _source_identity(source, max_source_bytes)

    raw_stream, compression = _open_dimacs(source)
    reader = _BoundedReader(raw_stream, max_dimacs_bytes)
    semantic = hashlib.sha256()

    header_seen = False
    declared_variables = declared_clauses = 0
    clauses_seen = 0
    current: list[int] = []
    used_bitmap = bytearray()
    used_variables = 0

    literal_occurrences = 0
    empty_clauses = unit_clauses = binary_clauses = long_clauses = 0
    maximum_clause_length = 0
    duplicate_clauses = tautological_clauses = 0
    horn_clauses = dual_horn_clauses = 0

    def finish_clause(line_number: int) -> None:
        nonlocal clauses_seen, literal_occurrences
        nonlocal empty_clauses, unit_clauses, binary_clauses, long_clauses
        nonlocal maximum_clause_length, duplicate_clauses, tautological_clauses
        nonlocal horn_clauses, dual_horn_clauses, current

        clauses_seen += 1
        if clauses_seen > declared_clauses:
            raise ValueError(
                f"line {line_number}: more clauses than declared in header"
            )

        length = len(current)
        literal_occurrences += length
        maximum_clause_length = max(maximum_clause_length, length)
        if length == 0:
            empty_clauses += 1
        elif length == 1:
            unit_clauses += 1
        elif length == 2:
            binary_clauses += 1
        else:
            long_clauses += 1

        unique = set(current)
        if len(unique) != length:
            duplicate_clauses += 1
        if any(-literal in unique for literal in unique):
            tautological_clauses += 1
        if sum(literal > 0 for literal in current) <= 1:
            horn_clauses += 1
        if sum(literal < 0 for literal in current) <= 1:
            dual_horn_clauses += 1

        semantic.update(b" ".join(str(value).encode("ascii") for value in current))
        if current:
            semantic.update(b" ")
        semantic.update(b"0\n")
        current = []

    line_number = 0
    try:
        while True:
            raw_line = reader.readline()
            if not raw_line:
                break
            line_number += 1
            try:
                line = raw_line.decode("ascii")
            except UnicodeDecodeError as exc:
                raise ValueError(f"line {line_number}: DIMACS must be ASCII") from exc
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("c"):
                if current:
                    raise ValueError(
                        f"line {line_number}: comment inside an unfinished clause"
                    )
                continue
            if stripped.startswith("p"):
                if header_seen:
                    raise ValueError(f"line {line_number}: duplicate DIMACS header")
                fields = stripped.split()
                if len(fields) != 4 or fields[0] != "p" or fields[1] != "cnf":
                    raise ValueError(f"line {line_number}: expected 'p cnf N M'")
                try:
                    declared_variables = _parse_int(
                        fields[2].encode("ascii"), line_number=line_number
                    )
                    declared_clauses = _parse_int(
                        fields[3].encode("ascii"), line_number=line_number
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"line {line_number}: invalid DIMACS header counts"
                    ) from exc
                if declared_variables < 0 or declared_clauses < 0:
                    raise ValueError(
                        f"line {line_number}: header counts must be non-negative"
                    )
                bitmap_bytes = (declared_variables + 7) // 8
                if bitmap_bytes > max_variable_bitmap_bytes:
                    raise MemoryError(
                        "declared variable bitmap exceeds max_variable_bitmap_bytes"
                    )
                used_bitmap = bytearray(bitmap_bytes)
                header_seen = True
                semantic.update(
                    f"p cnf {declared_variables} {declared_clauses}\n".encode("ascii")
                )
                continue
            if not header_seen:
                raise ValueError(f"line {line_number}: clause data precedes header")

            for token in stripped.encode("ascii").split():
                value = _parse_int(token, line_number=line_number)
                if value == 0:
                    finish_clause(line_number)
                    continue
                variable = abs(value)
                if variable > declared_variables:
                    raise ValueError(
                        f"line {line_number}: literal {value} exceeds declared variables"
                    )
                index = variable - 1
                byte_index = index >> 3
                bit = 1 << (index & 7)
                if not used_bitmap[byte_index] & bit:
                    used_bitmap[byte_index] |= bit
                    used_variables += 1
                current.append(value)
    finally:
        raw_stream.close()

    if not header_seen:
        raise ValueError("missing DIMACS header")
    if current:
        raise ValueError("unterminated final clause")
    if clauses_seen != declared_clauses:
        raise ValueError(
            f"declared {declared_clauses} clauses but parsed {clauses_seen}"
        )

    if declared_clauses == 0:
        problem_class = "EMPTY_CNF"
        spectra_role = "general_frontend_required"
    elif maximum_clause_length <= 1:
        problem_class = "UNIT_CNF"
        spectra_role = "general_frontend_required"
    elif maximum_clause_length <= 2:
        problem_class = "BINARY_CNF"
        spectra_role = "binary_core_candidate"
    else:
        problem_class = "GENERAL_CNF"
        spectra_role = "hybrid_solver_required"

    return CNFAdmission(
        schema="spectra.general_cnf_admission.v1",
        source_name=source.name,
        compression=compression,
        source_bytes=source_bytes,
        dimacs_bytes=reader.count,
        source_sha256=source_sha256,
        dimacs_sha256=reader.hash.hexdigest(),
        semantic_sha256=semantic.hexdigest(),
        declared_variables=declared_variables,
        declared_clauses=declared_clauses,
        used_variables=used_variables,
        literal_occurrences=literal_occurrences,
        empty_clauses=empty_clauses,
        unit_clauses=unit_clauses,
        binary_clauses=binary_clauses,
        long_clauses=long_clauses,
        maximum_clause_length=maximum_clause_length,
        clauses_with_duplicate_literals=duplicate_clauses,
        tautological_clauses=tautological_clauses,
        horn_clauses=horn_clauses,
        dual_horn_clauses=dual_horn_clauses,
        problem_class=problem_class,
        spectra_role=spectra_role,
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cnf", type=Path)
    parser.add_argument(
        "--max-source-bytes", type=int, default=DEFAULT_MAX_SOURCE_BYTES
    )
    parser.add_argument(
        "--max-dimacs-bytes", type=int, default=DEFAULT_MAX_DIMACS_BYTES
    )
    parser.add_argument(
        "--max-variable-bitmap-bytes",
        type=int,
        default=DEFAULT_MAX_VARIABLE_BITMAP_BYTES,
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    receipt = admit_cnf(
        args.cnf,
        max_source_bytes=args.max_source_bytes,
        max_dimacs_bytes=args.max_dimacs_bytes,
        max_variable_bitmap_bytes=args.max_variable_bitmap_bytes,
    )
    text = receipt.to_json()
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
