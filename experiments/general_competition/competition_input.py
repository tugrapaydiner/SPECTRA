"""Admit and load official SAT Competition CNF files without bypassing receipts.

GBD distributes competition formulas as XZ-compressed DIMACS. The original
admission command predates that distribution path and accepts plain or gzip input.
This module preserves the compressed source identity, decompresses into a bounded
temporary file, invokes the existing strict admission parser, and independently
replays the admitted DIMACS into an immutable clause bank.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import gzip
import hashlib
import lzma
from pathlib import Path
import tempfile
from typing import BinaryIO

from experiments.general_competition.binary_core import Clause
from experiments.general_competition.cnf_admission import (
    _BoundedReader,
    CNFAdmission,
    DEFAULT_MAX_DIMACS_BYTES,
    DEFAULT_MAX_SOURCE_BYTES,
    DEFAULT_MAX_VARIABLE_BITMAP_BYTES,
    admit_cnf,
)

_READ_CHUNK = 1 << 20


@dataclass(frozen=True)
class AdmittedCNF:
    receipt: CNFAdmission
    clauses: tuple[Clause, ...]

    @property
    def nvars(self) -> int:
        return self.receipt.declared_variables


def _positive(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive exact integer")
    return value


def _source_identity(path: Path, limit: int) -> tuple[int, str]:
    total = 0
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_READ_CHUNK):
            total += len(chunk)
            if total > limit:
                raise MemoryError("source file exceeds max_source_bytes")
            digest.update(chunk)
    return total, digest.hexdigest()


def _copy_xz(source: Path, destination: Path, limit: int) -> None:
    written = 0
    # Cleanup owns only a file this call created. In particular, an exclusive
    # open failure must not unlink an existing destination (or the source).
    output = destination.open("xb")
    try:
        with output, lzma.open(source, "rb") as input_stream:
            while chunk := input_stream.read(_READ_CHUNK):
                written += len(chunk)
                if written > limit:
                    raise MemoryError("DIMACS input exceeds max_dimacs_bytes")
                output.write(chunk)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _open_for_replay(path: Path, compression: str) -> BinaryIO:
    if compression == "gzip":
        return gzip.open(path, "rb")
    if compression == "xz":
        return lzma.open(path, "rb")
    if compression == "none":
        return path.open("rb")
    raise ValueError("unsupported admitted compression")


def _replay_clauses(path: Path, receipt: CNFAdmission) -> tuple[Clause, ...]:
    clauses: list[Clause] = []
    current: list[int] = []
    header_seen = False
    declared_variables = declared_clauses = 0
    semantic = hashlib.sha256()
    line_number = 0

    stream = _open_for_replay(path, receipt.compression)
    reader = _BoundedReader(stream, receipt.dimacs_bytes)
    try:
        while raw_line := reader.readline():
            line_number += 1
            try:
                stripped = raw_line.decode("ascii").strip()
            except UnicodeDecodeError as error:
                raise ValueError(f"line {line_number}: DIMACS must be ASCII") from error
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
                if len(fields) != 4 or fields[:2] != ["p", "cnf"]:
                    raise ValueError(f"line {line_number}: expected 'p cnf N M'")
                try:
                    if any(character not in "+-0123456789"
                           for token in fields[2:] for character in token):
                        raise ValueError("malformed integer token")
                    declared_variables = int(fields[2], 10)
                    declared_clauses = int(fields[3], 10)
                except ValueError as error:
                    raise ValueError(
                        f"line {line_number}: invalid DIMACS header counts"
                    ) from error
                if declared_variables < 0 or declared_clauses < 0:
                    raise ValueError(
                        f"line {line_number}: header counts must be non-negative"
                    )
                semantic.update(
                    f"p cnf {declared_variables} {declared_clauses}\n".encode("ascii")
                )
                header_seen = True
                continue
            if not header_seen:
                raise ValueError(f"line {line_number}: clause data precedes header")

            for token in stripped.split():
                # Valid tokens take the C-level digit scan. Preserve the original
                # error classification for malformed sign sequences on the cold path.
                if not token.lstrip("+-").isdigit() and any(
                    character not in "+-0123456789" for character in token
                ):
                    raise ValueError(f"line {line_number}: malformed integer token")
                try:
                    literal = int(token, 10)
                except ValueError as error:
                    raise ValueError(
                        f"line {line_number}: non-integer DIMACS token"
                    ) from error
                if literal:
                    if abs(literal) > declared_variables:
                        raise ValueError(
                            f"line {line_number}: literal exceeds declared variables"
                        )
                    current.append(literal)
                    continue
                row = tuple(current)
                clauses.append(row)
                semantic.update(b" ".join(str(value).encode("ascii") for value in row))
                if row:
                    semantic.update(b" ")
                semantic.update(b"0\n")
                current = []
    finally:
        stream.close()

    if reader.count != receipt.dimacs_bytes or reader.hash.hexdigest() != receipt.dimacs_sha256:
        raise ValueError("replay decoded identity differs from admission receipt")
    if not header_seen:
        raise ValueError("missing DIMACS header")
    if current:
        raise ValueError("unterminated final clause")
    if declared_variables != receipt.declared_variables:
        raise ValueError("replay variable count differs from admission receipt")
    if declared_clauses != receipt.declared_clauses or len(clauses) != declared_clauses:
        raise ValueError("replay clause count differs from admission receipt")
    if semantic.hexdigest() != receipt.semantic_sha256:
        raise ValueError("replay semantic identity differs from admission receipt")
    return tuple(clauses)


def admit_and_load_cnf(
    path: str | Path,
    *,
    max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
    max_dimacs_bytes: int = DEFAULT_MAX_DIMACS_BYTES,
    max_variable_bitmap_bytes: int = DEFAULT_MAX_VARIABLE_BITMAP_BYTES,
) -> AdmittedCNF:
    """Return clauses bound to the decoded bytes and rechecked source receipt.

    The final source recheck detects persistent changes during admission/replay;
    it is not a filesystem lock or an atomic snapshot against hostile writers.
    """
    max_source_bytes = _positive(max_source_bytes, "max_source_bytes")
    max_dimacs_bytes = _positive(max_dimacs_bytes, "max_dimacs_bytes")
    max_variable_bitmap_bytes = _positive(
        max_variable_bitmap_bytes, "max_variable_bitmap_bytes"
    )
    source = Path(path).resolve(strict=True)
    if not source.is_file():
        raise ValueError("CNF source must be a regular file")

    if source.suffix.lower() != ".xz":
        receipt = admit_cnf(
            source,
            max_source_bytes=max_source_bytes,
            max_dimacs_bytes=max_dimacs_bytes,
            max_variable_bitmap_bytes=max_variable_bitmap_bytes,
        )
        clauses = _replay_clauses(source, receipt)
        if _source_identity(source, max_source_bytes) != (receipt.source_bytes, receipt.source_sha256):
            raise ValueError("source identity changed during admission or replay")
        return AdmittedCNF(receipt, clauses)

    source_bytes, source_sha256 = _source_identity(source, max_source_bytes)
    with tempfile.TemporaryDirectory(prefix="spectra-general-cnf-") as temporary:
        plain = Path(temporary) / "admitted.cnf"
        _copy_xz(source, plain, max_dimacs_bytes)
        plain_receipt = admit_cnf(
            plain,
            max_source_bytes=max_dimacs_bytes,
            max_dimacs_bytes=max_dimacs_bytes,
            max_variable_bitmap_bytes=max_variable_bitmap_bytes,
        )
        receipt = replace(
            plain_receipt,
            source_name=source.name,
            compression="xz",
            source_bytes=source_bytes,
            source_sha256=source_sha256,
        )
        replay_receipt = replace(receipt, compression="none")
        clauses = _replay_clauses(plain, replay_receipt)
    if _source_identity(source, max_source_bytes) != (receipt.source_bytes, receipt.source_sha256):
        raise ValueError("source identity changed during admission or replay")
    return AdmittedCNF(receipt, clauses)
