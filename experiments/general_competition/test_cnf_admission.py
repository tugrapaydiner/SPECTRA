from __future__ import annotations

import gzip
import io
import json
from pathlib import Path
import sys

import pytest

from experiments.general_competition.cnf_admission import _BoundedReader, admit_cnf, main


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="ascii")
    return path


def test_general_cnf_receipt_is_deterministic_and_preserves_irregular_clauses(
    tmp_path,
):
    path = write(
        tmp_path / "mixed.cnf",
        """c retained identity
p cnf 5 6
1 0
-1 2 0
3 -4 5 0
2 2 -3 0
4 -4 0
0
""",
    )
    first = admit_cnf(path)
    second = admit_cnf(path)
    assert first == second
    assert first.schema == "spectra.general_cnf_admission.v1"
    assert first.problem_class == "GENERAL_CNF"
    assert first.spectra_role == "hybrid_solver_required"
    assert first.declared_variables == 5
    assert first.declared_clauses == 6
    assert first.used_variables == 5
    assert first.literal_occurrences == 11
    assert first.empty_clauses == 1
    assert first.unit_clauses == 1
    assert first.binary_clauses == 2
    assert first.long_clauses == 2
    assert first.maximum_clause_length == 3
    assert first.clauses_with_duplicate_literals == 1
    assert first.tautological_clauses == 1
    assert len(first.source_sha256) == len(first.dimacs_sha256) == 64
    assert len(first.semantic_sha256) == 64


def test_clauses_can_span_lines_and_gzip_has_separate_source_identity(tmp_path):
    plain = write(tmp_path / "binary.cnf", "p cnf 3 2\n1\n-2 0\n3 0\n")
    zipped = tmp_path / "binary.cnf.gz"
    with gzip.GzipFile(filename=str(zipped), mode="wb", mtime=0) as stream:
        stream.write(plain.read_bytes())

    direct = admit_cnf(plain)
    compressed = admit_cnf(zipped)
    assert direct.problem_class == compressed.problem_class == "BINARY_CNF"
    assert direct.spectra_role == compressed.spectra_role == "binary_core_candidate"
    assert direct.semantic_sha256 == compressed.semantic_sha256
    assert direct.dimacs_sha256 == compressed.dimacs_sha256
    assert direct.source_sha256 != compressed.source_sha256
    assert compressed.compression == "gzip"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("1 0\n", "precedes header"),
        ("p cnf 1 1\n2 0\n", "exceeds declared"),
        ("p cnf 1 1\n1\n", "unterminated"),
        ("p cnf 1 2\n1 0\n", "declared 2 clauses"),
        ("p cnf 1 1\n1 0\n0\n", "more clauses"),
        ("p cnf 1 1\n1\nc no\n0\n", "unfinished clause"),
        ("p sat 1 1\n1 0\n", "expected 'p cnf"),
        ("p cnf -1 0\n", "non-negative"),
        ("p cnf 1 1\n1x 0\n", "non-integer"),
    ],
)
def test_malformed_dimacs_is_rejected(tmp_path, text, message):
    path = write(tmp_path / "bad.cnf", text)
    with pytest.raises(ValueError, match=message):
        admit_cnf(path)


def test_byte_and_index_limits_fail_closed(tmp_path):
    path = write(tmp_path / "small.cnf", "p cnf 9 1\n9 0\n")
    with pytest.raises(MemoryError, match="source"):
        admit_cnf(path, max_source_bytes=4)
    with pytest.raises(MemoryError, match="DIMACS"):
        admit_cnf(path, max_dimacs_bytes=4)
    with pytest.raises(MemoryError, match="variable bitmap"):
        admit_cnf(path, max_variable_bitmap_bytes=1)


def test_decompressed_read_is_bounded_before_allocating_the_next_line():
    class GuardedStream(io.BytesIO):
        def readline(self, size=-1):
            # A small synthetic stream exposes the allocation contract without
            # actually expanding a huge gzip member in the test process.
            assert 0 < size <= 17 - self.tell()
            return super().readline(size)

    stream = GuardedStream(b"p cnf 1 1\n" + b"1 " * 1000)
    reader = _BoundedReader(stream, 16)
    assert reader.readline() == b"p cnf 1 1\n"
    with pytest.raises(MemoryError, match="DIMACS"):
        reader.readline()
    assert stream.tell() == 17


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("trailing_newline", [False, True])
def test_exact_decoded_byte_limit_accepts_eof_without_changing_identity(
    tmp_path, compressed, trailing_newline,
):
    content = b"p cnf 1 1\n1 0" + (b"\n" if trailing_newline else b"")
    path = tmp_path / ("limit.cnf.gz" if compressed else "limit.cnf")
    path.write_bytes(gzip.compress(content, mtime=0) if compressed else content)
    original = admit_cnf(path)
    assert admit_cnf(path, max_dimacs_bytes=len(content)) == original
    assert admit_cnf(path, max_dimacs_bytes=sys.maxsize) == original
    assert admit_cnf(path, max_dimacs_bytes=sys.maxsize + 1) == original
    with pytest.raises(MemoryError, match="DIMACS"):
        admit_cnf(path, max_dimacs_bytes=len(content) - 1)


@pytest.mark.parametrize("prefix", [b"c ", b" ", b"p cnf 1 1\n1 "])
def test_gzip_long_physical_line_stops_at_decoded_budget(tmp_path, monkeypatch, prefix):
    from experiments.general_competition import cnf_admission

    path = tmp_path / "long-line.cnf.gz"
    path.write_bytes(gzip.compress(prefix + b" " * (1 << 20), mtime=0))
    opened = []
    original_open = cnf_admission._open_dimacs

    class GuardedGzip:
        def __init__(self, stream):
            self.stream = stream
            self.decoded = 0

        def readline(self, size=-1):
            assert 0 < size <= 65 - self.decoded
            line = self.stream.readline(size)
            self.decoded += len(line)
            return line

        def close(self):
            self.stream.close()

    def guarded_open(source):
        stream, compression = original_open(source)
        guarded = GuardedGzip(stream)
        opened.append(guarded)
        return guarded, compression

    monkeypatch.setattr(cnf_admission, "_open_dimacs", guarded_open)
    with pytest.raises(MemoryError, match="DIMACS"):
        admit_cnf(path, max_dimacs_bytes=64)
    assert opened[0].decoded == 65
    assert opened[0].stream.closed


@pytest.mark.parametrize("compressed", [False, True])
def test_bounded_read_still_rejects_non_ascii_dimacs(tmp_path, compressed):
    content = b"p cnf 1 1\n1 0\nc " + "\u00e9".encode("utf-8")
    path = tmp_path / ("non-ascii.cnf.gz" if compressed else "non-ascii.cnf")
    path.write_bytes(gzip.compress(content, mtime=0) if compressed else content)
    with pytest.raises(ValueError, match="DIMACS must be ASCII"):
        admit_cnf(path, max_dimacs_bytes=len(content))


def test_comments_and_whitespace_do_not_change_semantic_identity(tmp_path):
    first = write(tmp_path / "a.cnf", "p cnf 2 1\n1 -2 0\n")
    second = write(
        tmp_path / "b.cnf",
        "c comment\n\n  p cnf 2 1\n  1   -2   0\n c final comment\n",
    )
    a = admit_cnf(first)
    b = admit_cnf(second)
    assert a.semantic_sha256 == b.semantic_sha256
    assert a.dimacs_sha256 != b.dimacs_sha256


def test_cli_refuses_to_overwrite_receipt(tmp_path, capsys):
    path = write(tmp_path / "small.cnf", "p cnf 1 1\n1 0\n")
    output = tmp_path / "receipt.json"
    main((str(path), "--out", str(output)))
    printed = json.loads(capsys.readouterr().out)
    retained = json.loads(output.read_text())
    assert printed == retained
    with pytest.raises(FileExistsError):
        main((str(path), "--out", str(output)))
