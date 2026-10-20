from __future__ import annotations

import gzip
import lzma
from pathlib import Path

import pytest

from experiments.general_competition.competition_input import admit_and_load_cnf


TEXT = "c identity\np cnf 4 4\n1 -2 0\n3\n4 0\n-1 -3 -4 0\n0\n"


def test_plain_gzip_and_xz_replay_the_same_admitted_formula(tmp_path):
    plain = tmp_path / "sample.cnf"
    zipped = tmp_path / "sample.cnf.gz"
    xz = tmp_path / "sample.cnf.xz"
    plain.write_text(TEXT, encoding="ascii")
    with gzip.GzipFile(filename=str(zipped), mode="wb", mtime=0) as stream:
        stream.write(TEXT.encode("ascii"))
    with lzma.open(xz, "wb", format=lzma.FORMAT_XZ) as stream:
        stream.write(TEXT.encode("ascii"))

    rows = [admit_and_load_cnf(path) for path in (plain, zipped, xz)]
    assert rows[0].clauses == rows[1].clauses == rows[2].clauses
    assert rows[0].receipt.semantic_sha256 == rows[1].receipt.semantic_sha256
    assert rows[0].receipt.semantic_sha256 == rows[2].receipt.semantic_sha256
    assert rows[0].receipt.dimacs_sha256 == rows[1].receipt.dimacs_sha256
    assert rows[0].receipt.dimacs_sha256 == rows[2].receipt.dimacs_sha256
    assert {row.receipt.compression for row in rows} == {"none", "gzip", "xz"}
    assert len({row.receipt.source_sha256 for row in rows}) == 3


def test_xz_limits_and_malformed_data_fail_closed(tmp_path):
    xz = tmp_path / "sample.cnf.xz"
    with lzma.open(xz, "wb", format=lzma.FORMAT_XZ) as stream:
        stream.write(TEXT.encode("ascii"))
    with pytest.raises(MemoryError, match="source"):
        admit_and_load_cnf(xz, max_source_bytes=4)
    with pytest.raises(MemoryError, match="DIMACS"):
        admit_and_load_cnf(xz, max_dimacs_bytes=4)

    broken = tmp_path / "broken.cnf.xz"
    broken.write_bytes(b"not xz")
    with pytest.raises(lzma.LZMAError):
        admit_and_load_cnf(broken)
