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


@pytest.mark.parametrize("destination_is_source", [False, True])
def test_xz_staging_preserves_files_when_exclusive_creation_fails(
    tmp_path, destination_is_source,
):
    from experiments.general_competition.competition_input import _copy_xz

    source = tmp_path / "input.cnf.xz"
    source.write_bytes(lzma.compress(TEXT.encode("ascii")))
    destination = source if destination_is_source else tmp_path / "existing.cnf"
    if destination != source:
        destination.write_bytes(b"previously retained evidence\n")
    original_source = source.read_bytes()
    original_destination = destination.read_bytes()

    with pytest.raises(FileExistsError):
        _copy_xz(source, destination, 1024)

    assert source.read_bytes() == original_source
    assert destination.read_bytes() == original_destination


@pytest.mark.parametrize("failure", ["truncated", "limit", "missing-source"])
def test_xz_staging_removes_its_new_output_on_failure(tmp_path, failure):
    from experiments.general_competition.competition_input import _copy_xz

    source = tmp_path / "input.cnf.xz"
    compressed = lzma.compress(TEXT.encode("ascii"))
    if failure != "missing-source":
        source.write_bytes(compressed[:-8] if failure == "truncated" else compressed)
    destination = tmp_path / "new.cnf"
    error = {"truncated": EOFError, "limit": MemoryError,
             "missing-source": FileNotFoundError}[failure]

    with pytest.raises(error):
        _copy_xz(source, destination, 4 if failure == "limit" else 1024)

    assert not destination.exists()
    if source.exists():
        assert source.read_bytes() == (compressed[:-8] if failure == "truncated" else compressed)


@pytest.mark.parametrize("compression", ["none", "gzip", "xz"])
@pytest.mark.parametrize("change", ["same-size-comment", "shorter-whitespace", "growth"])
def test_replay_rejects_changed_decoded_bytes_even_when_semantics_match(
    tmp_path, monkeypatch, compression, change,
):
    from experiments.general_competition import competition_input

    content = TEXT.encode("ascii")
    suffix = {"none": ".cnf", "gzip": ".cnf.gz", "xz": ".cnf.xz"}[compression]
    source = tmp_path / ("source" + suffix)
    source.write_bytes({"none": lambda data: data,
                        "gzip": lambda data: gzip.compress(data, mtime=0),
                        "xz": lzma.compress}[compression](content))
    changed = {"same-size-comment": content.replace(b"identity", b"IDENTITY"),
               "shorter-whitespace": content[:-1],
               "growth": content + b" " * (1 << 20)}[change]
    open_replay = competition_input._open_for_replay
    opened = []

    def mutate_before_replay(path, replay_compression):
        # XZ replay uses the private decoded file; plain/gzip use the source.
        path.write_bytes(gzip.compress(changed, mtime=0)
                         if replay_compression == "gzip" else changed)
        stream = open_replay(path, replay_compression)
        opened.append(stream)
        return stream

    monkeypatch.setattr(competition_input, "_open_for_replay", mutate_before_replay)
    expected_error = MemoryError if change == "growth" else ValueError
    with pytest.raises(expected_error, match="DIMACS|decoded identity"):
        admit_and_load_cnf(source)
    assert opened[0].closed


@pytest.mark.parametrize("compression", ["gzip", "xz"])
def test_replay_rejects_changed_compressed_source_with_identical_decoded_bytes(
    tmp_path, monkeypatch, compression,
):
    from experiments.general_competition import competition_input

    content = TEXT.encode("ascii")
    source = tmp_path / ("source.cnf." + {"gzip": "gz", "xz": "xz"}[compression])
    if compression == "gzip":
        original = gzip.compress(content, mtime=0)
        changed = gzip.compress(content, mtime=1)
        assert len(original) == len(changed)
        assert gzip.decompress(original) == gzip.decompress(changed)
    else:
        original = lzma.compress(content)
        changed = lzma.compress(content, check=lzma.CHECK_CRC32)
        assert lzma.decompress(original) == lzma.decompress(changed)
    source.write_bytes(original)
    replay = competition_input._replay_clauses

    def mutate_source_before_replay(path, receipt):
        source.write_bytes(changed)
        return replay(path, receipt)

    monkeypatch.setattr(competition_input, "_replay_clauses", mutate_source_before_replay)
    with pytest.raises(ValueError, match="source identity changed"):
        admit_and_load_cnf(source)


@pytest.mark.parametrize("counts", ["1_0 1", "10 0_1"])
@pytest.mark.parametrize("compression", ["none", "gzip", "xz"])
def test_loader_rejects_header_digit_separators(tmp_path, counts, compression):
    content = f"p cnf {counts}\n1 0\n".encode("ascii")
    suffix = {"none": ".cnf", "gzip": ".cnf.gz", "xz": ".cnf.xz"}[compression]
    source = tmp_path / ("counts" + suffix)
    source.write_bytes({"none": lambda data: data,
                        "gzip": lambda data: gzip.compress(data, mtime=0),
                        "xz": lzma.compress}[compression](content))
    with pytest.raises(ValueError, match="header counts"):
        admit_and_load_cnf(source)


@pytest.mark.parametrize("counts", ["1_0 1", "10 0_1"])
def test_replay_checks_header_tokens_even_with_matching_identity(tmp_path, counts):
    from dataclasses import replace
    import hashlib
    from experiments.general_competition import competition_input

    source = tmp_path / "counts.cnf"
    source.write_bytes(b"p cnf 10 1\n1 0\n")
    receipt = admit_and_load_cnf(source).receipt
    content = f"p cnf {counts}\n1 0\n".encode("ascii")
    source.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    # A matching byte/semantic receipt does not make Python-only syntax DIMACS.
    receipt = replace(receipt, source_bytes=len(content), source_sha256=digest,
                      dimacs_bytes=len(content), dimacs_sha256=digest)
    with pytest.raises(ValueError, match="header counts"):
        competition_input._replay_clauses(source, receipt)
