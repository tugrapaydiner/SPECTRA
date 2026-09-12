import copy
import json

import pytest

from spectra.evidence import FreezeError, capture, verify, write_new


def evidence(tmp_path):
    (tmp_path / "heads").mkdir()
    (tmp_path / "heads" / "value.bin").write_bytes(b"frozen trained head")
    (tmp_path / "selection.json").write_text('{"target":"absolute_quality"}\n')
    return capture(tmp_path, ["heads/value.bin", "selection.json"])


def test_roundtrip_and_explicit_inventory(tmp_path):
    manifest = evidence(tmp_path)
    result = verify(tmp_path, manifest, required=["selection.json", "heads/value.bin"])
    assert result["verified"] and result["artifacts"] == 2


def test_same_length_mutation_fails(tmp_path):
    manifest = evidence(tmp_path)
    path = tmp_path / "heads" / "value.bin"
    path.write_bytes(b"X" * path.stat().st_size)
    with pytest.raises(FreezeError, match="changed"):
        verify(tmp_path, manifest)


def test_missing_artifact_fails(tmp_path):
    manifest = evidence(tmp_path)
    (tmp_path / "heads" / "value.bin").unlink()
    with pytest.raises(FreezeError, match="missing"):
        verify(tmp_path, manifest)


def test_dropped_head_is_not_a_valid_freeze(tmp_path):
    manifest = evidence(tmp_path)
    del manifest["artifacts"]["heads/value.bin"]
    with pytest.raises(FreezeError, match="inventory"):
        verify(tmp_path, manifest, required=["selection.json", "heads/value.bin"])


@pytest.mark.parametrize("name", ["../outside", "/absolute", "heads/../selection.json", "heads//value.bin", "heads\\value.bin", "./selection.json"])
def test_unsafe_paths_fail(tmp_path, name):
    evidence(tmp_path)
    with pytest.raises(FreezeError):
        capture(tmp_path, [name])


def test_symlink_inside_root_is_rejected(tmp_path):
    evidence(tmp_path)
    try:
        (tmp_path / "alias").symlink_to(tmp_path / "selection.json")
    except OSError:
        pytest.skip("symlink creation unavailable on this host")
    with pytest.raises(FreezeError, match="symlink"):
        capture(tmp_path, ["alias"])


@pytest.mark.parametrize("field,value", [("sha256", "z" * 64), ("sha256", "0" * 63), ("bytes", True), ("bytes", -1)])
def test_bad_metadata_fails(tmp_path, field, value):
    manifest = evidence(tmp_path)
    manifest["artifacts"]["selection.json"][field] = value
    with pytest.raises(FreezeError):
        verify(tmp_path, manifest)


def test_empty_and_duplicate_inventories_fail(tmp_path):
    (tmp_path / "a").write_text("a")
    for names in [[], ["a", "a"]]:
        with pytest.raises(FreezeError):
            capture(tmp_path, names)


def test_freeze_file_cannot_be_overwritten(tmp_path):
    manifest = evidence(tmp_path)
    path = tmp_path / "freeze.json"
    write_new(path, manifest)
    assert json.loads(path.read_text()) == manifest
    with pytest.raises(FileExistsError):
        write_new(path, manifest)


def test_manifest_can_move_with_evidence_root(tmp_path):
    manifest = evidence(tmp_path)
    moved = tmp_path.parent / (tmp_path.name + "_moved")
    tmp_path.rename(moved)
    assert verify(moved, manifest)["verified"]
