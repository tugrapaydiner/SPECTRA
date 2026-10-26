"""Version-independent documented build checks need only the standard library."""
import importlib.util
import io
from pathlib import Path
import tarfile
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("current_installation", ROOT / "scripts/check_current_installation.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def wheel(path, *, version=None, name="spectra", source=True, extra_metadata=False):
    if version is None: version = module.__version__
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("spectra.dist-info/METADATA", f"Name: {name}\nVersion: {version}\n")
        if source:
            for member in sorted(module.REQUIRED_SOURCES):
                archive.writestr(member, "fixture")
        if extra_metadata: archive.writestr("other.dist-info/METADATA", "Name: other\nVersion: 1\n")


def test_selects_current_wheel_without_version_in_command(tmp_path):
    p = tmp_path / f"spectra-{module.__version__}-py3-none-any.whl"; wheel(p)
    assert module.select_wheel(tmp_path) == p


@pytest.mark.parametrize("kind", ["empty", "ambiguous", "stale", "wrong_project", "missing_source", "ambiguous_metadata"])
def test_refuses_ambiguous_or_stale_build(kind, tmp_path):
    if kind != "empty":
        wheel(tmp_path / f"spectra-{module.__version__}-py3-none-any.whl", version="0.0.0" if kind == "stale" else None,
              name="other" if kind == "wrong_project" else "spectra",
              source=kind != "missing_source", extra_metadata=kind == "ambiguous_metadata")
    if kind == "ambiguous": wheel(tmp_path / "two.whl")
    with pytest.raises(ValueError): module.select_wheel(tmp_path)


@pytest.mark.parametrize("change", [
    None, "research_test", "private_doc", "model", "changed", "missing_source",
    "missing_guide", "duplicate", "symlink", "hardlink", "escape", "backslash",
    "wrong_root", "metadata", "missing_metadata", "oversized_metadata",
])
def test_source_archive_has_only_selected_matching_files(tmp_path, change):
    source = tmp_path / "source"
    files = {f"{name}/__init__.py": b'"""Synthetic package."""\n'
             for name in module.SOURCE_PACKAGES}
    files.update({
        "MANIFEST.in": b"include README.md docs/API.md assets/readme/chart-data.json\n",
        "pyproject.toml": b'[project]\nname = "spectra"\nversion = "1.2.3"\n',
        "setup.py": b"# synthetic build fixture\n",
        "README.md": b"# Fixture\n", "docs/API.md": b"# Public guide\n",
        "assets/readme/chart-data.json": b"{}\n", "examples/tiny.cnf": b"p cnf 0 0\n",
    })
    for name, raw in files.items():
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    (source / "docs/internal.md").write_text("Synthetic private note, not selected.\n")
    expected_count = len(files)
    files["PKG-INFO"] = b"Metadata-Version: 2.4\nName: spectra\nVersion: 1.2.3\n"
    extra = {"research_test": "tests/test_probe.py", "private_doc": "docs/internal.md",
             "model": "spectra/weights.pt", "escape": "../escape", "backslash": "dir\\file"}
    if change in extra:
        files[extra[change]] = b"synthetic fixture"
    elif change == "changed":
        files["spectra/__init__.py"] = b"changed bytes\n"
    elif change == "missing_source":
        files.pop("spectra/__init__.py")
    elif change == "missing_guide":
        files.pop("docs/API.md")
    elif change == "metadata":
        files["PKG-INFO"] = b"Name: other\nVersion: 1.2.3\n"
    elif change == "missing_metadata":
        files.pop("PKG-INFO")
    elif change == "oversized_metadata":
        files["spectra.egg-info/entry_points.txt"] = b"x" * (1024 * 1024 + 1)
    path = tmp_path / "spectra-1.2.3.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        prefix = "other" if change == "wrong_root" else "spectra-1.2.3"
        for name, raw in files.items():
            entry = tarfile.TarInfo(f"{prefix}/{name}")
            entry.size = len(raw)
            archive.addfile(entry, io.BytesIO(raw))
        if change == "duplicate":
            entry = tarfile.TarInfo(f"{prefix}/docs/API.md")
            archive.addfile(entry, io.BytesIO())
        elif change in ("symlink", "hardlink"):
            entry = tarfile.TarInfo(f"{prefix}/link")
            entry.type = tarfile.SYMTYPE if change == "symlink" else tarfile.LNKTYPE
            entry.linkname = "docs/API.md"
            archive.addfile(entry)
    if change:
        with pytest.raises(ValueError):
            module.verify_sdist(path, source, "1.2.3")
    else:
        assert module.verify_sdist(path, source, "1.2.3") == {
            "matched_sdist_files": expected_count, "sdist_identity": "PASS"}
