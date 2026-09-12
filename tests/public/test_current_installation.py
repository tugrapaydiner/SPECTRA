"""Version-independent documented build checks need only the standard library."""
import importlib.util
from pathlib import Path
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
        if source: archive.writestr("spectra/_native/blocked_linear.cpp", "fixture")
        if extra_metadata: archive.writestr("other.dist-info/METADATA", "Name: other\nVersion: 1\n")


def test_selects_current_wheel_without_version_in_command(tmp_path):
    p = tmp_path / "current.whl"; wheel(p)
    assert module.select_wheel(tmp_path) == p


@pytest.mark.parametrize("kind", ["empty", "ambiguous", "stale", "wrong_project", "missing_source", "ambiguous_metadata"])
def test_refuses_ambiguous_or_stale_build(kind, tmp_path):
    if kind != "empty":
        wheel(tmp_path / "one.whl", version="0.0.0" if kind == "stale" else None,
              name="other" if kind == "wrong_project" else "spectra",
              source=kind != "missing_source", extra_metadata=kind == "ambiguous_metadata")
    if kind == "ambiguous": wheel(tmp_path / "two.whl")
    with pytest.raises(ValueError): module.select_wheel(tmp_path)


def test_documented_check_has_no_stale_wheel_name():
    text = (ROOT / "docs/DEVELOPMENT.md").read_text()
    assert "python scripts/check_current_installation.py --dist dist/current --out install-check" in text
    assert "dist/spectra-0.7.0" not in text
