"""Select one current compiler-free wheel, then test its actual installation.

Use a clean build directory. No version literal or shell wildcard is required.
Adapted from PR27's wheel selector; it does not require PR27's unmerged runtime.
"""
from __future__ import annotations

import argparse
from email.parser import Parser
import json
import stat
from pathlib import Path, PurePosixPath
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spectra import __version__
from check_indexed_installation import verify

REQUIRED_SOURCES = {
    "spectra/_native/blocked_linear.cpp", "spectra/_native/fp_step.cpp",
    "deploy/m10_dense_extension.cpp", "deploy/replay_ordered_extension.cpp",
    "deploy/cpp_sparse_kernel/extension.cpp", "deploy/cpp_sparse_kernel/spectra_kernel.cpp",
}


def select_wheel(directory: Path, expected_version: str = __version__) -> Path:
    directory = directory.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("--dist must be a directory")
    wheels = sorted(directory.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"expected exactly one wheel; found {len(wheels)}; use a clean build directory")
    wheel = wheels[0]
    if wheel.name != f"spectra-{expected_version}-py3-none-any.whl":
        raise ValueError("expected the current compiler-free SPECTRA wheel")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in entry.filename or stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("unsafe or linked wheel member")
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise ValueError("duplicate wheel members or CRC failure")
        metadata_paths = [n for n in names if n.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise ValueError("wheel must contain exactly one distribution metadata file")
        metadata = Parser().parsestr(archive.read(metadata_paths[0]).decode("utf-8"))
        if metadata.get_all("Name") != ["spectra"] or metadata.get_all("Version") != [expected_version]:
            raise ValueError("wheel name/version metadata does not match the source")
        if not REQUIRED_SOURCES <= set(names):
            raise ValueError("wheel is missing native sources")
    return wheel


SOURCE_PACKAGES = ("spectra", "common", "model", "train", "eval", "data", "deploy")


def verify_sources(wheel: Path, source_root: Path = ROOT) -> dict:
    """Bind every shipped source file to this checkout, not only hot-path files.

    This is a pure-wheel contract. Native binary distributions need a separate
    build/ABI provenance check and are deliberately not accepted here.
    """
    expected = {}
    for package in SOURCE_PACKAGES:
        folder = source_root / package
        if not folder.is_dir():
            raise ValueError(f"missing source package: {package}")
        for path in folder.rglob("*"):
            if path.is_file() and (path.suffix in {".py", ".cpp", ".h", ".hpp"} or path.name == "CMakeLists.txt"):
                if path.is_symlink():
                    raise ValueError("symlinked package source")
                expected[path.relative_to(source_root).as_posix()] = path.read_bytes()
    with zipfile.ZipFile(wheel) as archive:
        names = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
        if len(names) != len(set(names)):
            raise ValueError("duplicate wheel source member")
        shipped = {name for name in names if not name.split("/", 1)[0].endswith(".dist-info")}
        if shipped != set(expected):
            raise ValueError(f"wheel/source inventory differs: missing={sorted(set(expected)-shipped)}, extra={sorted(shipped-set(expected))}")
        for name, content in expected.items():
            if archive.read(name) != content:
                raise ValueError(f"wheel/source bytes differ: {name}")
    return {"matched_source_files": len(expected), "source_identity": "PASS"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    wheel = select_wheel(args.dist)
    sources = verify_sources(wheel)
    result = {"wheel": str(wheel), "version": __version__, **sources, **verify(wheel, args.out)}
    (args.out / "source_report.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
