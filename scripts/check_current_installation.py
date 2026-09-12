"""Select the current wheel unambiguously and run installed-only public checks.

Usage: python scripts/check_current_installation.py --dist dist/current --out install-check
No shell expansion or hard-coded package version is needed. A dirty build directory
is rejected instead of silently choosing a stale or platform-inappropriate wheel.
"""
from __future__ import annotations

import argparse
from email.parser import Parser
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spectra import __version__
from check_indexed_installation import verify


def select_wheel(directory: Path, expected_version: str = __version__) -> Path:
    directory = directory.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("--dist must be a directory")
    wheels = sorted(directory.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"expected exactly one wheel in {directory}; found {len(wheels)}; use a clean output directory")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        if archive.testzip() is not None:
            raise ValueError("wheel CRC check failed")
        metadata_paths = [p for p in archive.namelist() if p.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise ValueError("wheel must contain exactly one distribution metadata file")
        metadata = Parser().parsestr(archive.read(metadata_paths[0]).decode("utf-8"))
        if metadata.get_all("Name") != ["spectra"] or metadata.get_all("Version") != [expected_version]:
            raise ValueError(f"expected spectra {expected_version}; wheel metadata does not match")
        if "spectra/_native/blocked_linear.cpp" not in archive.namelist():
            raise ValueError("wheel is missing the integrated runtime native source")
    return wheel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    wheel = select_wheel(args.dist)
    result = verify(wheel, args.out)
    print(json.dumps({"wheel": str(wheel), "version": __version__, **result}, indent=2))


if __name__ == "__main__":
    main()
