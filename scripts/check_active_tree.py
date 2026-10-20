"""Fail-closed size and hygiene checks for SPECTRA's active branch.

Historical evidence remains reachable from Git history and the dedicated
``archive/research-history-20261016`` ref.  This checker prevents those immutable
archives and one-shot automation files from drifting back into the normal source
checkout or release CI.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import subprocess
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_BYTES = 14 * 1024 * 1024
# Part 3 keeps the current competition adapters, contracts, and compact result
# receipts in the active tree. The historical archive and per-file limits remain
# unchanged; this is not permission to restore retired branches or large evidence.
MAX_TRACKED_FILES = 1000
MAX_SINGLE_FILE_BYTES = 2 * 1024 * 1024
ACTIVE_WORKFLOWS = {
    "cpu-regression.yml",
    "exact-public-api.yml",
    "itc-exam-suite.yml",
    "public-package.yml",
    "repository-audit.yml",
    "svm-portability.yml",
    "wap-support-regression.yml",
}
ALLOWED_ARCHIVES = {"results/sat_workload/pilot.zip"}
FORBIDDEN_PREFIXES = ("delivery/", "maintenance/")
FORBIDDEN_BRANCH_FRAGMENTS = (
    "research/100-blitz-20261016",
    "research/exact-public-api-20261015",
    "research/faster-queries-review-20261015",
    "research/prepared-transfer-20261015",
    "research/compiled-query-validation",
    "faster-queries",
)
HISTORICAL_PNG = re.compile(r"^assets/0[1-6]_[^/]+\.png$")
ARCHIVE_SUFFIXES = (".zip", ".tar.gz", ".tgz")


def _tracked(root: Path) -> list[Path]:
    """Return tracked regular files; never count build/test outputs."""
    try:
        raw = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "-z"], stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return sorted(
            path for path in root.rglob("*")
            if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts
        )
    paths: list[Path] = []
    for name in raw.decode("utf-8").split("\0"):
        if not name:
            continue
        path = root / name
        if path.is_file():
            paths.append(path)
    return sorted(paths)


def _archive(path: str) -> bool:
    return path.endswith(ARCHIVE_SUFFIXES)


def check(root: Path = ROOT) -> dict:
    root = root.resolve(strict=True)
    files = _tracked(root)
    relative = [path.relative_to(root).as_posix() for path in files]
    sizes = {name: (root / name).stat().st_size for name in relative}
    total = sum(sizes.values())

    if len(relative) > MAX_TRACKED_FILES:
        raise ValueError(f"active tree has {len(relative)} files; limit is {MAX_TRACKED_FILES}")
    if total > MAX_TRACKED_BYTES:
        raise ValueError(f"active tree has {total} bytes; limit is {MAX_TRACKED_BYTES}")

    oversized = sorted((name, size) for name, size in sizes.items()
                       if size > MAX_SINGLE_FILE_BYTES)
    if oversized:
        raise ValueError(f"oversized active files: {oversized}")

    forbidden = sorted(name for name in relative if name.startswith(FORBIDDEN_PREFIXES))
    if forbidden:
        raise ValueError(f"retired transport/maintenance files remain active: {forbidden}")

    archives = sorted(name for name in relative if _archive(name) and name not in ALLOWED_ARCHIVES)
    if archives:
        raise ValueError(f"historical archives must not be tracked on main: {archives}")

    pngs = sorted(name for name in relative if HISTORICAL_PNG.fullmatch(name))
    if pngs:
        raise ValueError(f"retired generated PNGs remain active: {pngs}")

    workflow_dir = root / ".github" / "workflows"
    workflows = {path.name for path in workflow_dir.glob("*.y*ml") if path.is_file()}
    if workflows != ACTIVE_WORKFLOWS:
        raise ValueError(
            f"active workflows differ: missing={sorted(ACTIVE_WORKFLOWS-workflows)}, "
            f"unexpected={sorted(workflows-ACTIVE_WORKFLOWS)}"
        )
    stale: list[str] = []
    for name in sorted(workflows):
        text = (workflow_dir / name).read_text(encoding="utf-8")
        for fragment in FORBIDDEN_BRANCH_FRAGMENTS:
            if fragment in text:
                stale.append(f"{name}:{fragment}")
    if stale:
        raise ValueError(f"workflows reference retired branches: {stale}")

    readme = (root / "README.md").read_text(encoding="utf-8")
    if "archive/research-history-20261016" not in readme:
        raise ValueError("README must identify the dedicated historical archive branch")
    results_readme = root / "results" / "README.md"
    if not results_readme.is_file():
        raise ValueError("results/README.md must explain the active-results boundary")

    by_top: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    for name, size in sizes.items():
        top = name.split("/", 1)[0]
        by_top[top]["files"] += 1
        by_top[top]["bytes"] += size

    largest = [
        {"path": name, "bytes": size}
        for name, size in sorted(sizes.items(), key=lambda item: (-item[1], item[0]))[:25]
    ]
    return {
        "status": "PASS",
        "tracked_files": len(relative),
        "tracked_bytes": total,
        "limits": {
            "tracked_files": MAX_TRACKED_FILES,
            "tracked_bytes": MAX_TRACKED_BYTES,
            "single_file_bytes": MAX_SINGLE_FILE_BYTES,
        },
        "active_workflows": sorted(workflows),
        "allowed_archives": sorted(ALLOWED_ARCHIVES),
        "top_level": dict(sorted(by_top.items(), key=lambda item: (-item[1]["bytes"], item[0]))),
        "largest_files": largest,
        "history_location": "archive/research-history-20261016 and immutable Git commits",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = check(args.root)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
