"""Fail-closed file identity checks for frozen research evidence.

A matching checksum proves byte identity, not data independence or scientific
validity. Manifests use relative paths and can be moved with their evidence root.
This module never opens confirmation data, trains models, or downloads files.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Iterable

FORMAT = "spectra.evidence_freeze.v1"


class FreezeError(ValueError):
    """A frozen artifact is missing, unsafe, malformed, or changed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise FreezeError("artifact paths must be nonempty relative POSIX paths")
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(part in {".", ".."} for part in name.split("/")):
        raise FreezeError("absolute and traversing artifact paths are forbidden")
    if "" in name.split("/"):
        raise FreezeError("empty artifact path segments are forbidden")
    path = root.joinpath(*relative.parts)
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise FreezeError("symlinked artifacts are not accepted")
    try:
        path.resolve(strict=True).relative_to(root)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise FreezeError(f"artifact is missing or escapes its root: {name}") from error
    if not path.is_file():
        raise FreezeError(f"artifact is not a regular file: {name}")
    return path


def capture(root: str | Path, names: Iterable[str]) -> dict:
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise FreezeError("evidence root must be a directory")
    names = list(names)
    if not names or len(names) != len(set(names)):
        raise FreezeError("a freeze needs a nonempty, duplicate-free artifact list")
    artifacts = {}
    for name in sorted(names):
        path = _resolve(root, name)
        artifacts[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return {"format": FORMAT, "artifacts": artifacts}


def verify(root: str | Path, manifest: dict, *, required: Iterable[str] | None = None) -> dict:
    root = Path(root).resolve(strict=True)
    if not isinstance(manifest, dict) or set(manifest) != {"format", "artifacts"}:
        raise FreezeError("unknown or incomplete manifest fields")
    if manifest["format"] != FORMAT:
        raise FreezeError("unsupported evidence-freeze format")
    entries = manifest["artifacts"]
    if not isinstance(entries, dict) or not entries:
        raise FreezeError("empty or malformed artifact inventory")
    if required is not None and set(required) != set(entries):
        raise FreezeError("frozen artifact inventory differs from the required inventory")
    for name, entry in entries.items():
        if not isinstance(entry, dict) or set(entry) != {"sha256", "bytes"}:
            raise FreezeError(f"malformed artifact entry: {name}")
        expected = entry["sha256"]
        size = entry["bytes"]
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise FreezeError(f"invalid SHA-256: {name}")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise FreezeError(f"invalid byte count: {name}")
        path = _resolve(root, name)
        if path.stat().st_size != size or sha256_file(path) != expected:
            raise FreezeError(f"frozen artifact changed: {name}")
    return {"verified": True, "artifacts": len(entries), "format": FORMAT}


def write_new(path: str | Path, manifest: dict) -> None:
    """Create a freeze without silently replacing an earlier frozen decision."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
