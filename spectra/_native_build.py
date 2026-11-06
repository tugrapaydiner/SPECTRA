"""Shared source snapshots, exclusive publication and native-load identity."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import threading


_LOAD_LOCK = threading.RLock()
_LOADED_IDENTITIES: dict[Path, str] = {}


def snapshot_source(source: Path, directory: Path) -> tuple[Path, str]:
    """Compile the same private bytes that the build receipt identifies."""
    content = source.read_bytes()
    snapshot = directory / source.name
    with snapshot.open('xb') as stream:
        stream.write(content)
    return snapshot, hashlib.sha256(content).hexdigest()


def publish_build(target: Path, library: Path, record: dict) -> None:
    """Publish complete files without replacing any existing output."""
    staged_receipt = target.parent / 'build.json'
    with staged_receipt.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, sort_keys=True, indent=2)
        stream.write('\n')
    receipt = library.with_name('build.json')
    attempted = []
    try:
        for staged, published in ((target, library), (staged_receipt, receipt)):
            attempted.append((staged, published))
            try:
                os.link(staged, published)
            except FileExistsError:
                # This call created nothing, even if the existing path is
                # already a hard link to the same staged inode.
                attempted.pop()
                raise
    except BaseException:
        # A failed/interrupted link may already have published its inode. Remove
        # only links to our staged files, never another writer's replacement.
        for staged, published in reversed(attempted):
            try:
                if os.path.samestat(staged.stat(), published.lstat()):
                    published.unlink()
            except FileNotFoundError:
                pass
        raise


@contextmanager
def native_load_guard(library: Path, expected_sha256: str | None = None):
    """Reject path reuse that could return a loader's cached older binary."""
    path = library.resolve(strict=True)
    with _LOAD_LOCK:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b''):
                digest.update(chunk)
        identity = digest.hexdigest()
        if expected_sha256 is not None and identity != expected_sha256:
            raise ValueError('native library hash differs from its receipt')
        previous = _LOADED_IDENTITIES.get(path)
        if previous is not None and previous != identity:
            raise ValueError(
                'native library path was already loaded with different bytes; '
                'use a new build directory')
        # Even a failed initialization can leave a native handle cached.
        _LOADED_IDENTITIES[path] = identity
        yield path
