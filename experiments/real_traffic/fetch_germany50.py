"""Acquire and inventory the one-day Germany50 development trace.

The resolved URLs come from the separately retained SNDlib overview inspection.
This utility performs bounded downloads and safe extraction, but does not construct
routes, queries, or solver results.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
from urllib.request import Request, urlopen

NETWORK_URL = "https://sndlib.put.poznan.pl/download/sndlib-networks-native/germany50.txt"
NETWORK_SHA256 = "a295adf283f42c5cce8cd5f9c2fbd8b3071bd02c011c841726c0ec29c744c1fe"
MATRICES_URL = "https://sndlib.put.poznan.pl/download/directed-germany50-DFN-aggregated-5min-over-1day-native.tgz"
README_URL = "https://sndlib.put.poznan.pl/download/germany50-rm.txt"
MAX_DOWNLOAD = {
    "network": 4 * 1024 * 1024,
    "matrices": 16 * 1024 * 1024,
    "readme": 2 * 1024 * 1024,
}
MAX_EXTRACTED = 96 * 1024 * 1024
MAX_MEMBERS = 2000


def fetch(url: str, limit: int) -> bytes:
    request = Request(url, headers={"User-Agent": "SPECTRA-germany50-development/1"})
    with urlopen(request, timeout=120) as response:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > limit:
            raise ValueError(f"declared source size exceeds cap: {url}")
        payload = response.read(limit + 1)
    if not payload or len(payload) > limit:
        raise ValueError(f"empty or oversized source: {url}")
    return payload


def safe_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe archive path: {name}")
    return path


def acquire(destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    network = fetch(NETWORK_URL, MAX_DOWNLOAD["network"])
    if hashlib.sha256(network).hexdigest() != NETWORK_SHA256:
        raise ValueError("Germany50 network identity changed")
    archive = fetch(MATRICES_URL, MAX_DOWNLOAD["matrices"])
    readme = fetch(README_URL, MAX_DOWNLOAD["readme"])
    (destination / "germany50.txt").write_bytes(network)
    (destination / "matrices.tgz").write_bytes(archive)
    (destination / "readme.txt").write_bytes(readme)

    matrix_root = destination / "matrices"
    matrix_root.mkdir()
    members = []
    total = 0
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        archive_members = bundle.getmembers()
        if len(archive_members) > MAX_MEMBERS:
            raise ValueError("too many archive members")
        for member in archive_members:
            path = safe_name(member.name)
            if member.isdir():
                continue
            if not member.isfile() or member.size < 0:
                raise ValueError(f"unsupported archive member: {member.name}")
            total += member.size
            if total > MAX_EXTRACTED:
                raise ValueError("extracted source exceeds cap")
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read member: {member.name}")
            payload = source.read(member.size + 1)
            if len(payload) != member.size:
                raise ValueError(f"archive member size differs: {member.name}")
            target = matrix_root.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(payload)
            members.append({
                "name": str(path),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
    record = {
        "schema": "spectra.germany50.sources.v1",
        "role": "Exposed development data only; no solver or confirmation result.",
        "network": {"url": NETWORK_URL, "bytes": len(network), "sha256": NETWORK_SHA256},
        "matrices_archive": {
            "url": MATRICES_URL,
            "bytes": len(archive),
            "sha256": hashlib.sha256(archive).hexdigest(),
        },
        "readme": {
            "url": README_URL,
            "bytes": len(readme),
            "sha256": hashlib.sha256(readme).hexdigest(),
        },
        "extracted_bytes": total,
        "members": sorted(members, key=lambda item: item["name"]),
    }
    (destination / "sources.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(acquire(args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
