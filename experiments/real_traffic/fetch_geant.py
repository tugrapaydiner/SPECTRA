"""Acquire the public GEANT development trace with bounded, hash-recorded I/O.

This downloads the complete public four-month matrix archive but does not construct
queries, run a solver, or reserve any portion as confirmation evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
from urllib.request import Request, urlopen

NETWORK_URL = "https://sndlib.put.poznan.pl/download/sndlib-networks-native/geant.txt"
NETWORK_SHA256 = "ff316c23220ff4485deafcb8701d49c353db0c80072152df3ed9023ed4b4f4ff"
MATRICES_URL = "https://sndlib.put.poznan.pl/download/directed-geant-uhlig-15min-over-4months-ALL-native.tgz"
README_URL = "https://sndlib.put.poznan.pl/download/geant-rm.txt"
MAX_DOWNLOAD = {
    "network": 4 * 1024 * 1024,
    "matrices": 64 * 1024 * 1024,
    "readme": 2 * 1024 * 1024,
}
MAX_EXTRACTED = 768 * 1024 * 1024
MAX_MEMBERS = 20_000
EXPECTED_MATRICES = 11_460


def fetch(url: str, limit: int) -> bytes:
    request = Request(url, headers={"User-Agent": "SPECTRA-geant-development/1"})
    with urlopen(request, timeout=240) as response:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > limit:
            raise ValueError(f"declared source size exceeds cap: {url}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = response.read(min(1024 * 1024, limit + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > limit:
                raise ValueError(f"source exceeds cap: {url}")
    if not chunks:
        raise ValueError(f"empty source: {url}")
    return b"".join(chunks)


def safe_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe archive path: {name}")
    return path


def inspect_archive(payload: bytes) -> dict:
    members = []
    total = 0
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as bundle:
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
            digest = hashlib.sha256()
            observed = 0
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                observed += len(chunk)
            if observed != member.size:
                raise ValueError(f"archive member size differs: {member.name}")
            members.append({
                "name": str(path),
                "bytes": observed,
                "sha256": digest.hexdigest(),
            })
    files = [m for m in members if m["name"].lower().endswith((".txt", ".demand", ".tm"))]
    if len(files) != EXPECTED_MATRICES:
        raise ValueError(f"expected {EXPECTED_MATRICES} matrix files, received {len(files)}")
    return {
        "member_count": len(members),
        "matrix_count": len(files),
        "extracted_bytes": total,
        "members_sha256": hashlib.sha256(
            json.dumps(members, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "first_member": members[0]["name"],
        "last_member": members[-1]["name"],
    }


def acquire(destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    network = fetch(NETWORK_URL, MAX_DOWNLOAD["network"])
    if hashlib.sha256(network).hexdigest() != NETWORK_SHA256:
        raise ValueError("GEANT network identity changed")
    archive = fetch(MATRICES_URL, MAX_DOWNLOAD["matrices"])
    readme = fetch(README_URL, MAX_DOWNLOAD["readme"])
    archive_info = inspect_archive(archive)
    (destination / "geant.txt").write_bytes(network)
    (destination / "matrices.tgz").write_bytes(archive)
    (destination / "readme.txt").write_bytes(readme)
    record = {
        "schema": "spectra.geant.sources.v1",
        "role": "Exposed development data only; no solver or confirmation result.",
        "network": {"url": NETWORK_URL, "bytes": len(network), "sha256": NETWORK_SHA256},
        "matrices_archive": {
            "url": MATRICES_URL,
            "bytes": len(archive),
            "sha256": hashlib.sha256(archive).hexdigest(),
            **archive_info,
        },
        "readme": {
            "url": README_URL,
            "bytes": len(readme),
            "sha256": hashlib.sha256(readme).hexdigest(),
        },
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
