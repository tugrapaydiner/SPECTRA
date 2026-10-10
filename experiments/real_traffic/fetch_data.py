"""Download the fixed public Abilene inputs with bounded reads and hash receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from experiments.real_traffic import SOURCE_URLS, traffic_url

LIMITS = {
    "A": 128 * 1024,
    "demands": 64 * 1024,
    "links": 64 * 1024,
    "readme.txt": 64 * 1024,
    "traffic": 20 * 1024 * 1024,
}


def download(url: str, *, max_bytes: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SPECTRA-real-traffic-study/1"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > max_bytes:
            raise ValueError(f"source exceeds declared cap: {url}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = response.read(min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"source exceeds declared cap: {url}")
    if not chunks:
        raise ValueError(f"source is empty: {url}")
    return b"".join(chunks)


def fetch_week(week: str, destination: str | Path) -> dict:
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    inventory = {
        "A": SOURCE_URLS["routing"],
        "demands": SOURCE_URLS["demands"],
        "links": SOURCE_URLS["links"],
        "readme.txt": SOURCE_URLS["readme"],
        f"{week}.gz": traffic_url(week),
    }
    records = []
    for name, url in inventory.items():
        limit = LIMITS["traffic"] if name.endswith(".gz") else LIMITS[name]
        payload = download(url, max_bytes=limit)
        path = root / name
        with path.open("xb") as stream:
            stream.write(payload)
        records.append({
            "name": name,
            "url": url,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "max_bytes": limit,
        })
    manifest = {
        "schema": "spectra.real_traffic.sources.v1",
        "week": week,
        "sources": records,
    }
    (root / "sources.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", default="X01")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = fetch_week(args.week, args.out)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
