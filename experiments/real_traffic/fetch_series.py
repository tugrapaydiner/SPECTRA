"""Download several fixed Abilene weeks and retain exact source receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.real_traffic.fetch_data import LIMITS, download
from experiments.real_traffic.sources import SOURCE_URLS, traffic_url


def fetch_series(weeks, destination):
    weeks = tuple(weeks)
    if not weeks or len(set(weeks)) != len(weeks):
        raise ValueError("weeks must be nonempty and distinct")
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    inventory = {
        "A": SOURCE_URLS["routing"],
        "demands": SOURCE_URLS["demands"],
        "links": SOURCE_URLS["links"],
        "readme.txt": SOURCE_URLS["readme"],
    }
    inventory.update({week + ".gz": traffic_url(week) for week in weeks})
    records = []
    for name, url in inventory.items():
        limit = LIMITS["traffic"] if name.endswith(".gz") else LIMITS[name]
        payload = download(url, max_bytes=limit)
        with (root / name).open("xb") as stream:
            stream.write(payload)
        records.append({
            "name": name, "url": url, "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "max_bytes": limit,
        })
    manifest = {
        "schema": "spectra.real_traffic.series_sources.v1",
        "weeks": list(weeks),
        "sources": records,
    }
    (root / "sources.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weeks", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(fetch_series(args.weeks, args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
