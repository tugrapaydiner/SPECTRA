"""Resolve and record the public SNDlib sources before using them.

This is a development-source inspection utility. It does not build a workload,
run a solver, or open any future confirmation set.
"""
from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

BASE = "https://sndlib.put.poznan.pl/"
NETWORKS = {
    "germany50": {
        "overview": urljoin(BASE, "germany50.overview.action"),
        "network": urljoin(BASE, "download/sndlib-networks-native/germany50.txt"),
    },
    "geant": {
        "overview": urljoin(BASE, "geant.overview.action"),
        "network": urljoin(BASE, "download/sndlib-networks-native/geant.txt"),
    },
}
MAX_HTML = 2 * 1024 * 1024
MAX_NETWORK = 4 * 1024 * 1024


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.items: list[dict[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            self._href = dict(attrs).get("href")
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href is not None:
            self.items.append({"href": self._href, "text": " ".join("".join(self._text).split())})
            self._href = None
            self._text = []


def fetch(url: str, limit: int) -> bytes:
    request = Request(url, headers={"User-Agent": "SPECTRA-source-inspection/1"})
    with urlopen(request, timeout=90) as response:
        payload = response.read(limit + 1)
        if len(payload) > limit:
            raise ValueError(f"source exceeds cap: {url}")
    if not payload:
        raise ValueError(f"empty source: {url}")
    return payload


def inspect(name: str, out: Path) -> dict:
    if name not in NETWORKS:
        raise ValueError(f"unknown development network: {name}")
    out.mkdir(parents=True, exist_ok=False)
    spec = NETWORKS[name]
    page = fetch(spec["overview"], MAX_HTML)
    network = fetch(spec["network"], MAX_NETWORK)
    parser = Links()
    parser.feed(page.decode("utf-8", errors="strict"))
    links = []
    for item in parser.items:
        absolute = urljoin(spec["overview"], item["href"])
        if absolute.startswith(BASE) and ("download" in absolute or "readme" in absolute.lower()):
            links.append({**item, "url": absolute})
    record = {
        "schema": "spectra.sndlib.source_inspection.v1",
        "network": name,
        "overview": spec["overview"],
        "overview_bytes": len(page),
        "overview_sha256": hashlib.sha256(page).hexdigest(),
        "network_url": spec["network"],
        "network_bytes": len(network),
        "network_sha256": hashlib.sha256(network).hexdigest(),
        "links": links,
        "role": "Development-source inspection only; no solver or confirmation input was run.",
    }
    (out / "overview.html").write_bytes(page)
    (out / f"{name}.txt").write_bytes(network)
    (out / "sources.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", choices=sorted(NETWORKS), default="germany50")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.network, args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
