"""Keep post-migration update receipts synchronized with the tracked bytes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re


ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "maintenance" / "relocation-updates.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def test_relocation_update_current_hashes_match_tracked_files():
    payload = json.loads(RECEIPT.read_text())
    assert set(payload) == {"schema", "base", "base_manifest", "updates"}
    assert payload["schema"] == "spectra.relocation_updates.v1"
    assert type(payload["updates"]) is list

    seen = set()
    for item in payload["updates"]:
        assert set(item) == {"path", "previous_sha256", "current_sha256", "reason"}
        path = item["path"]
        parsed = PurePosixPath(path)
        assert type(path) is str and path and not parsed.is_absolute()
        assert ".." not in parsed.parts and parsed.as_posix() == path
        assert path not in seen
        seen.add(path)

        assert HEX64.fullmatch(item["previous_sha256"])
        assert HEX64.fullmatch(item["current_sha256"])
        assert item["previous_sha256"] != item["current_sha256"]
        assert type(item["reason"]) is str and item["reason"].strip()

        target = ROOT / path
        assert target.is_file(), path
        assert hashlib.sha256(target.read_bytes()).hexdigest() == item["current_sha256"], path
