#!/usr/bin/env python3
"""Offline integrity, ancestry, semantic-answer and summary audit of retained M17.

This does not train a model, regenerate a study seed, or treat repeated timings
as independent outcomes. Fixed-pool labels are reaggregated, not independently
re-inferred here; stored complete-solve answers ARE checked with the NumPy task
checker. A valid negative experiment exits successfully without claiming success.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import tarfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from data.ancestry import ManifestSource, digest, require_sha
from data.splits import example_fingerprint
from eval.checkable_tasks import MAZE11, SUDOKU_SHIFT
from eval.fixed_pool import summarize_pools
from scripts.m16_evidence import Evidence, write_json
from scripts.m17_cross_task import closed_summary
from scripts.m17_sources import AcceptedM16, safe_name

M17_INVENTORY_SHA = "d64ab237453799ce93d340b0544cc2e07e0232def4f2164b3badbe19e71109c5"
M17_ZIP_SHA = "afce8e97bc97f8600548b1487f6331de1c5e8b4fcd5414518614d4936939d628"


def read_bound_archive(path: Path, *, inventory_sha: str, zip_sha: str) -> dict[str, bytes]:
    """Read a pinned ZIP or a canonical TAR with exactly the pinned member set."""
    members = {}
    total = 0

    def add(name, content):
        nonlocal total
        name = safe_name(name)
        total += len(content)
        if name in members or total > 512*1024*1024:
            raise ValueError("duplicate or excessive archive contents")
        members[name] = content

    if zipfile.is_zipfile(path):
        if digest(path.read_bytes()) != require_sha(zip_sha):
            raise ValueError("retained ZIP identity mismatch")
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                if member.file_size > 256*1024*1024:
                    raise ValueError("excessive ZIP member")
                add(member.filename, archive.read(member))
    else:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                if member.isdir():
                    continue
                if not member.isfile() or member.size > 256*1024*1024:
                    raise ValueError("unsupported or excessive TAR member")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("unreadable TAR member")
                add(member.name, stream.read())
    inventory = members.get("evidence_sha256.txt", b"")
    if digest(inventory) != require_sha(inventory_sha):
        raise ValueError("retained inventory identity mismatch")
    expected = set()
    for line in inventory.decode().splitlines():
        checksum, name = line.split(maxsplit=1)
        name = safe_name(name)
        if name in expected or name not in members or digest(members[name]) != require_sha(checksum):
            raise ValueError("retained member missing, duplicated or changed")
        expected.add(name)
    if set(members) != expected | {"evidence_sha256.txt"}:
        raise ValueError("unexpected unbound archive member")
    return members


def json_rows(raw: bytes) -> list[dict]:
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def verify_m17(members: dict[str, bytes], consumed: frozenset[str]) -> dict:
    """Recompute summaries and validate stored answers against hash-bound inputs."""
    forbidden = set(consumed)
    report = {"families": {}, "manifest_rows_verified": 0, "stored_answers_checked": 0}
    overall = json.loads(members["experiment/summary.json"])
    for family, spec in (("sudoku_shift", SUDOKU_SHIFT), ("maze", MAZE11)):
        prefix = f"experiment/{family}/"
        manifests = {}
        for name in sorted(m for m in members if m.startswith(prefix+"manifests/seed")
                           and m.endswith(".json") and not m.endswith("_audit.json")):
            source = ManifestSource.from_bytes(name, members[name], role="development",
                                               expected_sha256=digest(members[name]))
            if (source.fingerprints | source.groups) & forbidden:
                raise ValueError("M17 data overlap consumed ancestor or prior M17 data")
            forbidden.update(source.fingerprints | source.groups)
            manifest = json.loads(members[name])
            with np.load(io.BytesIO(members[name[:-5]+"_arrays.npz"]), allow_pickle=False) as arrays:
                for split, entry in manifest["splits"].items():
                    x, y = arrays[f"{split}_inputs"], arrays[f"{split}_targets"]
                    if len(x) != entry["count"] or len(y) != entry["count"]:
                        raise ValueError("manifest/array count mismatch")
                    for row, inp, target in zip(entry["examples"], x, y):
                        if example_fingerprint(spec.task, inp, target, spec.height, spec.width) != row["fingerprint"]:
                            raise ValueError("manifest/array fingerprint mismatch")
                    report["manifest_rows_verified"] += len(x)
                manifests[manifest["seed"]] = (manifest, arrays["test_inputs"].copy())
        family_summary = json.loads(members[prefix+"summary.json"])
        if family_summary != overall["families"][family]:
            raise ValueError("top-level/family scientific status mismatch")
        development_seed = 2026091703 if family == "sudoku_shift" else 2026091701
        manifest, inputs = manifests[development_seed]
        input_by_id = {row["id"]: inp for row, inp in zip(manifest["splits"]["test"]["examples"], inputs)}
        rows = json_rows(members[prefix+"development_closed_rows.jsonl"])
        for row in rows:
            if row["example_id"] not in input_by_id:
                raise ValueError("stored answer names a non-development input")
            inp = torch.from_numpy(input_by_id[row["example_id"]].copy()).long().reshape(1, -1)
            raw_answer = row["answer"]
            if raw_answer is not None and (len(raw_answer) != spec.height*spec.width or
                                          any(type(v) is not int for v in raw_answer)):
                raise ValueError("malformed stored answer")
            answer = None if raw_answer is None else torch.tensor([raw_answer], dtype=torch.int64)
            if type(row["valid"]) is not bool or spec.independent_correct(inp, answer) != row["valid"]:
                raise ValueError("stored semantic decision disagrees with independent checker")
            report["stored_answers_checked"] += 1
        recomputed_closed = closed_summary(rows)
        if recomputed_closed != family_summary["development_closed_loop"]:
            raise ValueError("retained closed-loop summary does not reproduce")
        surfaces = {}
        for surface in ("development", "confirmation"):
            name = prefix+surface+"_pools.jsonl"
            if name not in members:
                if surface == "development" or family_summary["confirmation_opened"]:
                    raise ValueError("required fixed pool is missing")
                continue
            pools = json_rows(members[name])
            actual = summarize_pools(pools)
            declared = family_summary[surface+"_fixed_pool"]
            if any(declared.get(k) != v for k, v in actual.items()) or declared["raw_rows_sha256"] != digest(members[name]):
                raise ValueError("retained fixed-pool summary does not reproduce")
            surfaces[surface] = {"pools": actual["model_example_pools"], "coverage": actual["coverage"],
                                 "quality_minus_improvement": actual["quality_minus_improvement"],
                                 "ci95": actual["ci95"], "gate_pass": actual["gate_pass"]}
        opened = "confirmation" in surfaces
        if opened != family_summary["confirmation_opened"] or (opened and not surfaces["development"]["gate_pass"]):
            raise ValueError("confirmation was not protected by the development gate")
        confirmed = opened and surfaces["confirmation"]["gate_pass"]
        if confirmed != family_summary["confirmation_gate_pass"]:
            raise ValueError("confirmation decision mismatch")
        report["families"][family] = {"fixed_pool": surfaces, "confirmation_opened": opened,
            "confirmation_gate_pass": confirmed, "status": family_summary["status"]}
    passed = all(f["confirmation_gate_pass"] for f in report["families"].values())
    status = "TWO_FAMILY_FIXED_POOL_EFFECT_CONFIRMED" if passed else "TWO_FAMILY_CLAIM_NOT_ESTABLISHED"
    if overall["two_family_gate_pass"] != passed or overall["scientific_status"] != status:
        raise ValueError("scientific summary incorrectly promotes or demotes the result")
    report.update(integrity_status="PASS", scientific_status=status, two_family_gate_pass=passed,
        scope="Retained bytes, ancestry, arrays, stored-answer semantics and summary replay; no retraining or new confirmation")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-dir", type=Path, default=Path("results/m16/sources"))
    parser.add_argument("--m16", type=Path, default=Path("results/m16/runs/accepted-34522192590.tar.gz"))
    parser.add_argument("--m17", type=Path, default=Path("results/m17/runs/34534009702-33e548d3e0f52d0664c00929316b2198d5fcffe4.tar.gz"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite an existing verification record")
    historical = Evidence(args.historical_dir)
    accepted = AcceptedM16(args.m16, historical)
    members = read_bound_archive(args.m17, inventory_sha=M17_INVENTORY_SHA, zip_sha=M17_ZIP_SHA)
    report = verify_m17(members, accepted.exclusion().forbidden)
    report.update(m17_container_sha256=digest(args.m17.read_bytes()), inventory_sha256=M17_INVENTORY_SHA,
                  m16_identity=accepted.identity())
    write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
