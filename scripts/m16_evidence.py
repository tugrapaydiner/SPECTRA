"""Read pinned source archives, close ancestry, and replay the historical audit."""
from __future__ import annotations

import io
import json
from collections import defaultdict
from pathlib import Path
import zipfile

from data.ancestry import ArtifactAncestry, ExclusionIndex, ManifestSource, digest
from eval.verified_search import pool_metrics

SPECS = {
    "m14_original": ("spectra_m14_evidence.zip", "0306f64efb48c264aa7b387ff6b449def011fb64ad02aa74eb03ee3b4e58e314"),
    "m14_source": ("spectra_m14_source_evidence.zip", "5729932600743b2cef19d9e0b9baf112ec75b627552af4d30a089266994a2f37"),
    "m15": ("spectra_m15_evidence.zip", "c49e5c06e0f2822088cb9ce807fcc6bc37da6236962953ccc8c594be01c55b2e"),
}
CORE_SHA = {1401: "d5d4769726e3e45822e84a947dd4e8cb007a35374a8a40ae61a786c4c2ba55a4",
            2402: "b43f111af13bc8f7b667c9b7e97558b2b9743f522945193a7a625db77ea194ff"}
MANIFEST = "experiment/manifests/train_validation_development.json"


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


class Evidence:
    def __init__(self, directory: Path):
        self.archives = {}
        self.paths = {}
        for key, (name, sha) in SPECS.items():
            path = directory / name
            raw = path.read_bytes()
            if digest(raw) != sha:
                raise ValueError(f"source archive hash mismatch: {name}")
            archive = zipfile.ZipFile(io.BytesIO(raw))
            if len(set(archive.namelist())) != len(archive.namelist()) or archive.testzip() is not None:
                raise ValueError(f"invalid archive: {name}")
            self.archives[key] = archive
            self.paths[key] = path

    def json(self, archive: str, path: str):
        return json.loads(self.archives[archive].read(path))

    def rows(self, archive: str, path: str):
        with self.archives[archive].open(path) as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)

    def manifest(self, archive: str, path: str, role: str, splits=("train", "validation", "test")):
        raw = self.archives[archive].read(path)
        return ManifestSource.from_bytes(f"{archive}/{path}", raw, role=role,
                                         expected_sha256=digest(raw), splits=splits)

    def exclusion(self) -> ExclusionIndex:
        train = self.manifest("m14_source", MANIFEST, "training", ("train",))
        registry = {sha: ArtifactAncestry(sha, manifests=(train,)) for sha in CORE_SHA.values()}
        previous = []
        for archive in self.archives:
            for name in self.archives[archive].namelist():
                if name.startswith("experiment/manifests/") and name.endswith(".json"):
                    item = self.json(archive, name)
                    if "splits" in item:
                        previous.append(self.manifest(archive, name, "development"))
        return ExclusionIndex.close(CORE_SHA.values(), registry, additional=previous)

    def load_model(self, kind: str, seed: int, out: Path):
        import torch
        from scripts.m14_primary_experiment import make_model
        name = f"{kind}_seed{seed}.pt"
        raw = self.archives["m14_source"].read(f"experiment/checkpoints/{name}")
        sha = digest(raw)
        if kind == "fp_recursive_dim64" and sha != CORE_SHA[seed]:
            raise ValueError("accepted core identity mismatch")
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        if payload.get("format") != "spectra.m14_model" or payload.get("version") != 1 or payload["kind"] != kind or payload["seed"] != seed:
            raise ValueError("source checkpoint contract mismatch")
        model = make_model(kind, seed).cpu().eval()
        model.load_state_dict(payload["model_state"], strict=True)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        out.mkdir(parents=True, exist_ok=True)
        (out / name).write_bytes(raw)
        return model, sha

    def retrospective(self) -> dict:
        train = self.manifest("m14_source", MANIFEST, "training", ("train",))
        prior = ExclusionIndex.close(CORE_SHA.values(),
                                    {s: ArtifactAncestry(s, manifests=(train,)) for s in CORE_SHA.values()})
        conf_path = "experiment/manifests/confirmation.json"
        conf = self.manifest("m15", conf_path, "previous_confirmation")
        overlap = prior.audit(conf)
        conf_rows = self.json("m15", conf_path)["splits"]["test"]["examples"]
        excluded_ids = {r["id"] for r in conf_rows if r["fingerprint"] in prior.forbidden or r["group_id"] in prior.forbidden}
        pools = defaultdict(list)
        for row in self.rows("m15", "experiment/confirmation_leaf_rows.jsonl"):
            if row["config_id"] == "learned_strong_d4_int8":
                pools[(row["core_seed"], row["example_id"])].append(row)
        groups = []
        for key, rows in sorted(pools.items()):
            selected = [i for i, r in enumerate(rows) if r["selected"]]
            if not selected:
                raise ValueError("historical pool lacks its selected path")
            # Repeated evaluations of a path can all carry selected=True.
            if len({tuple(rows[i]["path"]) for i in selected}) != 1:
                raise ValueError("historical pool has multiple selected paths")
            groups.append({"pool_id": f"{key[0]}/{key[1]}", "example_id": key[1],
                           "candidate_validity": [bool(r["semantic_valid"]) for r in rows],
                           "selected_index": selected[0]})
        flat = list(self.rows("m15", "experiment/confirmation_rows.jsonl"))
        def totals(exclude):
            result = {}
            baseline = {(r["core_seed"], r["example_id"]): r["semantic_success"] for r in flat if r["config_id"] == "semantic_exit_k4"}
            for row in flat:
                if row["example_id"] in exclude:
                    continue
                entry = result.setdefault(row["config_id"], {"rows": 0, "valid": 0, "regressions": 0})
                entry["rows"] += 1
                entry["valid"] += int(row["semantic_success"])
                entry["regressions"] += int(bool(baseline[(row["core_seed"], row["example_id"])]) and not row["semantic_success"])
            return result
        return {"source_archive_hashes": {k: v[1] for k, v in SPECS.items()},
                "scope": "retrospective replay of recorded labels; not fresh inference or confirmation",
                "ancestry_overlap": overlap, "excluded_example_ids": sorted(excluded_ids),
                "fixed_pool_original": pool_metrics(groups),
                "fixed_pool_sensitivity": pool_metrics([g for g in groups if g["example_id"] not in excluded_ids]),
                "original": totals(set()), "sensitivity": totals(excluded_ids)}
