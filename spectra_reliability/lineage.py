"""Fail-closed holdout checks over declared, hash-verified model/data ancestry.

Exact separation is not symmetry-family separation. Undeclared private training
sets cannot be ruled out by software. Legacy missing input-only hashes are
reported, never silently claimed complete.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from .identity import canonical_json, require_digest, strict_json


class LineageError(ValueError):
    pass


@dataclass(frozen=True)
class Exposure:
    manifest: str
    manifest_sha256: str
    split: str
    ordinal: int
    row_id: str
    fingerprint: str
    group_id: str
    input_fingerprint: str | None


@dataclass(frozen=True)
class ManifestSource:
    name: str
    sha256: str
    raw: bytes

    def exposures(self) -> tuple[Exposure, ...]:
        if not isinstance(self.name, str) or not self.name:
            raise LineageError("manifest name is required")
        require_digest(self.sha256, "manifest sha256")
        if hashlib.sha256(self.raw).hexdigest() != self.sha256:
            raise LineageError(f"manifest hash mismatch: {self.name}")
        obj = strict_json(self.raw)
        if not isinstance(obj, dict) or not isinstance(obj.get("splits"), dict) or not obj["splits"]:
            raise LineageError(f"missing split inventory: {self.name}")
        result = []; ids = set()
        for split, payload in sorted(obj["splits"].items()):
            if not isinstance(split, str) or not split or not isinstance(payload, dict):
                raise LineageError("malformed split")
            rows = payload.get("examples"); count = payload.get("count")
            if not isinstance(rows, list) or type(count) is not int or count != len(rows):
                raise LineageError(f"split count mismatch: {self.name}:{split}")
            for ordinal, row in enumerate(rows):
                if not isinstance(row, dict): raise LineageError("malformed example")
                row_id = row.get("id")
                if not isinstance(row_id, str) or not row_id or row_id in ids:
                    raise LineageError(f"missing/duplicate row ID: {self.name}:{row_id}")
                ids.add(row_id)
                fp = require_digest(row.get("fingerprint"), "example fingerprint")
                group = require_digest(row.get("group_id"), "group identity")
                inp = row.get("input_fingerprint")
                if inp is not None: require_digest(inp, "input fingerprint")
                result.append(Exposure(self.name, self.sha256, split, ordinal, row_id, fp, group, inp))
        return tuple(result)

    @classmethod
    def from_path(cls, name: str, path: str | Path, expected_sha256: str) -> ManifestSource:
        return cls(name, expected_sha256, Path(path).read_bytes())


@dataclass(frozen=True)
class ArtifactNode:
    name: str
    sha256: str
    parents: tuple[str, ...]
    manifests: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name: raise LineageError("artifact name required")
        require_digest(self.sha256, "artifact identity")
        if len(set(self.parents)) != len(self.parents) or len(set(self.manifests)) != len(self.manifests):
            raise LineageError("duplicate dependency declarations")


class ExposureIndex:
    def __init__(self, sources: Iterable[ManifestSource]):
        self.sources = tuple(sources)
        if not self.sources: raise LineageError("explicit source inventory is required")
        if len({s.name for s in self.sources}) != len(self.sources):
            raise LineageError("manifest names must be unique")
        self.exposures = tuple(e for source in self.sources for e in source.exposures())
        self._pair: dict[str, list[Exposure]] = {}; self._group: dict[str, list[Exposure]] = {}; self._input: dict[str, list[Exposure]] = {}
        for e in self.exposures:
            self._pair.setdefault(e.fingerprint, []).append(e)
            self._group.setdefault(e.group_id, []).append(e)
            if e.input_fingerprint is not None: self._input.setdefault(e.input_fingerprint, []).append(e)

    @property
    def fingerprint_set(self) -> frozenset[str]:
        return frozenset(self._pair)

    @property
    def complete_input_coverage(self) -> bool:
        return bool(self.exposures) and all(e.input_fingerprint is not None for e in self.exposures)

    def matches(self, fingerprint: str, group_id: str, input_hash: str | None = None) -> tuple[Exposure, ...]:
        require_digest(fingerprint); require_digest(group_id)
        if input_hash is not None: require_digest(input_hash)
        values = self._pair.get(fingerprint, []) + self._group.get(group_id, [])
        if input_hash is not None: values = values + self._input.get(input_hash, [])
        return tuple(sorted(set(values), key=lambda e: (e.manifest, e.split, e.ordinal)))

    def audit(self, candidate: ManifestSource) -> dict[str, Any]:
        entries = candidate.exposures(); overlaps = []; duplicates = []
        seen = {key: {} for key in ("pair", "group", "input")}
        for e in entries:
            hits = self.matches(e.fingerprint, e.group_id, e.input_fingerprint)
            if hits:
                overlaps.append({"candidate_id": e.row_id, "candidate_split": e.split, "candidate_ordinal": e.ordinal,
                    "fingerprint": e.fingerprint, "ancestors": [dict(manifest=h.manifest, manifest_sha256=h.manifest_sha256,
                    split=h.split, row_id=h.row_id, ordinal=h.ordinal) for h in hits]})
            for kind, key in (("pair", e.fingerprint), ("group", e.group_id), ("input", e.input_fingerprint)):
                if key is None: continue
                if key in seen[kind]: duplicates.append({"kind": kind, "first_id": seen[kind][key], "second_id": e.row_id})
                else: seen[kind][key] = e.row_id
        return {"schema": "spectra.lineage_audit.v1", "pass": not overlaps and not duplicates,
            "candidate_rows": len(entries), "ancestor_exposure_rows": len(self.exposures),
            "ancestor_unique_pair_fingerprints": len(self._pair), "overlap_count": len(overlaps),
            "overlaps": overlaps, "candidate_duplicates": duplicates,
            "complete_input_only_coverage": self.complete_input_coverage and all(e.input_fingerprint is not None for e in entries),
            "symmetry_disjointness_established": False,
            "source_inventory": [{"name": s.name, "sha256": s.sha256} for s in self.sources]}

    def require_disjoint(self, candidate: ManifestSource, *, require_input_coverage: bool = False) -> dict[str, Any]:
        report = self.audit(candidate)
        if not report["pass"]: raise LineageError(f"holdout overlap: {report['overlap_count']} ancestor matches; {len(report['candidate_duplicates'])} duplicate keys")
        if require_input_coverage and not report["complete_input_only_coverage"]:
            raise LineageError("input-only coverage is incomplete")
        return report


def ancestral_index(nodes: Iterable[ArtifactNode], roots: Iterable[str], sources: Mapping[str, ManifestSource]) -> ExposureIndex:
    nodes = tuple(nodes); roots = tuple(roots); graph = {n.name: n for n in nodes}
    if len(graph) != len(nodes) or not roots: raise LineageError("unique nodes and nonempty roots required")
    visiting = set(); visited = set(); used = set()
    def visit(name):
        if name in visiting: raise LineageError(f"cycle in artifact ancestry: {name}")
        if name in visited: return
        if name not in graph: raise LineageError(f"unresolved ancestor: {name}")
        visiting.add(name); node = graph[name]
        for parent in node.parents: visit(parent)
        for manifest in node.manifests:
            if manifest not in sources: raise LineageError(f"unresolved manifest: {manifest}")
            used.add(manifest)
        visiting.remove(name); visited.add(name)
    for root in roots: visit(root)
    return ExposureIndex(sources[name] for name in sorted(used))


def make_manifest_source(name: str, manifest: dict[str, Any]) -> ManifestSource:
    raw = canonical_json(manifest)
    return ManifestSource(name, hashlib.sha256(raw).hexdigest(), raw)
