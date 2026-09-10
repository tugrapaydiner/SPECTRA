"""Fail-closed, content-addressed ancestry for held-out data.

This closes over *declared* checkpoint parents and hash-bound manifests. It cannot
prove that an undeclared dataset never existed, or that symmetry-related tasks
are independent. IDs are metadata, never a substitute for content fingerprints.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Mapping

_SHA = re.compile(r"[0-9a-f]{64}\Z")
ROLES = {"training", "validation", "development", "previous_confirmation"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require_sha(value: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError("expected a lowercase SHA-256 content digest")
    return value


@dataclass(frozen=True)
class ManifestSource:
    name: str
    sha256: str
    role: str
    splits: tuple[str, ...]
    fingerprints: frozenset[str]
    groups: frozenset[str]
    row_count: int

    @classmethod
    def read(cls, path: str | Path, *, role: str, expected_sha256: str,
             splits: Iterable[str] = ("train", "validation", "test")) -> "ManifestSource":
        path = Path(path)
        return cls.from_bytes(path.name, path.read_bytes(), role=role,
                              expected_sha256=expected_sha256, splits=splits)

    @classmethod
    def from_bytes(cls, name: str, data: bytes, *, role: str, expected_sha256: str,
                   splits: Iterable[str] = ("train", "validation", "test")) -> "ManifestSource":
        if not name or role not in ROLES:
            raise ValueError("manifest needs a name and a declared consumed-data role")
        if digest(data) != require_sha(expected_sha256):
            raise ValueError(f"manifest content hash mismatch: {name}")
        names = tuple(splits)
        if not names or len(set(names)) != len(names):
            raise ValueError("consumed split names must be nonempty and unique")
        manifest = json.loads(data)
        available = manifest.get("splits")
        if not isinstance(available, dict):
            raise ValueError("manifest must contain a splits mapping")
        fps: set[str] = set()
        groups: set[str] = set()
        count = 0
        for split in names:
            if split not in available:
                raise ValueError(f"missing declared split {split!r} in {name}")
            entry = available[split]
            rows = entry.get("examples")
            if not isinstance(rows, list) or type(entry.get("count")) is not int or entry["count"] != len(rows):
                raise ValueError(f"invalid manifest row inventory: {name}/{split}")
            for row in rows:
                fps.add(require_sha(row.get("fingerprint")))
                groups.add(require_sha(row.get("group_id")))
            count += len(rows)
        return cls(name, expected_sha256, role, names, frozenset(fps), frozenset(groups), count)

    def metadata(self) -> dict:
        return {"name": self.name, "sha256": self.sha256, "role": self.role,
                "splits": list(self.splits), "row_count": self.row_count,
                "unique_fingerprints": len(self.fingerprints), "unique_groups": len(self.groups)}


@dataclass(frozen=True)
class ArtifactAncestry:
    artifact_sha256: str
    parents: tuple[str, ...] = ()
    manifests: tuple[ManifestSource, ...] = ()
    data_free: bool = False

    def __post_init__(self):
        require_sha(self.artifact_sha256)
        for parent in self.parents:
            require_sha(parent)
        if len(set(self.parents)) != len(self.parents):
            raise ValueError("duplicate ancestry parent")
        if self.data_free and (self.parents or self.manifests):
            raise ValueError("data_free cannot hide parents or consumed manifests")
        if not self.data_free and not self.parents and not self.manifests:
            raise ValueError("learned artifact has missing consumed-data ancestry")
        if self.manifests and not all(isinstance(m, ManifestSource) for m in self.manifests):
            raise TypeError("ancestry manifests must be verified ManifestSource objects")


@dataclass(frozen=True)
class ExclusionIndex:
    fingerprints: frozenset[str]
    groups: frozenset[str]
    artifacts: tuple[str, ...]
    sources: tuple[ManifestSource, ...]

    @classmethod
    def close(cls, roots: Iterable[str], registry: Mapping[str, ArtifactAncestry],
              *, additional: Iterable[ManifestSource] = ()) -> "ExclusionIndex":
        roots = tuple(roots)
        if not roots:
            raise ValueError("explicit artifact ancestry roots are required")
        visited: set[str] = set()
        active: set[str] = set()
        sources: dict[tuple, ManifestSource] = {}

        def add(source: ManifestSource):
            sources[(source.sha256, source.role, source.splits)] = source

        def visit(key: str):
            require_sha(key)
            if key in active:
                raise ValueError("cycle in artifact ancestry")
            if key in visited:
                return
            if key not in registry:
                raise ValueError(f"missing ancestor artifact: {key}")
            record = registry[key]
            if record.artifact_sha256 != key:
                raise ValueError("registry key does not match bound artifact digest")
            active.add(key)
            for parent in record.parents:
                visit(parent)
            for source in record.manifests:
                add(source)
            active.remove(key)
            visited.add(key)

        for root in roots:
            visit(root)
        for source in additional:
            if not isinstance(source, ManifestSource):
                raise TypeError("additional source must be a verified manifest")
            add(source)
        ordered = tuple(sources[k] for k in sorted(sources))
        return cls(frozenset().union(*(m.fingerprints for m in ordered)),
                   frozenset().union(*(m.groups for m in ordered)), tuple(sorted(visited)), ordered)

    @property
    def forbidden(self) -> frozenset[str]:
        # Both pre-augmentation groups and post-augmentation content are banned.
        return self.fingerprints | self.groups

    def audit(self, candidate: ManifestSource) -> dict:
        fp = sorted(candidate.fingerprints & self.forbidden)
        groups = sorted(candidate.groups & self.forbidden)
        return {"schema": "spectra.ancestry_audit.v1", "candidate": candidate.metadata(),
                "ancestor_artifacts": list(self.artifacts),
                "consumed_sources": [s.metadata() for s in self.sources],
                "exact_overlap_count": len(fp), "group_overlap_count": len(groups),
                "overlapping_fingerprints": fp, "overlapping_groups": groups,
                "disjoint": not fp and not groups,
                "scope": "declared content/group ancestry; not symmetry-family independence"}

    def require_disjoint(self, candidate: ManifestSource) -> dict:
        audit = self.audit(candidate)
        if not audit["disjoint"]:
            raise ValueError("held-out data overlap an ancestor's consumed data")
        return audit
