"""Read the exact accepted M16 bytes from either retained container.

The original ZIP hash is pinned. The durable canonical TAR has a different
container identity; its hash inventory must itself match the exact original
inventory, and every listed member is checked. Missing historical input ZIPs in
that TAR are supplied only from independently pinned M14/M15 archives. Neither
route rebuilds, retrains, or substitutes the accepted learned sources.
"""
from __future__ import annotations

import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import zipfile

from data.ancestry import ArtifactAncestry, ExclusionIndex, ManifestSource, digest, require_sha
from eval.verified_search import ValueTarget
from model.typed_value import load_m16_value
from scripts.m16_evidence import CORE_SHA, MANIFEST, Evidence, write_json

M16_ARCHIVE_SHA = "4e002847e476f5f30226395329ca4f28ed9d61467a079c4e58aae4694fff0adc"
M16_INVENTORY_SHA = "5145184804ac7eecfc4eaa142d7584de9d7d54d184a39231429cc3648047611c"
M16_SOURCE_COMMIT = "39dd7b7b8b5491c38862a2c40f566cad1cd45301"
TARGETS = (ValueTarget.IMPROVEMENT, ValueTarget.TERMINAL, ValueTarget.QUALITY)


def safe_name(name: str) -> str:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("invalid archive member name")
    while name.startswith("./"):
        name = name[2:]
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError("unsafe archive member path")
    return str(p)


class AcceptedM16:
    def __init__(self, path: Path, historical: Evidence):
        self.path = Path(path)
        self.container_sha256 = digest(self.path.read_bytes())
        self.members: dict[str, bytes] = {}
        if zipfile.is_zipfile(self.path):
            if self.container_sha256 != M16_ARCHIVE_SHA:
                raise ValueError("not the exact frozen M16 ZIP")
            with zipfile.ZipFile(self.path) as z:
                for member in z.infolist():
                    if member.is_dir():
                        continue
                    self._add(member.filename, z.read(member))
            self.materialization = "exact_accepted_zip"
        else:
            total = 0
            with tarfile.open(self.path, "r:gz") as tar:
                for member in tar:
                    if member.isdir():
                        continue
                    if not member.isfile() or member.size > 256*1024*1024:
                        raise ValueError("unsupported or excessive TAR member")
                    total += member.size
                    if total > 512*1024*1024:
                        raise ValueError("excessive TAR expansion")
                    stream = tar.extractfile(member)
                    if stream is None:
                        raise ValueError("unreadable TAR member")
                    self._add(member.name, stream.read())
            self.materialization = "canonical_tar_exact_inventory_and_member_identity"
        inventory = self.members.get("evidence_sha256.txt", b"")
        if digest(inventory) != M16_INVENTORY_SHA:
            raise ValueError("accepted evidence inventory identity mismatch")
        missing_inputs = {f"inputs/{p.name}": p for p in historical.paths.values()}
        expected = {}
        for line in inventory.decode("utf-8").splitlines():
            checksum, name = line.split(maxsplit=1)
            name = safe_name(name)
            if name in expected:
                raise ValueError("duplicate inventory member")
            expected[name] = require_sha(checksum)
            if name not in self.members and name in missing_inputs:
                self._add(name, missing_inputs[name].read_bytes())
            if name not in self.members or digest(self.members[name]) != checksum:
                raise ValueError(f"accepted member missing or changed: {name}")
        self.verified_inventory = expected
        if self.members["source_commit.txt"].decode().strip() != M16_SOURCE_COMMIT:
            raise ValueError("wrong accepted M16 source commit")
        # Retain all consumed ancestry, not just the new auxiliary training set.
        m14_train = historical.manifest("m14_source", MANIFEST, "training", ("train",))
        self.registry = {s: ArtifactAncestry(s, manifests=(m14_train,)) for s in CORE_SHA.values()}
        self.additional = list(historical.exclusion().sources)
        for seed in (2026091601, 2026091602):
            self.additional.append(self.manifest(f"experiment/manifests/seed{seed}.json",
                                                  role="previous_confirmation" if seed == 2026091602 else "development"))
        training = self.manifest("experiment/manifests/seed2026091601.json", role="training", splits=("train",))
        self.roots = list(CORE_SHA.values())
        for core_seed, core_sha in CORE_SHA.items():
            for target in TARGETS:
                name = f"experiment/targets_development/verifier_{core_seed}_{target.value}.pt"
                sha = self.member_sha(name)
                self.registry[sha] = ArtifactAncestry(sha, parents=(core_sha,), manifests=(training,))
                self.roots.append(sha)

    def _add(self, name: str, data: bytes):
        name = safe_name(name)
        if name in self.members:
            raise ValueError("duplicate normalized archive member")
        self.members[name] = data

    def member_sha(self, name: str) -> str:
        name = safe_name(name)
        if name not in self.verified_inventory:
            raise ValueError("consumed source is not in the pinned inventory")
        return self.verified_inventory[name]

    def manifest(self, name: str, *, role: str, splits=("train", "validation", "test")) -> ManifestSource:
        return ManifestSource.from_bytes(name, self.members[name], role=role,
                                          expected_sha256=self.member_sha(name), splits=splits)

    def exclusion(self) -> ExclusionIndex:
        return ExclusionIndex.close(self.roots, self.registry, additional=self.additional)

    def load_values(self, seed: int, out: Path):
        out.mkdir(parents=True, exist_ok=True)
        models, contracts, records = {}, {}, []
        manifest_sha = self.member_sha("experiment/manifests/seed2026091601.json")
        for target in TARGETS:
            name = f"experiment/targets_development/verifier_{seed}_{target.value}.pt"
            sha = self.member_sha(name)
            path = out/Path(name).name
            if path.exists():
                raise FileExistsError("refusing to replace a source checkpoint")
            path.write_bytes(self.members[name])
            model, contract = load_m16_value(path, expected_sha256=sha,
                expected_core_sha256=CORE_SHA[seed], expected_training_manifest_sha256=manifest_sha,
                diagnostic_improvement=target is ValueTarget.IMPROVEMENT)
            if model.target is not target:
                raise ValueError("source target does not match its frozen slot")
            models[target], contracts[target] = model, contract
            records.append({"path": str(path), "sha256": sha, "contract": contract.metadata(),
                            "materialization": "exact_frozen_M16_checkpoint_no_additional_training"})
        return models, contracts, records

    def identity(self) -> dict:
        return {"materialization": self.materialization, "container_sha256": self.container_sha256,
                "accepted_zip_sha256": M16_ARCHIVE_SHA, "inventory_sha256": M16_INVENTORY_SHA,
                "source_commit": M16_SOURCE_COMMIT, "verified_members": len(self.verified_inventory),
                "ancestry_roots": sorted(self.roots)}
