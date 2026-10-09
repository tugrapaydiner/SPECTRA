"""Trusted, hash-bound binary transport for prevalidated WAP service cases.

The frozen research endpoint intentionally charged canonical JSON decoding, deep
list-to-tuple reconstruction, structural validation, dataclass deep copying,
canonical reserialization, and SHA-256 recomputation.  Those are useful evidence
operations, but they are not a sensible persistent service format.

This module compiles an already fully validated :class:`WorkloadCase` into a
versioned data-only pickle container.  Loading always verifies the complete-file
SHA-256 supplied by an immutable manifest and the payload digest stored in the
header.  A restricted unpickler refuses GLOBAL/persistent-object resolution.
The format is therefore for trusted, hash-bound artifacts; it is not a parser for
arbitrary untrusted pickle bytes.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import io
import json
from pathlib import Path
import pickle
import struct
from typing import Iterable

from experiments.wap_support.workload import SCHEMA, WorkloadCase

MAGIC = b"SPWAPC1\0"
VERSION = 1
# magic, version, flags, n, query_count, payload_bytes, payload_sha256,
# original canonical case_sha256
HEADER = struct.Struct("<8sHHIIQ32s32s")
FLAGS = 0
PACKED_SUFFIX = ".spwap"
MANIFEST_SCHEMA = "spectra.wap_service_format.manifest.v1"


class PackedCaseError(ValueError):
    """The packed artifact, manifest identity, or decoded shape is invalid."""


class _DataOnlyUnpickler(pickle.Unpickler):
    """Reject every opcode path that asks Python to resolve executable objects."""

    def find_class(self, module: str, name: str):  # pragma: no cover - only malicious input
        raise pickle.UnpicklingError(f"global object forbidden: {module}.{name}")

    def persistent_load(self, pid):  # pragma: no cover - only malicious input
        raise pickle.UnpicklingError("persistent objects are forbidden")


@dataclass(frozen=True)
class PackedCaseReceipt:
    schema: str
    source_name: str
    source_file_sha256: str
    case_sha256: str
    packed_name: str
    packed_file_sha256: str
    payload_sha256: str
    packed_bytes: int
    payload_bytes: int
    n: int
    edges: int
    query_count: int
    query_width: int


@dataclass(frozen=True)
class LoadedPackedCase:
    case: WorkloadCase
    file_sha256: str
    payload_sha256: str
    packed_bytes: int
    payload_bytes: int


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _record(case: WorkloadCase) -> tuple:
    return tuple(getattr(case, field.name) for field in fields(WorkloadCase))


def _decode_record(payload: bytes) -> tuple:
    try:
        value = _DataOnlyUnpickler(io.BytesIO(payload)).load()
    except Exception as exc:
        raise PackedCaseError(f"data-only payload decode failed: {type(exc).__name__}: {exc}") from exc
    if type(value) is not tuple or len(value) != len(fields(WorkloadCase)):
        raise PackedCaseError("decoded record has the wrong immutable field shape")
    return value


def _shallow_shape(case: WorkloadCase, *, n: int, query_count: int,
                   case_sha256: str) -> None:
    """Check transport geometry without repeating the expensive canonical audit.

    Full validation happens before packing.  At service load the complete
    and payload hashes authenticate those validated bytes.  These checks still
    prevent a valid pickle with an incompatible record shape from entering the API.
    """
    if case.schema != SCHEMA:
        raise PackedCaseError("unsupported case schema")
    if type(case.n) is not int or case.n != n or case.n < 1:
        raise PackedCaseError("header/record vertex count differs")
    if type(case.query_count) is not int or case.query_count != query_count:
        raise PackedCaseError("header/record query count differs")
    if case.case_sha256 != case_sha256:
        raise PackedCaseError("header/record canonical case identity differs")
    for value, name in ((case.edges, "edges"), (case.dsatur_coloring, "dsatur_coloring"),
                        (case.masks, "masks"), (case.queries, "queries")):
        if type(value) is not tuple:
            raise PackedCaseError(name + " is not immutable")
    if len(case.masks) != n or len(case.dsatur_coloring) != n:
        raise PackedCaseError("record vertex banks differ from n")
    if len(case.queries) != query_count:
        raise PackedCaseError("record query bank differs from query_count")
    if type(case.case_sha256) is not str or len(case.case_sha256) != 64:
        raise PackedCaseError("invalid canonical case identity")


def pack_case(source: str | Path, destination: str | Path) -> PackedCaseReceipt:
    """Validate canonical JSON once and publish an immutable service artifact."""
    source_path = Path(source)
    destination_path = Path(destination)
    if destination_path.exists():
        raise FileExistsError(destination_path)
    case = WorkloadCase.from_json(source_path)  # includes the complete canonical audit
    payload = pickle.dumps(_record(case), protocol=5, fix_imports=False)
    payload_digest = hashlib.sha256(payload).digest()
    header = HEADER.pack(
        MAGIC, VERSION, FLAGS, case.n, case.query_count, len(payload),
        payload_digest, bytes.fromhex(case.case_sha256),
    )
    artifact = header + payload
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_suffix(destination_path.suffix + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(artifact)
        stream.flush()
    temporary.replace(destination_path)
    # Round-trip equality at compile time prevents an encoder/decoder mismatch
    # from becoming a benchmark property.
    loaded = load_packed_case(destination_path, expected_sha256=_sha256(artifact))
    if loaded.case != case:
        destination_path.unlink(missing_ok=True)
        raise AssertionError("packed round trip changed the validated WorkloadCase")
    return PackedCaseReceipt(
        MANIFEST_SCHEMA, source_path.name, _sha256(source_path.read_bytes()),
        case.case_sha256, destination_path.name, loaded.file_sha256,
        loaded.payload_sha256, len(artifact), len(payload), case.n,
        len(case.edges), case.query_count, case.query_width,
    )


def load_packed_case(path: str | Path, *, expected_sha256: str) -> LoadedPackedCase:
    """Verify and load one prevalidated artifact without canonical JSON replay."""
    if type(expected_sha256) is not str or len(expected_sha256) != 64:
        raise PackedCaseError("an exact expected file SHA-256 is required")
    raw = Path(path).read_bytes()
    file_digest = _sha256(raw)
    if file_digest != expected_sha256:
        raise PackedCaseError("packed file SHA-256 differs from manifest")
    if len(raw) < HEADER.size:
        raise PackedCaseError("packed file is shorter than its header")
    magic, version, flags, n, query_count, payload_size, payload_digest, case_digest = HEADER.unpack_from(raw)
    if magic != MAGIC or version != VERSION or flags != FLAGS:
        raise PackedCaseError("unsupported packed format identity")
    if payload_size != len(raw) - HEADER.size:
        raise PackedCaseError("packed payload length differs from header")
    payload = raw[HEADER.size:]
    if hashlib.sha256(payload).digest() != payload_digest:
        raise PackedCaseError("packed payload SHA-256 differs from header")
    case_sha256 = case_digest.hex()
    try:
        case = WorkloadCase(*_decode_record(payload))
    except TypeError as exc:
        raise PackedCaseError("decoded WorkloadCase constructor shape differs") from exc
    _shallow_shape(case, n=n, query_count=query_count, case_sha256=case_sha256)
    return LoadedPackedCase(case, file_digest, payload_digest.hex(), len(raw), len(payload))


def pack_directory(source_directory: str | Path, output_directory: str | Path) -> dict:
    """Compile every canonical ``*.json`` case and seal a deterministic manifest."""
    source_root = Path(source_directory)
    output_root = Path(output_directory)
    output_root.mkdir(parents=True, exist_ok=False)
    receipts = []
    for source in sorted(source_root.glob("*.json")):
        destination = output_root / (source.stem + PACKED_SUFFIX)
        receipts.append(pack_case(source, destination).__dict__)
    if not receipts:
        raise PackedCaseError("no canonical case JSON files found")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "format_magic_hex": MAGIC.hex(),
        "format_version": VERSION,
        "cases": receipts,
    }
    manifest_path = output_root / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


def read_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text())
    if type(manifest) is not dict or manifest.get("schema") != MANIFEST_SCHEMA:
        raise PackedCaseError("unsupported packed-case manifest")
    if manifest.get("format_magic_hex") != MAGIC.hex() or manifest.get("format_version") != VERSION:
        raise PackedCaseError("manifest format identity differs")
    cases = manifest.get("cases")
    if type(cases) is not list or not cases:
        raise PackedCaseError("manifest has no cases")
    names: set[str] = set()
    for receipt in cases:
        if type(receipt) is not dict or set(receipt) != set(PackedCaseReceipt.__dataclass_fields__):
            raise PackedCaseError("manifest receipt fields differ")
        parsed = PackedCaseReceipt(**receipt)
        if parsed.schema != MANIFEST_SCHEMA or parsed.packed_name in names:
            raise PackedCaseError("invalid or duplicate packed receipt")
        if len(parsed.packed_file_sha256) != 64 or len(parsed.case_sha256) != 64:
            raise PackedCaseError("invalid manifest digest")
        names.add(parsed.packed_name)
    return manifest


def receipt_by_name(manifest: dict) -> dict[str, PackedCaseReceipt]:
    return {row["packed_name"]: PackedCaseReceipt(**row) for row in manifest["cases"]}
