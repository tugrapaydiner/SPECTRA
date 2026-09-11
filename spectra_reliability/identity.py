"""Stable content identities and strict evidence serialization."""
from __future__ import annotations
import hashlib
import json
import math
import struct
from collections.abc import Iterable
from numbers import Integral
from pathlib import Path
from typing import Any


def require_digest(value: str, name: str = "digest") -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex string")
    return value


def positive_int(value: int, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be an integer {'>= 0' if allow_zero else '> 0'}")
    return int(value)


def digest_parts(*parts: bytes | str) -> str:
    h = hashlib.sha256()
    for part in parts:
        raw = part.encode("utf-8") if isinstance(part, str) else part
        h.update(len(raw).to_bytes(8, "little")); h.update(raw)
    return h.hexdigest()


def integral_tuple(values: Iterable[int]) -> tuple[int, ...]:
    out = tuple(values)
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in out):
        raise ValueError("tokens must be integers, not booleans or rounded floats")
    if any(not -(1 << 63) <= int(v) < (1 << 63) for v in out):
        raise ValueError("tokens must fit signed int64")
    return tuple(int(v) for v in out)


def tokens_bytes(values: Iterable[int]) -> bytes:
    values = integral_tuple(values)
    return struct.pack(f"<{len(values)}q", *values)


def pair_fingerprint(task: str, x: Iterable[int], y: Iterable[int], height: int, width: int) -> str:
    """Compatible with data.splits.example_fingerprint, including its target field."""
    height = positive_int(height, "height"); width = positive_int(width, "width")
    if not isinstance(task, str) or not task:
        raise ValueError("task must be nonempty")
    xx, yy = integral_tuple(x), integral_tuple(y)
    if len(xx) != height * width or len(yy) != len(xx):
        raise ValueError("example geometry mismatch")
    return digest_parts(task, f"{height}x{width}", tokens_bytes(xx), tokens_bytes(yy))


def input_fingerprint(task: str, x: Iterable[int], height: int, width: int) -> str:
    height = positive_int(height, "height"); width = positive_int(width, "width")
    xx = integral_tuple(x)
    if len(xx) != height * width or not isinstance(task, str) or not task:
        raise ValueError("invalid task/geometry")
    return digest_parts("spectra.input.v1", task, f"{height}x{width}", tokens_bytes(xx))


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False).encode("utf-8")


def strict_json(data: bytes | str) -> Any:
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError(f"duplicate JSON key: {key}")
            out[key] = value
        return out
    def bad_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")
    obj = json.loads(data, object_pairs_hook=pairs, parse_constant=bad_constant)
    def check(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite JSON number (including exponent overflow)")
        if isinstance(value, dict):
            for child in value.values(): check(child)
        elif isinstance(value, list):
            for child in value: check(child)
    check(obj)
    return obj


def write_json(path: str | Path, value: Any, *, exclusive: bool = False) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as f: f.write(raw)
    else:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(raw, encoding="utf-8"); tmp.replace(path)
