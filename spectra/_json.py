"""Bounded, unambiguous JSON input for the public CLI, not historical replay."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

DEFAULT_MAX_BYTES = 16 * 1024 * 1024


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number: {value}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON number is outside the finite floating-point range")
    return parsed


def load_file(path: Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> Any:
    """Reject duplicate keys, nonfinite numbers, excess bytes and deep nesting.

    The byte cap bounds input, not total decoder memory or CPU time. This is not
    a resource sandbox. Existing Python evidence APIs are intentionally unchanged.
    """
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max-json-bytes must be a positive integer")
    with Path(path).open("rb") as source:
        raw = source.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError(f"JSON input exceeds the {max_bytes}-byte limit")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                          parse_constant=_reject_constant, parse_float=_finite_float)
    except RecursionError as error:
        raise ValueError("JSON nesting exceeds the decoder limit") from error
