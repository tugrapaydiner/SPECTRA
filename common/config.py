"""Lightweight YAML configuration loading with inheritance and overrides.

The BLUEPRINT (section 14) mentions hydra/omegaconf, but to keep the core
dependency-light we implement just the two features SPECTRA actually needs:

  * ``_base_`` inheritance: a config may list one or more parent files whose
    values it overrides (deep-merged). This lets ``sudoku.yaml`` extend
    ``base.yaml`` without duplication.
  * dotted-key CLI overrides: ``model.dim=256`` style strings, parsed with YAML
    scalar semantics, so scripts can tweak any field from the command line.

``load_config`` returns a :class:`DotDict` giving both ``cfg["model"]["dim"]``
and ``cfg.model.dim`` access.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


class DotDict(dict):
    """A ``dict`` subclass that also allows attribute access on nested maps.

    Nested dict values are wrapped lazily so ``cfg.model.dim`` works while the
    object still behaves like an ordinary dict (e.g. ``**cfg``, ``json.dumps``).
    """

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:  # surface as AttributeError for natural access
            raise AttributeError(key) from exc
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
            self[key] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Recursively merge ``override`` into a copy of ``base``.

    Dicts are merged key-by-key; any non-dict value in ``override`` replaces the
    corresponding value in ``base`` wholesale.
    """
    merged: dict = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml_with_bases(path: Path, _seen: set[Path] | None = None) -> dict:
    """Load a YAML file, resolving ``_base_`` inheritance relative to it."""
    path = path.resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"Circular _base_ inheritance detected at {path}")
    _seen = _seen | {path}

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw)} in {path}")

    bases = raw.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]

    config: dict = {}
    for base in bases:
        base_path = (path.parent / base).resolve()
        config = _deep_merge(config, _load_yaml_with_bases(base_path, _seen))
    return _deep_merge(config, raw)


def _coerce_scalar(value: str) -> Any:
    """Parse a CLI override value using YAML scalar rules (int/float/bool/str)."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _apply_override(config: dict, dotted_key: str, value: Any) -> None:
    """Set ``config[a][b][c] = value`` for a ``"a.b.c"`` dotted key."""
    keys = dotted_key.split(".")
    node = config
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"Cannot override into non-mapping at '{key}'")
    node[keys[-1]] = value


def load_config(
    path: str | Path,
    overrides: Iterable[str] | None = None,
) -> DotDict:
    """Load a SPECTRA YAML config with inheritance and CLI overrides.

    Args:
        path: Path to the YAML config file.
        overrides: Iterable of ``"dotted.key=value"`` strings (e.g. from argv).

    Returns:
        A :class:`DotDict` with the fully-resolved configuration.
    """
    config = _load_yaml_with_bases(Path(path))

    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override '{override}' must be of the form key=value")
        dotted_key, raw_value = override.split("=", 1)
        _apply_override(config, dotted_key.strip(), _coerce_scalar(raw_value.strip()))

    return DotDict(config)
