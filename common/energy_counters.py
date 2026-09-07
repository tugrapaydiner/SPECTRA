"""Validated Linux powercap/RAPL energy counters used by all SPECTRA telemetry.

M13 measurement contract:
- never fabricate a missing/invalid reading as zero;
- retain explicit domain identity and per-counter range;
- handle one conservative range wrap;
- reject resets/corruption/partial counter sets;
- aggregate independent package domains only (never nested children).

The historical public name ``rapl_available`` is retained by ``eval.edge_energy``
for compatibility, but this module scans generic Linux powercap paths and does not
assume that a package counter is whole-system or GPU energy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable

DEFAULT_POWERCAP_ROOT = Path("/sys/class/powercap")
WRAP_GUARD_FRACTION = 0.25


@dataclass(frozen=True)
class EnergyDomain:
    domain_id: str
    name: str
    path: Path
    parent_domain_id: str | None
    domain_class: str
    max_energy_range_uj: int
    include_in_package_total: bool

    def record(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "name": self.name,
            "path": str(self.path),
            "parent_domain_id": self.parent_domain_id,
            "domain_class": self.domain_class,
            "max_energy_range_uj": int(self.max_energy_range_uj),
            "include_in_package_total": bool(self.include_in_package_total),
        }


@dataclass(frozen=True)
class EnergyDiscovery:
    root: Path
    domains: tuple[EnergyDomain, ...]
    rejected: tuple[dict[str, Any], ...]

    @property
    def package_domains(self) -> tuple[EnergyDomain, ...]:
        return tuple(d for d in self.domains if d.include_in_package_total)

    @property
    def package_discovery_complete(self) -> bool:
        if not self.package_domains:
            return False
        return not any(bool(r.get("would_include_in_package_total")) for r in self.rejected)

    def record(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "domains": [d.record() for d in self.domains],
            "package_domain_ids": [d.domain_id for d in self.package_domains],
            "package_discovery_complete": self.package_discovery_complete,
            "rejected": list(self.rejected),
        }


@dataclass(frozen=True)
class EnergyReading:
    domain: EnergyDomain
    value_uj: int | None
    valid: bool
    error: str | None

    def record(self) -> dict[str, Any]:
        return {
            **self.domain.record(),
            "value_uj": self.value_uj,
            "valid": bool(self.valid),
            "error": self.error,
        }


@dataclass(frozen=True)
class EnergySnapshot:
    discovery: EnergyDiscovery
    timestamp_perf_ns: int
    readings: tuple[EnergyReading, ...]

    @property
    def reading_map(self) -> dict[str, EnergyReading]:
        return {r.domain.domain_id: r for r in self.readings}

    @property
    def package_valid(self) -> bool:
        if not self.discovery.package_discovery_complete:
            return False
        pids = {d.domain_id for d in self.discovery.package_domains}
        rows = self.reading_map
        return bool(pids) and all(pid in rows and rows[pid].valid for pid in pids)

    @property
    def all_valid(self) -> bool:
        return self.package_valid and all(r.valid for r in self.readings)

    def record(self) -> dict[str, Any]:
        return {
            "timestamp_perf_ns": int(self.timestamp_perf_ns),
            "package_valid": bool(self.package_valid),
            "all_valid": bool(self.all_valid),
            "discovery": self.discovery.record(),
            "readings": [r.record() for r in self.readings],
        }


def _read_text(path: Path) -> tuple[str | None, str | None]:
    try:
        return path.read_text(encoding="utf-8").strip(), None
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _classify_domain(name: str) -> str:
    lower = name.strip().lower()
    if lower.startswith("package") or lower.startswith("socket") or lower == "pkg":
        return "package"
    if "dram" in lower:
        return "dram"
    if lower in {"core", "cores", "pp0"} or "core" in lower:
        return "core"
    if lower in {"psys", "platform"} or "psys" in lower:
        return "psys"
    if "gpu" in lower or "graphics" in lower or lower in {"uncore", "pp1"}:
        return "gpu_or_uncore"
    if "uncore" in lower:
        return "uncore"
    return "other"


def _candidate_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    try:
        return sorted({p.parent for p in root.rglob("energy_uj")}, key=lambda p: str(p))
    except OSError:
        return []


def discover_energy_domains(root: str | Path = DEFAULT_POWERCAP_ROOT) -> EnergyDiscovery:
    """Discover readable powercap domains without silently dropping bad package counters."""
    root = Path(root)
    candidates = _candidate_dirs(root)
    candidate_ids = {p: p.relative_to(root).as_posix() for p in candidates}
    names: dict[Path, str] = {}
    for path in candidates:
        text, _ = _read_text(path / "name")
        names[path] = text or path.name

    def nearest_parent(path: Path) -> str | None:
        parent = path.parent
        while parent != root and parent != parent.parent:
            if parent in candidate_ids:
                return candidate_ids[parent]
            parent = parent.parent
        return candidate_ids.get(parent)

    domains: list[EnergyDomain] = []
    rejected: list[dict[str, Any]] = []
    for path in candidates:
        did = candidate_ids[path]
        name_text, name_err = _read_text(path / "name")
        name = name_text or path.name
        domain_class = _classify_domain(name)
        parent_id = nearest_parent(path)
        package_candidate = parent_id is None and domain_class == "package"

        energy_text, energy_err = _read_text(path / "energy_uj")
        range_text, range_err = _read_text(path / "max_energy_range_uj")
        reason: str | None = None
        try:
            initial = int(energy_text) if energy_text is not None else None
        except ValueError:
            initial = None
            reason = "malformed_energy_uj"
        try:
            max_range = int(range_text) if range_text is not None else None
        except ValueError:
            max_range = None
            reason = reason or "malformed_max_energy_range_uj"
        if name_err:
            reason = reason or "unreadable_name"
        if energy_err:
            reason = reason or "unreadable_energy_uj"
        if range_err:
            reason = reason or "unreadable_max_energy_range_uj"
        if max_range is None or max_range <= 0:
            reason = reason or "invalid_max_energy_range_uj"
        if initial is None:
            reason = reason or "invalid_energy_uj"
        elif max_range is not None and max_range > 0 and not (0 <= initial <= max_range):
            reason = reason or "energy_out_of_range"

        if reason is not None:
            rejected.append({
                "domain_id": did,
                "name": name,
                "path": str(path),
                "parent_domain_id": parent_id,
                "domain_class": domain_class,
                "would_include_in_package_total": package_candidate,
                "reason": reason,
            })
            continue

        assert max_range is not None
        domains.append(EnergyDomain(
            domain_id=did,
            name=name,
            path=path,
            parent_domain_id=parent_id,
            domain_class=domain_class,
            max_energy_range_uj=max_range,
            include_in_package_total=package_candidate,
        ))

    return EnergyDiscovery(root=root, domains=tuple(domains), rejected=tuple(rejected))


def read_energy_snapshot(
    root: str | Path = DEFAULT_POWERCAP_ROOT,
    *,
    discovery: EnergyDiscovery | None = None,
) -> EnergySnapshot:
    """Read all discovered counters and preserve invalid readings explicitly."""
    disc = discovery if discovery is not None else discover_energy_domains(root)
    rows: list[EnergyReading] = []
    for domain in disc.domains:
        text, err = _read_text(domain.path / "energy_uj")
        value: int | None = None
        error = err
        if text is not None:
            try:
                value = int(text)
            except ValueError:
                error = "malformed_energy_uj"
        if value is not None and not (0 <= value <= domain.max_energy_range_uj):
            error = "energy_out_of_range"
        rows.append(EnergyReading(domain, value, error is None and value is not None, error))
    return EnergySnapshot(disc, time.perf_counter_ns(), tuple(rows))


def _counter_delta_uj(
    start: EnergyReading,
    end: EnergyReading,
    *,
    wrap_guard_fraction: float = WRAP_GUARD_FRACTION,
) -> dict[str, Any]:
    if start.domain.domain_id != end.domain.domain_id:
        return {"available": False, "delta_uj": None, "status": "domain_identity_changed"}
    if start.domain.max_energy_range_uj != end.domain.max_energy_range_uj:
        return {"available": False, "delta_uj": None, "status": "range_changed"}
    if not start.valid or not end.valid or start.value_uj is None or end.value_uj is None:
        return {"available": False, "delta_uj": None, "status": "invalid_endpoint"}
    maximum = int(start.domain.max_energy_range_uj)
    a, b = int(start.value_uj), int(end.value_uj)
    if not (0 <= a <= maximum and 0 <= b <= maximum):
        return {"available": False, "delta_uj": None, "status": "endpoint_out_of_range"}
    if b >= a:
        return {"available": True, "delta_uj": b - a, "status": "monotonic"}

    # Two-point measurement cannot distinguish arbitrary reset from wrap. Accept a
    # single wrap only when it crosses the declared counter boundary conservatively.
    guard = float(maximum) * float(wrap_guard_fraction)
    if a >= maximum - guard and b <= guard:
        return {"available": True, "delta_uj": (maximum - a) + b, "status": "single_wrap"}
    return {"available": False, "delta_uj": None, "status": "reset_or_corrupt_decrease"}


def energy_delta(
    start: EnergySnapshot,
    end: EnergySnapshot,
    *,
    wrap_guard_fraction: float = WRAP_GUARD_FRACTION,
) -> dict[str, Any]:
    """Validate a snapshot pair and compute package energy without nested double count."""
    start_ids = {d.domain_id for d in start.discovery.domains}
    end_ids = {d.domain_id for d in end.discovery.domains}
    start_rejected = {(r.get("domain_id"), r.get("reason")) for r in start.discovery.rejected}
    end_rejected = {(r.get("domain_id"), r.get("reason")) for r in end.discovery.rejected}
    package_ids = [d.domain_id for d in start.discovery.package_domains]

    base = {
        "available": False,
        "energy_joules": None,
        "energy_microjoules": None,
        "package_domain_ids": package_ids,
        "scope": "cpu_package_rapl_not_gpu_not_whole_system",
        "scope_note": (
            "RAPL package energy is attributed by the host package counter and can include "
            "CPU cores plus other on-package components. It is not a discrete-GPU or "
            "whole-system/wall-energy measurement."
        ),
        "domain_deltas": [],
        "failure_reason": None,
    }
    if not package_ids:
        return {**base, "failure_reason": "no_package_domain"}
    if not start.discovery.package_discovery_complete or not end.discovery.package_discovery_complete:
        return {**base, "failure_reason": "package_counter_discovery_incomplete"}
    if start_ids != end_ids or start_rejected != end_rejected:
        return {**base, "failure_reason": "counter_set_changed_or_partial"}

    sm, em = start.reading_map, end.reading_map
    total = 0
    domain_rows: list[dict[str, Any]] = []
    for did in sorted(start_ids):
        if did not in sm or did not in em:
            return {**base, "failure_reason": "counter_set_changed_or_partial"}
        result = _counter_delta_uj(
            sm[did], em[did], wrap_guard_fraction=wrap_guard_fraction
        )
        row = {
            "domain_id": did,
            "name": sm[did].domain.name,
            "domain_class": sm[did].domain.domain_class,
            "parent_domain_id": sm[did].domain.parent_domain_id,
            "include_in_package_total": sm[did].domain.include_in_package_total,
            "start_uj": sm[did].value_uj,
            "end_uj": em[did].value_uj,
            "max_energy_range_uj": sm[did].domain.max_energy_range_uj,
            **result,
        }
        domain_rows.append(row)
        if not result["available"]:
            return {
                **base,
                "domain_deltas": domain_rows,
                "failure_reason": f"invalid_counter_delta:{did}:{result['status']}",
            }
        if sm[did].domain.include_in_package_total:
            total += int(result["delta_uj"])

    return {
        **base,
        "available": True,
        "energy_joules": total / 1e6,
        "energy_microjoules": total,
        "domain_deltas": domain_rows,
        "failure_reason": None,
    }


def package_energy_available(root: str | Path = DEFAULT_POWERCAP_ROOT) -> bool:
    disc = discover_energy_domains(root)
    snap = read_energy_snapshot(discovery=disc)
    return disc.package_discovery_complete and snap.all_valid


def measure_energy(
    fn: Callable[[], object],
    n_runs: int = 1,
    *,
    root: str | Path = DEFAULT_POWERCAP_ROOT,
) -> dict[str, Any]:
    """Measure a callable with explicit unavailable/invalid energy states."""
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    start = read_energy_snapshot(root)
    t0 = time.perf_counter()
    last = None
    for _ in range(int(n_runs)):
        last = fn()
    elapsed = time.perf_counter() - t0
    end = read_energy_snapshot(root)
    delta = energy_delta(start, end)
    joules = delta["energy_joules"] if delta["available"] else None
    return {
        **delta,
        "n_runs": int(n_runs),
        "joules_per_run": (float(joules) / n_runs) if joules is not None else None,
        "wall_seconds": float(elapsed),
        "avg_package_power_watts": (
            float(joules) / elapsed if joules is not None and elapsed > 0 else None
        ),
        "start_snapshot": start.record(),
        "end_snapshot": end.record(),
        "last_result_type": type(last).__name__ if last is not None else None,
    }
