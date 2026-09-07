"""Physical CPU-package energy profiling with validated Linux powercap counters.

M13 centralizes all energy-counter logic in :mod:`common.energy_counters`. Invalid,
partial, reset or corrupt readings remain unavailable; they are never clamped to
zero. Package energy is not labelled as GPU or whole-system energy.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from common.energy_counters import (
    DEFAULT_POWERCAP_ROOT,
    discover_energy_domains,
    measure_energy,
    package_energy_available,
    read_energy_snapshot,
    energy_delta,
)


def rapl_available(root: str | Path = DEFAULT_POWERCAP_ROOT) -> bool:
    """Whether a complete readable CPU-package powercap measurement is available."""
    return package_energy_available(root)


def energy_counter_inventory(root: str | Path = DEFAULT_POWERCAP_ROOT) -> dict[str, Any]:
    """Serializable explicit powercap domain inventory for provenance."""
    return discover_energy_domains(root).record()


def measure_energy_joules(
    fn: Callable[[], object],
    n_runs: int = 1,
    *,
    root: str | Path = DEFAULT_POWERCAP_ROOT,
) -> dict[str, Any]:
    """Measure CPU-package RAPL energy around ``fn``.

    Always returns an explicit record. If physical counters are unavailable or the
    window is invalid, ``available`` is false and joule fields are ``None``.
    """
    return measure_energy(fn, n_runs=n_runs, root=root)


def idle_power_watts(
    seconds: float = 5.0,
    *,
    root: str | Path = DEFAULT_POWERCAP_ROOT,
) -> float | None:
    """Observe idle CPU-package power without silently substituting zero."""
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    start = read_energy_snapshot(root)
    t0 = time.perf_counter()
    time.sleep(float(seconds))
    elapsed = time.perf_counter() - t0
    end = read_energy_snapshot(root)
    delta = energy_delta(start, end)
    joules = delta["energy_joules"] if delta["available"] else None
    return (float(joules) / elapsed) if joules is not None and elapsed > 0 else None


# --------------------------------------------------------------------------- #
# cgroup constraints
# --------------------------------------------------------------------------- #
def cgroups_available() -> bool:
    """True if ``systemd-run`` cgroup scoping is installed (usability is host-specific)."""
    import shutil

    return shutil.which("systemd-run") is not None


def cgroup_command(cpu_cores: int, mem_max_mb: int, argv: list[str]) -> list[str]:
    """Build a ``systemd-run`` scope command applying CPU/memory limits."""
    cmd = ["systemd-run", "--scope", "--quiet"]
    if cpu_cores and cpu_cores > 0:
        cmd += ["-p", f"CPUQuota={cpu_cores * 100}%"]
    if mem_max_mb and mem_max_mb > 0:
        cmd += ["-p", f"MemoryMax={mem_max_mb}M"]
    return cmd + list(argv)
