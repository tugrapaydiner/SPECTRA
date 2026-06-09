"""Physical energy profiling via RAPL + cgroup constraints (BLUEPRINT section 22).

The paper claim uses *measured joules*, not FLOP estimates. On Linux x86 we read
Intel/AMD RAPL energy counters in ``/sys/class/powercap/intel-rapl`` and (for the
catastrophic/standard edge profiles) launch under cgroup CPU/memory limits via
``systemd-run``. None of this exists on Windows, so every entry point degrades
gracefully (``rapl_available()`` is False and energy measurement returns ``None``)
-- the harness still runs; only the joule numbers require the Linux eval box.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

_RAPL_ROOT = Path("/sys/class/powercap/intel-rapl")


# --------------------------------------------------------------------------- #
# RAPL energy
# --------------------------------------------------------------------------- #
def rapl_available() -> bool:
    """True if RAPL energy counters are readable (Linux Intel/AMD only)."""
    return _RAPL_ROOT.exists() and any(_RAPL_ROOT.glob("intel-rapl:*/energy_uj"))


def _read_rapl_microjoules() -> int | None:
    """Sum the package-domain RAPL energy counters in microjoules."""
    if not rapl_available():
        return None
    total = 0
    for counter in _RAPL_ROOT.glob("intel-rapl:*/energy_uj"):
        try:
            total += int(counter.read_text().strip())
        except (OSError, ValueError):
            continue
    return total


def measure_energy_joules(
    fn: Callable[[], object], n_runs: int = 1
) -> dict[str, float] | None:
    """Measure energy (J) and wall time (s) for ``n_runs`` of ``fn`` via RAPL.

    Returns ``None`` when RAPL is unavailable (e.g. Windows). Energy is the RAPL
    counter delta; subtract an idle baseline (see :func:`idle_power_watts`) for
    net inference energy as in section 22.3.
    """
    start_uj = _read_rapl_microjoules()
    if start_uj is None:
        return None
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn()
    elapsed = time.perf_counter() - t0
    end_uj = _read_rapl_microjoules()
    joules = max(0, (end_uj - start_uj)) / 1e6  # counters are monotonic per window
    return {
        "energy_joules": joules,
        "joules_per_run": joules / max(1, n_runs),
        "wall_seconds": elapsed,
        "avg_power_watts": joules / elapsed if elapsed > 0 else float("nan"),
    }


def idle_power_watts(seconds: float = 5.0) -> float | None:
    """Estimate idle package power by sampling RAPL over an idle window (W)."""
    start_uj = _read_rapl_microjoules()
    if start_uj is None:
        return None
    t0 = time.perf_counter()
    time.sleep(seconds)
    elapsed = time.perf_counter() - t0
    end_uj = _read_rapl_microjoules()
    return (max(0, end_uj - start_uj) / 1e6) / elapsed if elapsed > 0 else float("nan")


# --------------------------------------------------------------------------- #
# cgroup constraints (section 22.2)
# --------------------------------------------------------------------------- #
def cgroups_available() -> bool:
    """True if ``systemd-run`` cgroup scoping is usable (Linux)."""
    import shutil

    return shutil.which("systemd-run") is not None


def cgroup_command(cpu_cores: int, mem_max_mb: int, argv: list[str]) -> list[str]:
    """Build a ``systemd-run`` scope command applying CPU/memory limits.

    ``CPUQuota=100%`` per core; ``MemoryMax`` caps RAM. Used to simulate the
    Catastrophic Edge (1 core / 512 MB) and Standard Edge (2 cores / 2 GB)
    profiles on a single machine (section 22.2). A value of 0 means unconstrained.
    """
    cmd = ["systemd-run", "--scope", "--quiet"]
    if cpu_cores and cpu_cores > 0:
        cmd += ["-p", f"CPUQuota={cpu_cores * 100}%"]
    if mem_max_mb and mem_max_mb > 0:
        cmd += ["-p", f"MemoryMax={mem_max_mb}M"]
    return cmd + list(argv)
