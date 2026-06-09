"""CPU frequency pinning for iso-joule measurement fidelity (DT #4).

AVX2 pulls large current; a legacy x86 core hits its TDP and downclocks within
milliseconds. Under throttling, latency rises while power falls -- a non-linear
artifact that corrupts the iso-joule scaling curve (you are no longer measuring a
fixed silicon operating point). Scientific validity requires holding frequency
constant during the measurement.

This module pins the Linux ``cpufreq`` governor to ``userspace`` at a fixed
frequency and disables Intel Turbo Boost / AMD Core Performance Boost, restoring
the previous state afterward. It is a no-op (clearly reported) on non-Linux hosts
or without the needed permissions, so the harness never crashes -- it just records
that the run was not frequency-locked.
"""

from __future__ import annotations

import glob
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_CPU_ROOT = Path("/sys/devices/system/cpu")
_INTEL_NO_TURBO = _CPU_ROOT / "intel_pstate" / "no_turbo"
_CPUFREQ_BOOST = _CPU_ROOT / "cpufreq" / "boost"


def cpufreq_available() -> bool:
    """True if Linux cpufreq sysfs is present (governors are controllable)."""
    return bool(glob.glob(str(_CPU_ROOT / "cpu*/cpufreq/scaling_governor")))


def _governor_paths() -> list[Path]:
    return [Path(p) for p in glob.glob(str(_CPU_ROOT / "cpu*/cpufreq/scaling_governor"))]


def _write(path: Path, value, use_sudo: bool = False) -> bool:
    """Write ``value`` to a sysfs ``path``; return True on success (never raises)."""
    try:
        if use_sudo:  # subprocess binding for privileged sysfs writes
            subprocess.run(
                ["sudo", "tee", str(path)], input=f"{value}\n".encode(),
                check=True, stdout=subprocess.DEVNULL,
            )
        else:
            path.write_text(f"{value}\n")
        return True
    except (OSError, PermissionError, subprocess.CalledProcessError):
        return False


def available_governors() -> list[str]:
    p = _CPU_ROOT / "cpu0" / "cpufreq" / "scaling_available_governors"
    if not p.exists():
        return []
    return p.read_text().split()


def read_state() -> dict:
    """Snapshot the current governor / setspeed / turbo state (graceful)."""
    if not cpufreq_available():
        return {"available": False}
    govs = [p.read_text().strip() for p in _governor_paths()]
    turbo = None
    if _INTEL_NO_TURBO.exists():
        turbo = "off" if _INTEL_NO_TURBO.read_text().strip() == "1" else "on"
    elif _CPUFREQ_BOOST.exists():
        turbo = "on" if _CPUFREQ_BOOST.read_text().strip() == "1" else "off"
    return {"available": True, "governors": govs, "turbo": turbo}


def set_governor(governor: str, use_sudo: bool = False) -> bool:
    return all(_write(p, governor, use_sudo) for p in _governor_paths())


def set_frequency_khz(freq_khz: int, use_sudo: bool = False) -> bool:
    """Pin every core to ``freq_khz`` (requires the ``userspace`` governor)."""
    ok = True
    for gov in _governor_paths():
        ok &= _write(gov.parent / "scaling_setspeed", int(freq_khz), use_sudo)
    return ok


def set_turbo(enabled: bool, use_sudo: bool = False) -> bool:
    """Enable/disable Turbo Boost / Core Performance Boost across vendors."""
    if _INTEL_NO_TURBO.exists():
        return _write(_INTEL_NO_TURBO, 0 if enabled else 1, use_sudo)  # no_turbo is inverted
    if _CPUFREQ_BOOST.exists():
        return _write(_CPUFREQ_BOOST, 1 if enabled else 0, use_sudo)
    return False


@contextmanager
def pinned_frequency(
    governor: str = "userspace",
    freq_khz: int | None = None,
    disable_turbo: bool = True,
    use_sudo: bool = False,
) -> Iterator[dict]:
    """Hold the CPU at a fixed operating point for the duration of the block.

    Yields a status dict (``pinned``: whether the lock actually took effect). On
    exit the previous governor/turbo state is restored. Wrap the *measured*
    inference loop with this so RAPL joules reflect a constant silicon state.
    """
    before = read_state()
    status = {"pinned": False, "reason": "", "before": before}
    if not before.get("available"):
        status["reason"] = "cpufreq sysfs unavailable (non-Linux or no /sys)"
        yield status
        return

    prev_gov = before["governors"][0] if before["governors"] else None
    prev_turbo = before.get("turbo")
    applied_gov = set_governor(governor, use_sudo)
    applied_freq = set_frequency_khz(freq_khz, use_sudo) if (freq_khz and governor == "userspace") else True
    applied_turbo = set_turbo(False, use_sudo) if disable_turbo else True

    status["pinned"] = bool(applied_gov and applied_freq)
    if not status["pinned"]:
        status["reason"] = "permission denied (run as root or pass use_sudo=True)"
    try:
        yield status
    finally:  # restore the prior operating point
        if prev_gov:
            set_governor(prev_gov, use_sudo)
        if prev_turbo is not None:
            set_turbo(prev_turbo == "on", use_sudo)
