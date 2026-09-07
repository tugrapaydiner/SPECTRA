"""Host provenance for defensible performance/energy measurements."""
from __future__ import annotations

import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import torch


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
            if line.lower().startswith("hardware"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or None


def cpu_affinity() -> list[int] | None:
    try:
        return sorted(int(x) for x in os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


def frequency_policies(root: str | Path = "/sys/devices/system/cpu/cpufreq") -> list[dict[str, Any]]:
    base = Path(root); rows: list[dict[str, Any]] = []
    if not base.exists():
        return rows
    try:
        paths = sorted(p for p in base.glob("policy*") if p.is_dir())
    except OSError:
        return rows
    for p in paths:
        rows.append({
            "policy": p.name,
            "affected_cpus": _read(p / "affected_cpus"),
            "related_cpus": _read(p / "related_cpus"),
            "scaling_governor": _read(p / "scaling_governor"),
            "scaling_driver": _read(p / "scaling_driver"),
            "scaling_min_freq_khz": _read(p / "scaling_min_freq"),
            "scaling_max_freq_khz": _read(p / "scaling_max_freq"),
            "scaling_cur_freq_khz": _read(p / "scaling_cur_freq"),
            "cpuinfo_min_freq_khz": _read(p / "cpuinfo_min_freq"),
            "cpuinfo_max_freq_khz": _read(p / "cpuinfo_max_freq"),
        })
    return rows


def power_context(root: str | Path = "/sys/class/power_supply") -> dict[str, Any]:
    base = Path(root)
    if not base.exists():
        return {"available": False, "supplies": [], "summary": "power_supply_sysfs_unavailable"}
    supplies: list[dict[str, Any]] = []
    try:
        dirs = sorted(p for p in base.iterdir() if p.is_dir() or p.is_symlink())
    except OSError:
        dirs = []
    for p in dirs:
        supplies.append({
            "name": p.name,
            "type": _read(p / "type"),
            "online": _read(p / "online"),
            "status": _read(p / "status"),
            "capacity_percent": _read(p / "capacity"),
        })
    return {"available": bool(supplies), "supplies": supplies,
            "summary": "observed_from_sysfs_not_a_power_meter"}


def compiler_version(executable: str = "g++") -> str | None:
    try:
        return subprocess.check_output([executable, "--version"], text=True,
                                       stderr=subprocess.STDOUT, timeout=5).splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def measurement_environment(
    *,
    backend: dict[str, Any] | str | None = None,
    compiler_flags: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        interop = int(torch.get_num_interop_threads())
    except RuntimeError:
        interop = None
    return {
        "cpu_model": cpu_model(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "logical_cpu_count": os.cpu_count(),
        "process_affinity_cpus": cpu_affinity(),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": interop,
        "frequency_policies": frequency_policies(),
        "frequency_policy_note": "observed policy/frequency metadata; M13 does not claim a locked frequency",
        "backend": backend,
        "compiler": compiler_version("g++"),
        "compiler_flags": list(compiler_flags) if compiler_flags is not None else None,
        "power_context": power_context(),
        "energy_scope_note": (
            "RAPL package energy can include CPU cores and other on-package components; "
            "it is not discrete-GPU energy and not whole-system/wall energy."
        ),
        **(extra or {}),
    }
