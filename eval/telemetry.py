"""Device telemetry wrapper (BLUEPRINT section 21).

Reads a :class:`~model.halting.DeviceState` from the host where possible and
degrades gracefully where not. The blueprint targets legacy Linux x86 (battery in
``/sys/class/power_supply``, thermal in ``/sys/class/thermal``, energy in
``/sys/class/powercap/intel-rapl``); on other platforms we fall back to psutil
(battery) and sensible defaults (thermal/latency), so the same code runs on the
Windows dev box and the Linux eval box.
"""

from __future__ import annotations

from pathlib import Path

from model.halting import DeviceState

try:  # psutil is optional; telemetry still works (with defaults) without it.
    import psutil
except ImportError:  # pragma: no cover - environment dependent
    psutil = None


_THERMAL_ROOT = Path("/sys/class/thermal")
_RAPL_ROOT = Path("/sys/class/powercap/intel-rapl")


def _read_battery() -> tuple[float, float]:
    """Return ``(battery_level 0..1, power_mode 0/1)`` using psutil if available."""
    if psutil is None:
        return 1.0, 1.0
    try:
        batt = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
    except (OSError, NotImplementedError):
        # Containers may hide power_supply even when psutil supports the API.
        # These are policy defaults, not measurements or physical-energy data.
        batt = None
    if batt is None:
        return 1.0, 1.0  # desktop / no battery -> treat as plugged in, full
    level = max(0.0, min(1.0, batt.percent / 100.0))
    power_mode = 1.0 if batt.power_plugged else 0.0
    return level, power_mode


def _read_thermal() -> float:
    """Return a normalised thermal level 0..1 from Linux thermal zones (else 0)."""
    if not _THERMAL_ROOT.exists():
        return 0.0
    temps = []
    for zone in _THERMAL_ROOT.glob("thermal_zone*/temp"):
        try:
            milli_c = int(zone.read_text().strip())
            temps.append(milli_c / 1000.0)
        except (OSError, ValueError):
            continue
    if not temps:
        return 0.0
    # Map 40 C (cool) .. 95 C (throttle) to 0 .. 1.
    hottest = max(temps)
    return max(0.0, min(1.0, (hottest - 40.0) / 55.0))


def _read_available_ram_mb() -> float:
    if psutil is None:
        return 8192.0
    return psutil.virtual_memory().available / (1024 * 1024)


def rapl_available() -> bool:
    """True if Intel/AMD RAPL energy counters are exposed (Linux only)."""
    return _RAPL_ROOT.exists()


def read_device_state(
    latency_budget_ms: float = 1000.0,
    device_class: int = 1,
) -> DeviceState:
    """Sample a :class:`DeviceState` from the host (graceful where unavailable).

    Args:
        latency_budget_ms: Caller-supplied latency budget (not host-readable).
        device_class: Caller-supplied device class id.
    """
    battery_level, power_mode = _read_battery()
    return DeviceState(
        battery_level=battery_level,
        thermal_level=_read_thermal(),
        latency_budget_ms=latency_budget_ms,
        power_mode=power_mode,
        available_ram_mb=_read_available_ram_mb(),
        device_class=device_class,
    )
