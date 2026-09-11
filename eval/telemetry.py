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
import math

from model.halting import DeviceState

try:  # psutil is optional; telemetry still works (with defaults) without it.
    import psutil
except ImportError:  # pragma: no cover - environment dependent
    psutil = None


_THERMAL_ROOT = Path("/sys/class/thermal")
_RAPL_ROOT = Path("/sys/class/powercap/intel-rapl")


def battery_observation() -> dict[str, object]:
    """Observed battery fields or explicit unavailability, never policy defaults.

    A controller may still need numeric defaults on a desktop/container. Those
    defaults are not measurements and must not enter physical-energy evidence.
    """
    if psutil is None or not callable(getattr(psutil, "sensors_battery", None)):
        return {"level": None, "plugged": None, "status": "provider_unavailable"}
    try:
        batt = psutil.sensors_battery()
        if batt is None:
            return {"level": None, "plugged": None, "status": "no_battery"}
        percent = float(batt.percent)
        if not math.isfinite(percent) or not 0.0 <= percent <= 100.0:
            return {"level": None, "plugged": None, "status": "invalid_reading"}
        if type(batt.power_plugged) is not bool:
            return {"level": None, "plugged": None, "status": "invalid_reading"}
        return {"level": percent / 100.0, "plugged": batt.power_plugged, "status": "observed"}
    except (OSError, NotImplementedError, AttributeError, TypeError, ValueError) as exc:
        return {"level": None, "plugged": None, "status": "read_unavailable",
                "error_type": type(exc).__name__}


def _read_battery() -> tuple[float, float]:
    """Numeric control inputs; unobserved battery uses the historical AC default."""
    observed = battery_observation()
    if observed["status"] != "observed":
        return 1.0, 1.0
    return float(observed["level"]), 1.0 if observed["plugged"] else 0.0


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
