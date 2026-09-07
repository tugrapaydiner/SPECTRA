"""Console logging plus a tiny JSONL metric logger.

Section 29 of the BLUEPRINT requires logging a long list of training/quant/router
metrics "from day one". We keep this minimal: a configured stdlib logger for
human-readable output, and a :class:`MetricLogger` that appends one JSON object
per step to a ``.jsonl`` file for later analysis (no wandb dependency required).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

_CONFIGURED = False


def setup_logging(level: str | int = "INFO") -> None:
    """Configure the root logger once with a concise, timestamped format."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str, level: str | int = "INFO") -> logging.Logger:
    """Return a module-level logger, ensuring logging is configured."""
    setup_logging(level)
    return logging.getLogger(name)


class MetricLogger:
    """Append-only JSONL metric sink.

    Each call to :meth:`log` writes a single JSON line, so runs can be inspected
    or plotted later without a heavyweight tracking dependency. Flushing every
    line keeps partial runs analysable after a crash.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def log(self, global_step: int, **metrics: Any) -> None:
        """Write one metric row with exactly one authoritative step value.

        ``metrics`` may already carry a ``step`` field (evaluation history does).
        That is accepted only when it agrees with ``global_step``; this avoids
        both duplicate-argument failures and silently inconsistent JSONL rows.
        """
        if "step" in metrics and int(metrics["step"]) != int(global_step):
            raise ValueError(
                f"metric step mismatch: positional={global_step}, row={metrics['step']}"
            )
        record = {"step": int(global_step), **metrics}
        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "MetricLogger":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
