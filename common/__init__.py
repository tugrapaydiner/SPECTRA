"""Cross-cutting utilities shared across SPECTRA (seeding, config, logging).

These are intentionally lightweight and dependency-free beyond the core stack so
that every phase -- training scripts, tests, and evaluation -- can rely on them.
"""

from common.config import DotDict, load_config
from common.logging_utils import get_logger, setup_logging
from common.seed import resolve_device, set_seed

__all__ = [
    "DotDict",
    "load_config",
    "get_logger",
    "setup_logging",
    "set_seed",
    "resolve_device",
]
