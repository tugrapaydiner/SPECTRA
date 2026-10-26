"""Configuration loading, seeding and explicit CPU selection contracts."""

from common import DotDict, load_config, resolve_device, set_seed


def test_imports_and_seed():
    """Shared utilities import and seeding runs without error."""
    set_seed(0)
    device = resolve_device("cpu")
    assert device.type == "cpu"


def test_config_loads_with_inheritance():
    """A task config inherits base fields and applies its own overrides."""
    cfg = load_config("config/sudoku.yaml")
    assert isinstance(cfg, DotDict)
    # Inherited from base.yaml:
    assert cfg.model.n == 6
    assert cfg.seed == 42
    # Defined in sudoku.yaml:
    assert cfg.task == "sudoku"
    assert cfg.data.seq_len == 81


def test_config_cli_override():
    """Dotted-key overrides parse with YAML scalar semantics."""
    cfg = load_config("config/sudoku.yaml", overrides=["model.dim=256", "seed=7"])
    assert cfg.model.dim == 256
    assert cfg.seed == 7
