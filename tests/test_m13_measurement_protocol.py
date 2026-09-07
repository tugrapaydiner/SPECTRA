"""M13 controlled measurement-contract fixtures.

These are synthetic filesystem/counter fixtures, NOT physical hardware energy
measurements. They prove reader semantics under wrap/reset/corrupt/partial cases.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from common.energy_counters import (
    discover_energy_domains,
    energy_delta,
    measure_energy,
    read_energy_snapshot,
)
from common.measurement_env import measurement_environment
from eval.edge_energy import measure_energy_joules, measure_energy_record
from eval.memory import (
    measure_peak_ram,
    model_state_bytes,
    packed_weight_bytes,
    search_tree_tensor_bytes,
)
from eval.roofline import arithmetic_intensity, precomputed_input_reuse_traffic, roofline_report
from model.trm import TRM


def _domain(path: Path, *, name: str, energy: int | str, maximum: int | str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "name").write_text(str(name) + "\n")
    (path / "energy_uj").write_text(str(energy) + "\n")
    (path / "max_energy_range_uj").write_text(str(maximum) + "\n")
    return path


def _tree(root: Path, *, package=100, dram=20, maximum=1000):
    pkg = _domain(root / "p0", name="package-0", energy=package, maximum=maximum)
    child = _domain(pkg / "dram", name="dram", energy=dram, maximum=maximum)
    return pkg, child


def test_controlled_nested_domains_do_not_double_count(tmp_path):
    root = tmp_path / "powercap"; pkg, child = _tree(root, package=100, dram=20)
    disc = discover_energy_domains(root)
    assert [d.domain_id for d in disc.package_domains] == ["p0"]
    assert {d.name for d in disc.domains} == {"package-0", "dram"}
    child_domain = next(d for d in disc.domains if d.name == "dram")
    assert child_domain.parent_domain_id == "p0"
    assert child_domain.include_in_package_total is False

    start = read_energy_snapshot(discovery=disc)
    (pkg / "energy_uj").write_text("160\n")
    (child / "energy_uj").write_text("70\n")
    end = read_energy_snapshot(discovery=disc)
    delta = energy_delta(start, end)
    assert delta["available"] is True
    assert delta["energy_microjoules"] == 60  # package only, not package+DRAM=110
    assert any(r["name"] == "dram" and r["delta_uj"] == 50 for r in delta["domain_deltas"])
    assert "not_gpu_not_whole_system" in delta["scope"]


def test_controlled_valid_zero_is_zero_not_unavailable(tmp_path):
    root = tmp_path / "powercap"; pkg, child = _tree(root, package=400, dram=50)
    disc = discover_energy_domains(root); start = read_energy_snapshot(discovery=disc)
    end = read_energy_snapshot(discovery=disc); delta = energy_delta(start, end)
    assert delta["available"] is True
    assert delta["energy_joules"] == 0.0
    assert delta["failure_reason"] is None


def test_controlled_single_wrap_uses_counter_specific_range(tmp_path):
    root = tmp_path / "powercap"; pkg, child = _tree(root, package=950, dram=900, maximum=1000)
    disc = discover_energy_domains(root); start = read_energy_snapshot(discovery=disc)
    (pkg / "energy_uj").write_text("25\n")       # 950 -> wrap -> 25 = 75 uJ
    (child / "energy_uj").write_text("20\n")     # child wrap valid too, but not summed
    end = read_energy_snapshot(discovery=disc); delta = energy_delta(start, end)
    assert delta["available"] is True
    assert delta["energy_microjoules"] == 75
    pkg_row = next(r for r in delta["domain_deltas"] if r["name"] == "package-0")
    assert pkg_row["status"] == "single_wrap"
    assert pkg_row["max_energy_range_uj"] == 1000


def test_controlled_nonboundary_decrease_is_reset_or_corrupt_not_zero(tmp_path):
    root = tmp_path / "powercap"; pkg, child = _tree(root, package=500, dram=100)
    disc = discover_energy_domains(root); start = read_energy_snapshot(discovery=disc)
    (pkg / "energy_uj").write_text("100\n")
    (child / "energy_uj").write_text("110\n")
    end = read_energy_snapshot(discovery=disc); delta = energy_delta(start, end)
    assert delta["available"] is False
    assert delta["energy_joules"] is None
    assert "reset_or_corrupt_decrease" in delta["failure_reason"]


def test_controlled_corrupt_endpoint_and_partial_counter_set_remain_unavailable(tmp_path):
    root = tmp_path / "powercap"; pkg, child = _tree(root, package=100, dram=20)
    disc = discover_energy_domains(root); start = read_energy_snapshot(discovery=disc)
    (pkg / "energy_uj").write_text("not-an-integer\n")
    corrupt = read_energy_snapshot(discovery=disc)
    bad = energy_delta(start, corrupt)
    assert bad["available"] is False and bad["energy_joules"] is None
    assert "invalid_counter_delta" in bad["failure_reason"]

    (pkg / "energy_uj").write_text("110\n")
    start2 = read_energy_snapshot(root)
    (child / "energy_uj").unlink()
    end2 = read_energy_snapshot(root)
    partial = energy_delta(start2, end2)
    assert partial["available"] is False and partial["energy_joules"] is None
    assert partial["failure_reason"] == "counter_set_changed_or_partial"


def test_controlled_invalid_discovery_cannot_fabricate_zero(tmp_path):
    root = tmp_path / "powercap"
    _domain(root / "p0", name="package-0", energy=10, maximum="broken")
    record = measure_energy(lambda: None, root=root)
    assert record["available"] is False
    assert record["energy_joules"] is None
    explicit = measure_energy_record(lambda: None, root=root)
    assert explicit["available"] is False and explicit["energy_joules"] is None
    assert measure_energy_joules(lambda: None, root=root) is None


def test_controlled_out_of_range_counter_is_rejected(tmp_path):
    root = tmp_path / "powercap"
    _domain(root / "p0", name="package-0", energy=1001, maximum=1000)
    disc = discover_energy_domains(root)
    assert disc.package_domains == ()
    assert any(r["reason"] == "energy_out_of_range" for r in disc.rejected)


def test_memory_scopes_and_sampled_peak_are_distinct():
    model = TRM(dim=16, num_tokens=5, seq_len=16, n_layers=1, N_sup=2, max_grid_size=8)
    assert model_state_bytes(model) > 0
    payload = {"packed_linears": {
        "a": {"packed": torch.zeros(17, dtype=torch.uint8)},
        "b": {"packed": torch.zeros(9, dtype=torch.uint8)},
    }}
    assert packed_weight_bytes(payload) == 26

    child = SimpleNamespace(y=torch.zeros(1, 4, 8), z_codes=torch.zeros(1,4,8,dtype=torch.int8),
                            z_scale=torch.ones(1,4,1), children=[])
    root = SimpleNamespace(y=torch.zeros(1, 4, 8), z_codes=torch.zeros(1,4,8,dtype=torch.int8),
                           z_scale=torch.ones(1,4,1), children=[child])
    tree = search_tree_tensor_bytes(root)
    assert tree["applicable"] is True and tree["node_count"] == 2
    assert tree["tensor_storage_bytes"] > 0
    assert "excludes_python_object" in tree["scope"]

    result = measure_peak_ram(lambda: bytearray(2_000_000), interval_s=0.0005)
    assert result["peak_ram_definition"] == "window_local_sampled_process_rss_maximum"
    assert "sampling interval" in result["sampling_limitation"]
    assert result["sample_count"] >= 0
    assert "linux_vmhwm_after_mb" in result


def test_operation_convention_is_consistent_and_activation_traffic_is_included():
    t = precomputed_input_reuse_traffic(32, 64, weight_bits=2, reuse=4)
    assert t.macs == 4 * 32 * 64
    assert t.operations == 2 * t.macs
    assert t.input_activation_bytes == 4 * 64
    assert t.output_activation_bytes == 4 * 32
    assert t.requant_parameter_bytes == 4 * 32
    assert t.total_external_first_touch_bytes > t.packed_weight_bytes
    assert arithmetic_intensity(32, 64, weight_bits=2, reuse=4) == (
        t.operations / t.total_external_first_touch_bytes
    )
    rep = roofline_report(32, 64, 40.0, 10.0, recursion_steps=4)
    assert rep["operation_convention"] == "1_MAC_equals_2_arithmetic_operations"
    assert rep["cache_residency_established"] is False
    assert rep["bandwidth_bottleneck_established"] is False


def test_measurement_environment_has_required_provenance_keys():
    env = measurement_environment(backend={"backend": "test"}, compiler_flags=["-O3"])
    for key in ("cpu_model", "platform", "process_affinity_cpus", "torch_num_threads",
                "frequency_policies", "backend", "compiler_flags", "power_context",
                "energy_scope_note"):
        assert key in env
    assert "not discrete-GPU" in env["energy_scope_note"]
