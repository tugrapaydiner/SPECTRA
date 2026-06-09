"""Tests for the firehose SpectraTelemetryLogger pipelines."""

import json
import math

import numpy as np
import pytest
import torch

from common.telemetry_logger import (
    ChromeTracer,
    SpectraTelemetryLogger,
    extract_deepest_attention,
    policy_kl,
)
from eval.latent_mcts import LatentNativeMCTS
from model.energy import LatentEnergyVerifier
from model.latent_action import LatentActionCodebook
from model.trm import TRM


# --- 1. Chrome execution tracer ------------------------------------------- #
def test_chrome_tracer_emits_python_and_cpp_lanes(tmp_path):
    tr = ChromeTracer()
    with tr.span("python_recursion_step", cat="python", tid=0):
        pass
    with tr.span("ternary_gemv", cat="cpp", tid=1):
        for _ in range(1000):
            pass
    data = json.load(open(tr.export(tmp_path / "trace.json")))

    assert "traceEvents" in data
    x = [e for e in data["traceEvents"] if e.get("ph") == "X"]
    assert len(x) == 2
    assert all({"name", "ts", "dur", "pid", "tid"} <= set(e) for e in x)
    assert {e["tid"] for e in x} == {0, 1}                      # GIL lane + C++ lane
    assert any(e.get("ph") == "M" and e["name"] == "thread_name" for e in data["traceEvents"])


# --- 2. Policy divergence -------------------------------------------------- #
def test_policy_kl_matches_manual_and_is_nonnegative():
    p = torch.tensor([0.5, 0.5])
    assert policy_kl(p, p) == pytest.approx(0.0, abs=1e-6)
    p, q = torch.tensor([0.9, 0.1]), torch.tensor([0.5, 0.5])
    manual = 0.9 * math.log(0.9 / 0.5) + 0.1 * math.log(0.1 / 0.5)
    assert policy_kl(p, q) == pytest.approx(manual, abs=1e-5)
    assert policy_kl(p, q) >= 0.0


def test_log_mcts_policy_writes_kl(tmp_path):
    torch.manual_seed(0)
    model = TRM(dim=32, num_tokens=5, seq_len=16, N_sup=3, T=1, max_grid_size=8)
    verifier = LatentEnergyVerifier(num_tokens=5, dim=32, n_layers=1, max_grid_size=8)
    codebook = LatentActionCodebook(dim=32, n_actions=3)
    mcts = LatentNativeMCTS(model, verifier, codebook, 4, 4, n_rollouts=10)
    mcts.search(torch.randint(0, 5, (1, 16)))

    with SpectraTelemetryLogger(tmp_path) as log:
        rec = log.log_mcts_policy(0, mcts)
    assert rec["kl_posterior_to_prior"] >= 0.0
    assert len(rec["posterior"]) == codebook.n_actions
    line = json.loads(open(tmp_path / "policy_kl.jsonl").read().splitlines()[0])
    assert "kl_prior_to_posterior" in line


# --- 3. Deepest-layer attention + connectivity ----------------------------- #
@pytest.mark.parametrize("ternary", [False, True])
def test_extract_deepest_attention_rows_are_softmax(ternary):
    torch.manual_seed(0)
    model = TRM(dim=32, num_tokens=5, seq_len=16, N_sup=2, T=1, max_grid_size=8, ternary=ternary)
    w = extract_deepest_attention(model, torch.randint(0, 5, (1, 16)), 4, 4)
    assert w.shape == (16, 16)
    assert torch.allclose(w.sum(-1), torch.ones(16), atol=1e-4)


def test_log_connectivity_npz_roundtrip(tmp_path):
    L = 16
    attn = torch.rand(L, L)
    attn = attn / attn.sum(-1, keepdim=True)
    mask = torch.zeros(L, dtype=torch.bool)
    mask[[1, 3, 5, 7]] = True

    with SpectraTelemetryLogger(tmp_path) as log:
        path = log.log_connectivity(2, attn, mask, height=4, width=4)
    d = np.load(path)
    assert list(d["active_idx"]) == [1, 3, 5, 7]
    assert d["sub_dense"].shape == (4, 4)                       # active x active
    assert d["edge_rows"].size == d["edge_weights"].size       # COO consistency
    assert int(d["height"]) == 4 and int(d["seq_len"]) == 16


# --- torch.profiler chrome export ----------------------------------------- #
def test_torch_profiler_exports_chrome_trace(tmp_path):
    log = SpectraTelemetryLogger(tmp_path)
    out = tmp_path / "tp.json"
    with log.torch_profiler(out):
        a = torch.randn(48, 48)
        (a @ a).sum()
    data = json.load(open(out))
    assert "traceEvents" in data
    log.close()
