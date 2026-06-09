"""SpectraTelemetryLogger -- the full-firehose, ground-truth run recorder.

Pure data extraction. NO visualization libraries, NO synthetic data: every value
is read from a live model, a finished search tree, or the host's hardware counters.
The five production data streams (all decoupled, all resumable/appendable):

  1. MCTS tree dumper        -> mcts_graphs.jsonl       (one JSON node per line)
  2. Hardware profiler       -> hardware_telemetry.csv  (RAPL J / peak RAM / latency)
  3. Latent tensor exporter  -> latent_states.parquet   (INT8 z + active-token mask)
  4. Microsecond tracer      -> trace.json / torch_trace.json (Chrome/Perfetto)
  5. Scaling-law aggregator  -> spectra_scaling_laws.csv (macro per-task results)

Plus the original forensic pipelines this class already shipped: per-step policy
KL (System-1 prior vs System-2 visit posterior) and the deepest-layer active-token
connectivity graph. Hook points are the real SPECTRA APIs:
``eval.latent_mcts.LatentNativeMCTS`` (``.root`` / ``_LatentNode``),
``model.energy.EnsembleLatentEnergyVerifier.value_with_uncertainty``, and
``model.trm.TRM.forward`` step outputs.
"""

from __future__ import annotations

import csv
import json
import math
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F

from common.logging_utils import MetricLogger


# --------------------------------------------------------------------------- #
# Host hardware readers (Linux RAPL + psutil RSS) -- real counters, graceful
# degradation to None/NaN where unavailable (mirrors eval/edge_energy.py + memory).
# --------------------------------------------------------------------------- #
_RAPL_ROOT = Path("/sys/class/powercap/intel-rapl")

try:  # psutil is optional; RSS becomes NaN without it.
    import psutil
except ImportError:  # pragma: no cover - environment dependent
    psutil = None


def rapl_available() -> bool:
    """True if Intel/AMD RAPL package energy counters are readable (Linux x86)."""
    return _RAPL_ROOT.exists() and any(_RAPL_ROOT.glob("intel-rapl:*/energy_uj"))


def read_rapl_microjoules() -> int | None:
    """Sum the package-domain RAPL energy counters (microjoules), or None."""
    if not rapl_available():
        return None
    total = 0
    for counter in _RAPL_ROOT.glob("intel-rapl:*/energy_uj"):
        try:
            total += int(counter.read_text().strip())
        except (OSError, ValueError):
            continue
    return total


def read_rss_mb() -> float:
    """Resident set size of this process in MB (NaN if psutil is absent)."""
    if psutil is None:
        return float("nan")
    return psutil.Process().memory_info().rss / (1024 * 1024)


def _quantize_int8(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token symmetric INT8 quantize -- identical to eval/latent_mcts._quantize_int8.

    Returns ``(codes int8 [.., D], scale float [.., 1])`` so the exported codes are
    the literal cache-resident representation, not a re-derived approximation.
    """
    scale = z.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    codes = torch.clamp(torch.round(z / scale), -128, 127).to(torch.int8)
    return codes, scale


# --------------------------------------------------------------------------- #
# 4. Microsecond execution tracer (Chrome Trace Event Format)
# --------------------------------------------------------------------------- #
class ChromeTracer:
    """Records duration events in the Chrome Trace Event Format (microseconds).

    Lane (``tid``) convention: 0 = Python interpreter / GIL, 1 = C++ AVX2 kernel,
    2 = MCTS. Drop the exported JSON into ``chrome://tracing`` or Perfetto.
    """

    LANES = {0: "Python interpreter (GIL)", 1: "C++ AVX2 GEMV kernel", 2: "Latent MCTS"}

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.pid = os.getpid()
        self._t0 = time.perf_counter_ns()

    def _now_us(self) -> float:
        return (time.perf_counter_ns() - self._t0) / 1000.0

    @contextmanager
    def span(self, name: str, cat: str = "python", tid: int = 0,
             args: dict | None = None) -> Iterator[None]:
        """Time a complete (``ph="X"``) event on lane ``tid``."""
        ts = self._now_us()
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            dur = (time.perf_counter_ns() - start) / 1000.0
            self.events.append({
                "name": name, "cat": cat, "ph": "X", "ts": ts, "dur": dur,
                "pid": self.pid, "tid": tid, "args": args or {},
            })

    def instant(self, name: str, tid: int = 0, args: dict | None = None) -> None:
        self.events.append({"name": name, "ph": "i", "s": "t", "ts": self._now_us(),
                            "pid": self.pid, "tid": tid, "args": args or {}})

    def export(self, path: str | Path) -> str:
        """Write ``{traceEvents: [...]}`` with thread-name metadata for each lane."""
        meta = [{"name": "thread_name", "ph": "M", "pid": self.pid, "tid": tid,
                 "args": {"name": label}} for tid, label in self.LANES.items()]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"traceEvents": meta + self.events, "displayTimeUnit": "ms"}, fh)
        return str(path)


# --------------------------------------------------------------------------- #
# Policy divergence (System 1 prior vs System 2 posterior)
# --------------------------------------------------------------------------- #
def policy_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
    """KL(p || q) for two action distributions (normalised internally)."""
    p = p.float() / p.float().sum().clamp_min(eps)
    q = q.float() / q.float().sum().clamp_min(eps)
    return float((p * (p.clamp_min(eps).log() - q.clamp_min(eps).log())).sum())


# --------------------------------------------------------------------------- #
# Deepest-layer attention + routing connectivity
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_deepest_attention(model, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Head-averaged attention ``[L, L]`` of the deepest block's *last* recursion call.

    Works for both the ternary ``SelfAttention`` (recomputed from q,k) and the FP16
    ``nn.MultiheadAttention`` (``need_weights=True``). B=1.
    """
    captured: dict[str, torch.Tensor] = {}

    def hook(_mod, inp, _out):
        captured["h"] = inp[0].detach()  # last call wins -> deepest recursion step

    block = model.blocks[-1]
    handle = block.register_forward_hook(hook)
    try:
        model(x[:1], height=height, width=width)
    finally:
        handle.remove()

    h = captured["h"][:1]
    qn = block.norm1(h)
    if getattr(block, "ternary_attn", False):
        att = block.attn
        b, n, _ = qn.shape
        q = att.q(qn).view(b, n, att.heads, att.head_dim).transpose(1, 2)
        k = att.k(qn).view(b, n, att.heads, att.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(att.head_dim)
        weights = scores.softmax(dim=-1).mean(dim=1)  # [B, L, L] head-averaged
    else:
        _, weights = block.attn(qn, qn, qn, need_weights=True, average_attn_weights=True)
    return weights[0]  # [L, L]


# --------------------------------------------------------------------------- #
# 2. Hardware monitor -- background sampling thread (RAPL + RSS) during forward()
# --------------------------------------------------------------------------- #
class HardwareMonitor(threading.Thread):
    """Daemon thread that samples (perf_time, RSS_MB, RAPL_uJ) at a fixed interval.

    Designed to run *concurrently* with a ``forward()`` / search call so the energy
    and peak-RAM are measured under real load. RAPL/RSS read None/NaN where the host
    does not expose them; the harness records that fact rather than faking a number.
    """

    def __init__(self, interval_s: float = 0.005):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self._stop_event = threading.Event()
        self.samples: list[tuple[float, float, int | None]] = []

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.samples.append((time.perf_counter(), read_rss_mb(), read_rapl_microjoules()))
            time.sleep(self.interval_s)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=2.0)


# --------------------------------------------------------------------------- #
# The logger
# --------------------------------------------------------------------------- #
class SpectraTelemetryLogger:
    """Full-firehose run logger: 5 ground-truth data streams + forensic extras."""

    def __init__(self, run_dir: str | Path, run_id: str | None = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or self.run_dir.name
        # Stream 4 (tracing) + the original forensic pipelines.
        self.metrics = MetricLogger(self.run_dir / "metrics.jsonl")
        self.tracer = ChromeTracer()
        self.policy_log = MetricLogger(self.run_dir / "policy_kl.jsonl")
        self._connectivity_dir = self.run_dir / "connectivity"
        # Lazily-opened sinks for streams 1/2/3/5 (so empty streams write no file).
        self._mcts_fh = None                      # JSONL handle (stream 1)
        self._mcts_search_counter = 0
        self._csv_writers: dict[str, tuple] = {}  # streams 2 + 5
        self._parquet_writer = None               # stream 3
        self._parquet_path = None
        self._hooks: list[Any] = []               # forward hooks to remove on close
        self._current_task_id = "run"

    # --- baseline metrics ---
    def log(self, step: int, **metrics: Any) -> None:
        self.metrics.log(step, **metrics)

    # === (4) Microsecond tracer ============================================ #
    @contextmanager
    def trace(self, name: str, cat: str = "python", tid: int = 0,
              args: dict | None = None) -> Iterator[None]:
        with self.tracer.span(name, cat=cat, tid=tid, args=args):
            yield

    def python_span(self, name: str = "python_recursion_loop", **args: Any):
        """Lane-0 span: time spent in the Python interpreter (under the GIL)."""
        return self.trace(name, cat="python", tid=0, args=args or None)

    def kernel_span(self, name: str = "cpp_avx2_gemv", **args: Any):
        """Lane-1 span: wrap the C++ AVX2 GEMV call to prove it overlaps the GIL."""
        return self.trace(name, cat="cpp", tid=1, args=args or None)

    def mcts_span(self, name: str = "latent_mcts", **args: Any):
        """Lane-2 span: a Latent-MCTS rollout / verifier batch."""
        return self.trace(name, cat="mcts", tid=2, args=args or None)

    def export_chrome_trace(self, path: str | Path | None = None) -> str:
        return self.tracer.export(path or self.run_dir / "trace.json")

    @staticmethod
    def torch_region(name: str):
        """``torch.profiler.record_function(name)`` -- label regions inside torch_profiler."""
        return torch.profiler.record_function(name)

    @contextmanager
    def torch_profiler(self, path: str | Path | None = None) -> Iterator[Any]:
        """Block-scoped ``torch.profiler`` that exports a Chrome trace on exit.

        CPU (+CUDA if present) activities, op shapes and memory recorded;
        ``export_chrome_trace`` writes Perfetto/chrome-loadable JSON. Combine with
        ``torch_region("python_loop")`` / ``torch_region("cpp_avx2_gemv")`` to see
        the kernel and the Python loop on the timeline.
        """
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        prof = torch.profiler.profile(
            activities=activities, record_shapes=True,
            profile_memory=True, with_stack=False,
        )
        prof.start()
        try:
            yield prof
        finally:
            prof.stop()
            out = str(path or self.run_dir / "torch_trace.json")
            prof.export_chrome_trace(out)

    # === policy divergence (forensic) ====================================== #
    def log_policy_divergence(self, step: int, prior: torch.Tensor,
                              posterior: torch.Tensor) -> dict[str, float]:
        """Log KL between the System-1 prior and System-2 visit posterior."""
        rec = {
            "kl_prior_to_posterior": policy_kl(prior, posterior),
            "kl_posterior_to_prior": policy_kl(posterior, prior),
            "prior": [float(v) for v in prior.flatten()],
            "posterior": [float(v) for v in posterior.flatten()],
        }
        self.policy_log.log(step, **rec)
        return rec

    def log_mcts_policy(self, step: int, mcts, prior: torch.Tensor | None = None,
                        temperature: float = 1.0) -> dict[str, float]:
        """Pull the posterior from a finished MCTS search + the codebook prior."""
        posterior = mcts.root_visit_policy(temperature=temperature).detach().cpu()
        if prior is None:
            prior = mcts.codebook.priors().detach().cpu()
        return self.log_policy_divergence(step, prior, posterior)

    # === connectivity (forensic) =========================================== #
    def log_connectivity(self, step: int, attention: torch.Tensor,
                         active_mask: torch.Tensor, height: int, width: int,
                         threshold: float = 1e-3) -> str:
        """Dump the active-token attention subgraph as a compressed ``.npz`` (COO)."""
        self._connectivity_dir.mkdir(parents=True, exist_ok=True)
        attn = attention.detach().cpu().numpy()
        active_idx = np.nonzero(active_mask.detach().cpu().numpy().reshape(-1))[0].astype(np.int32)
        sub = attn[np.ix_(active_idx, active_idx)]              # active x active
        rows, cols = np.nonzero(sub > threshold)               # sparse COO
        path = self._connectivity_dir / f"step_{step:06d}.npz"
        np.savez_compressed(
            path,
            active_idx=active_idx,
            edge_rows=active_idx[rows], edge_cols=active_idx[cols],
            edge_weights=sub[rows, cols].astype(np.float32),
            sub_dense=sub.astype(np.float32),
            height=np.int32(height), width=np.int32(width), seq_len=np.int32(attn.shape[0]),
        )
        return str(path)

    # === (1) MCTS tree dumper -> mcts_graphs.jsonl ========================= #
    @torch.no_grad()
    def dump_mcts_tree(self, mcts, x: torch.Tensor, task_id: str,
                       width: int | None = None, chunk: int = 256) -> int:
        """Traverse a *finished* Latent-MCTS tree and dump the whole graph to JSONL.

        Per-node fields: node_id, parent_id, action (index into the codebook;
        0 = identity), depth, visits N, value_sum W, q_value (=W/N, the search
        backup that the PRM regresses onto), prm_process_reward (the energy
        verifier's value V_psi(x,z) -- the learned PRM's score of this latent),
        epistemic_sigma (ensemble std; NaN for a single verifier), prior P,
        is_leaf, n_children, is_principal_variation. Call ``mcts.search(x)`` first.

        Returns the number of nodes written.
        """
        root = getattr(mcts, "root", None)
        if root is None:
            raise ValueError("run mcts.search(x) before dump_mcts_tree()")
        width = int(width if width is not None else getattr(mcts, "width", x.shape[1]))

        # 1) Enumerate every node (iterative DFS). action = index within parent.children.
        order: list[tuple] = []           # (node, parent_node, action, depth)
        node_id: dict[int, int] = {}
        stack = [(root, None, None, 0)]
        while stack:
            nd, parent, action, depth = stack.pop()
            node_id[id(nd)] = len(order)
            order.append((nd, parent, action, depth))
            for a, ch in enumerate(nd.children):
                stack.append((ch, nd, a, depth + 1))

        # 2) Principal variation: from root, descend by max visit count.
        pv: set[int] = set()
        cur = root
        while cur is not None:
            pv.add(id(cur))
            if not cur.children:
                break
            nxt = max(cur.children, key=lambda c: c.visits)
            cur = nxt if nxt.visits > 0 else None

        # 3) Batched verifier value + epistemic uncertainty over ALL node latents.
        M = len(order)
        prm_vals = [float("nan")] * M
        sigmas = [float("nan")] * M
        verifier = getattr(mcts, "verifier", None)
        if verifier is not None:
            has_unc = hasattr(verifier, "value_with_uncertainty")
            Z = torch.cat([nd.latent() for (nd, _, _, _) in order], dim=0)  # [M, L, D]
            for s in range(0, M, chunk):
                zb = Z[s:s + chunk]
                xb = x[:1].expand(zb.shape[0], -1)
                if has_unc:
                    v, sd = verifier.value_with_uncertainty(xb, zb, width)
                    for j in range(zb.shape[0]):
                        prm_vals[s + j] = float(v[j]); sigmas[s + j] = float(sd[j])
                else:
                    v = verifier.value(xb, zb, width)
                    for j in range(zb.shape[0]):
                        prm_vals[s + j] = float(v[j])

        # 4) Write one JSON object per node.
        if self._mcts_fh is None:
            self._mcts_fh = open(self.run_dir / "mcts_graphs.jsonl", "a", encoding="utf-8")
        search_id = self._mcts_search_counter
        self._mcts_search_counter += 1
        for i, (nd, parent, action, depth) in enumerate(order):
            rec = {
                "run_id": self.run_id, "task_id": task_id, "search_id": search_id,
                "node_id": i,
                "parent_id": (node_id[id(parent)] if parent is not None else None),
                "action": action, "depth": depth,
                "visits": int(nd.visits), "value_sum": float(nd.value_sum),
                "q_value": float(nd.q),
                "prm_process_reward": prm_vals[i],
                "epistemic_sigma": sigmas[i],
                "prior": float(nd.prior),
                "is_leaf": not nd.children, "n_children": len(nd.children),
                "is_principal_variation": id(nd) in pv,
            }
            self._mcts_fh.write(json.dumps(rec) + "\n")
        self._mcts_fh.flush()
        return M

    # === shared CSV sink (streams 2 + 5) =================================== #
    def _csv_row(self, filename: str, row: dict) -> None:
        w = self._csv_writers.get(filename)
        if w is None:
            fh = open(self.run_dir / filename, "a", newline="", encoding="utf-8")
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()), extrasaction="ignore")
            if fh.tell() == 0:
                writer.writeheader()
            self._csv_writers[filename] = (fh, writer)
        else:
            fh, writer = w
        writer.writerow(row)
        fh.flush()

    # === (2) Hardware profiler -> hardware_telemetry.csv =================== #
    @contextmanager
    def profile_hardware(self, task_id: str, tag: str = "forward",
                         interval_s: float = 0.005) -> Iterator[None]:
        """Sample RAPL energy + RSS in a background thread for the wrapped block.

        Usage::

            with log.profile_hardware(task_id, tag="forward"):
                model(x, height=h, width=w)

        Writes one summary row (timestamps, energy J, peak RAM MB, latency ms,
        avg power W, sample count, rapl_available) to hardware_telemetry.csv.
        """
        mon = HardwareMonitor(interval_s)
        uj0 = read_rapl_microjoules()
        rss0 = read_rss_mb()
        ts_start = time.time()
        t0 = time.perf_counter()
        mon.start()
        try:
            yield
        finally:
            t1 = time.perf_counter()
            mon.stop()
            ts_end = time.time()
            uj1 = read_rapl_microjoules()
            latency_ms = (t1 - t0) * 1000.0
            energy_j = ((uj1 - uj0) / 1e6) if (uj0 is not None and uj1 is not None) else None
            rss_vals = [s[1] for s in mon.samples if not math.isnan(s[1])]
            if not math.isnan(rss0):
                rss_vals.append(rss0)
            peak_ram = max(rss_vals) if rss_vals else float("nan")
            avg_power = (energy_j / (latency_ms / 1000.0)) if (energy_j is not None and latency_ms > 0) else None
            self._csv_row("hardware_telemetry.csv", {
                "run_id": self.run_id, "task_id": task_id, "tag": tag,
                "ts_start_unix": ts_start, "ts_end_unix": ts_end,
                "latency_ms": latency_ms,
                "energy_joules": energy_j, "avg_power_watts": avg_power,
                "peak_ram_mb": peak_ram, "rss_start_mb": rss0,
                "n_samples": len(mon.samples), "rapl_available": rapl_available(),
            })

    # === (3) Latent tensor exporter -> latent_states.parquet ============== #
    @staticmethod
    def _pa():
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("latent Parquet export requires pyarrow: pip install pyarrow") from e
        return pa, pq

    @torch.no_grad()
    def export_latents(self, step_outputs, task_id: str, batch_index: int = 0) -> int:
        """Stream the per-step INT8 latent ``z`` + active-token mask to Parquet.

        ``step_outputs`` is the list returned by ``TRM.forward`` (each dict has
        ``z`` and, when a router is used, ``mask``). One row per (step, token):
        run_id, task_id, step, batch_index, token_index, active (routed vs frozen),
        z_scale (per-token dequant scale), z_int8 (the raw INT8 latent vector).
        Returns the number of token-rows written.
        """
        pa, pq = self._pa()
        cols: dict[str, list] = {k: [] for k in (
            "run_id", "task_id", "step", "batch_index", "token_index",
            "active", "z_scale", "z_int8")}
        for step_idx, out in enumerate(step_outputs):
            z = out["z"]
            if z.dim() == 2:
                z = z.unsqueeze(0)
            codes, scale = _quantize_int8(z)            # [B,L,D] int8, [B,L,1]
            zc = codes[batch_index]                     # [L, D]
            zs = scale[batch_index, :, 0]               # [L]
            mask = out.get("mask")
            if mask is not None:
                active = (mask[batch_index, :, 0] > 0.5).tolist()
            else:
                active = [True] * zc.shape[0]
            zc_list = zc.cpu().tolist()
            zs_list = zs.cpu().tolist()
            for t in range(zc.shape[0]):
                cols["run_id"].append(self.run_id)
                cols["task_id"].append(task_id)
                cols["step"].append(step_idx)
                cols["batch_index"].append(batch_index)
                cols["token_index"].append(t)
                cols["active"].append(bool(active[t]))
                cols["z_scale"].append(float(zs_list[t]))
                cols["z_int8"].append(zc_list[t])
        table = pa.table({
            "run_id": pa.array(cols["run_id"], pa.string()),
            "task_id": pa.array(cols["task_id"], pa.string()),
            "step": pa.array(cols["step"], pa.int32()),
            "batch_index": pa.array(cols["batch_index"], pa.int32()),
            "token_index": pa.array(cols["token_index"], pa.int32()),
            "active": pa.array(cols["active"], pa.bool_()),
            "z_scale": pa.array(cols["z_scale"], pa.float32()),
            "z_int8": pa.array(cols["z_int8"], pa.list_(pa.int8())),
        })
        if self._parquet_writer is None:
            self._parquet_path = self.run_dir / "latent_states.parquet"
            self._parquet_writer = pq.ParquetWriter(str(self._parquet_path), table.schema)
        self._parquet_writer.write_table(table)
        return table.num_rows

    def attach_latent_hook(self, model, task_id: str = "run"):
        """Register a forward hook on a ``TRM`` that auto-exports latents each call.

        Set the current task with ``set_task(task_id)`` before each ``forward``.
        Returns the hook handle (also removed on ``close()``).
        """
        self._current_task_id = task_id

        def hook(_module, _inputs, output):
            steps = output[1] if isinstance(output, (tuple, list)) and len(output) > 1 else None
            if steps:
                self.export_latents(steps, task_id=self._current_task_id)

        handle = model.register_forward_hook(hook)
        self._hooks.append(handle)
        return handle

    def set_task(self, task_id: str) -> None:
        self._current_task_id = task_id

    # === (5) Scaling-law aggregator -> spectra_scaling_laws.csv =========== #
    def log_scaling_row(self, task_id: str, model=None, model_params: int | None = None,
                        mcts_rollouts: int = 0, is_correct=None,
                        total_joules: float | None = None,
                        total_latency_ms: float | None = None, **extra: Any) -> None:
        """Append one macro per-task result row (the iso-joule scaling table)."""
        if model_params is None and model is not None:
            model_params = int(sum(p.numel() for p in model.parameters()))
        row = {
            "run_id": self.run_id, "task_id": task_id,
            "model_params": model_params, "mcts_rollouts": int(mcts_rollouts),
            "is_correct": (int(bool(is_correct)) if is_correct is not None else None),
            "total_joules": total_joules, "total_latency_ms": total_latency_ms,
        }
        row.update(extra)
        self._csv_row("spectra_scaling_laws.csv", row)

    # === lifecycle ========================================================= #
    def close(self) -> None:
        self.export_chrome_trace()
        self.metrics.close()
        self.policy_log.close()
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        if self._mcts_fh is not None:
            self._mcts_fh.close(); self._mcts_fh = None
        for fh, _ in self._csv_writers.values():
            fh.close()
        self._csv_writers.clear()
        if self._parquet_writer is not None:
            self._parquet_writer.close(); self._parquet_writer = None

    def __enter__(self) -> "SpectraTelemetryLogger":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
