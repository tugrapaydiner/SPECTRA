#!/usr/bin/env python3
"""Measure the lazy-routing payoff on the real sparse ternary kernel (no fake data).

spectra_sparse_ternary_gemv only computes the *active* tokens (the RL router freezes
the rest). We sweep active-token density and measure latency -> the compute saving
is linear in the frozen fraction. Writes assets/data/bench_sparse_density.csv.
"""
from __future__ import annotations

import csv
import ctypes
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from deploy.pack_ternary import pack_ternary

SRC = ROOT / "deploy" / "cpp_sparse_kernel" / "spectra_kernel.cpp"
I8 = ctypes.POINTER(ctypes.c_int8); U8 = ctypes.POINTER(ctypes.c_uint8); I32 = ctypes.POINTER(ctypes.c_int32)


def timeit(fn, target=0.30):
    fn()
    reps = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        dt = time.perf_counter() - t0
        if dt >= target:
            return dt / reps
        reps = max(reps * 2, int(reps * target / max(dt, 1e-9)) + 1)


def main():
    build = Path(tempfile.mkdtemp(prefix="spectra_sparse_"))
    so = build / "lib.so"
    subprocess.run(["g++", "-O3", "-mavx2", "-std=c++17", "-shared", "-fPIC",
                    str(SRC), "-o", str(so)], check=True)
    lib = ctypes.CDLL(str(so))
    lib.spectra_sparse_ternary_gemv.restype = None
    lib.spectra_sparse_ternary_gemv.argtypes = [I8, I32, ctypes.c_int, U8, I32, ctypes.c_int,
                                                ctypes.c_int, ctypes.c_int, I8]

    num_tokens, hidden, out_dim, shift = 256, 512, 512, 15
    rng = np.random.default_rng(7)
    W = rng.integers(-1, 2, size=(out_dim, hidden)).astype(np.int8)
    packed = np.ascontiguousarray(np.concatenate([pack_ternary(W[o])[0] for o in range(out_dim)]).astype(np.uint8))
    X = np.ascontiguousarray(rng.integers(-127, 128, size=(num_tokens, hidden)).astype(np.int8))
    mult = np.full(out_dim, 1000, dtype=np.int32)
    Y = np.zeros((num_tokens, out_dim), dtype=np.int8)

    densities = [0.10, 0.25, 0.50, 0.75, 1.00]
    rows = []
    base_ms = None
    for d in densities:
        na = max(1, int(round(d * num_tokens)))
        idx = np.ascontiguousarray(np.arange(na, dtype=np.int32))
        t = timeit(lambda: lib.spectra_sparse_ternary_gemv(
            X.ctypes.data_as(I8), idx.ctypes.data_as(I32), na,
            packed.ctypes.data_as(U8), mult.ctypes.data_as(I32), shift,
            hidden, out_dim, Y.ctypes.data_as(I8)))
        ms = t * 1e3
        if d == 1.00:
            base_ms = ms
        rows.append({"density": d, "active_tokens": na, "num_tokens": num_tokens, "latency_ms": ms})
        print(f"[sparse] density={d:.2f} active={na:<4} {ms:.3f} ms")
    for r in rows:
        r["speedup_vs_dense"] = base_ms / r["latency_ms"] if base_ms else float("nan")

    out = ROOT / "assets" / "data"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "bench_sparse_density.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"[done] wrote {out/'bench_sparse_density.csv'}")


if __name__ == "__main__":
    main()
