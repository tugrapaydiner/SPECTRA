#!/usr/bin/env python3
"""Measure lazy-routing payoff on the checked native ternary kernel."""
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
from deploy.pack_ternary import pack_ternary_rows
from deploy.torch_kernel import cpu_supports_avx2

SRC = ROOT / "deploy" / "cpp_sparse_kernel" / "spectra_kernel.cpp"
I8 = ctypes.POINTER(ctypes.c_int8); U8 = ctypes.POINTER(ctypes.c_uint8); I32 = ctypes.POINTER(ctypes.c_int32); SZ = ctypes.c_size_t


def timeit(fn, target=0.30):
    fn(); reps = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(reps): fn()
        dt = time.perf_counter() - t0
        if dt >= target: return dt / reps
        reps = max(reps * 2, int(reps * target / max(dt, 1e-9)) + 1)


def main():
    build = Path(tempfile.mkdtemp(prefix="spectra_sparse_")); so = build / "lib.so"
    flags = ["g++", "-O3", "-std=c++17", "-shared", "-fPIC"]
    backend = "avx2" if cpu_supports_avx2() else "scalar"
    if backend == "avx2": flags.append("-mavx2")
    subprocess.run(flags + [str(SRC), "-o", str(so)], check=True)
    lib = ctypes.CDLL(str(so))
    lib.spectra_sparse_ternary_gemv.restype = ctypes.c_int
    lib.spectra_sparse_ternary_gemv.argtypes = [I8,SZ,I32,SZ,U8,SZ,I32,SZ,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,I8,SZ]

    num_tokens, hidden, out_dim, shift = 256, 512, 512, 15
    rng = np.random.default_rng(7)
    W = rng.integers(-1, 2, size=(out_dim, hidden), dtype=np.int8)
    packed, _ = pack_ternary_rows(W)
    X = np.ascontiguousarray(rng.integers(-128, 128, size=(num_tokens, hidden), dtype=np.int16).astype(np.int8))
    mult = np.full(out_dim, 1000, dtype=np.int32); Y = np.zeros((num_tokens, out_dim), dtype=np.int8)

    def call(idx):
        status = lib.spectra_sparse_ternary_gemv(
            X.ctypes.data_as(I8), X.size, idx.ctypes.data_as(I32), idx.size,
            packed.ctypes.data_as(U8), packed.size, mult.ctypes.data_as(I32), mult.size,
            shift, num_tokens, hidden, out_dim, Y.ctypes.data_as(I8), Y.size)
        if status != 0: raise RuntimeError(f"native sparse kernel status={status}")

    rows=[]; base_ms=None
    for d in [0.10,0.25,0.50,0.75,1.00]:
        na=max(1,int(round(d*num_tokens))); idx=np.ascontiguousarray(np.arange(na,dtype=np.int32))
        ms=timeit(lambda i=idx: call(i))*1e3
        if d==1.0: base_ms=ms
        rows.append({"density":d,"active_tokens":na,"num_tokens":num_tokens,"latency_ms":ms,"backend":backend})
        print(f"[sparse] backend={backend} density={d:.2f} active={na:<4} {ms:.3f} ms")
    for r in rows: r["speedup_vs_dense"] = base_ms/r["latency_ms"] if base_ms else float("nan")
    out=ROOT/"assets"/"data"; out.mkdir(parents=True,exist_ok=True)
    with open(out/"bench_sparse_density.csv","w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

if __name__ == "__main__": main()
