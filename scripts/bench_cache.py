#!/usr/bin/env python3
"""Empirical cache-residency proof for the ternary kernel (NO synthetic data).

We sweep the resident weight-matrix size across this CPU's real L2/L3 boundaries
and measure throughput. While the packed weights fit in cache, re-reading them is
fast; once they exceed L3 the kernel falls off a DRAM cliff. This is a direct
hardware demonstration (timing only, no modeling) that the SPECTRA core is small
enough to live in fast cache. Writes assets/data/bench_cache_residency.csv.
"""
from __future__ import annotations

import csv
import ctypes
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from deploy.pack_ternary import pack_ternary, packed_size_bytes

SRC = ROOT / "deploy" / "cpp_sparse_kernel" / "spectra_kernel.cpp"
DATA = ROOT / "assets" / "data"
I8 = ctypes.POINTER(ctypes.c_int8); U8 = ctypes.POINTER(ctypes.c_uint8); I32 = ctypes.POINTER(ctypes.c_int32)
NA, HIDDEN, SHIFT = 8, 1024, 15          # re-read the matrix NA times per call


def host_caches():
    base = Path("/sys/devices/system/cpu/cpu0/cache"); out = {}
    try:
        for idx in base.glob("index*"):
            lvl = int((idx / "level").read_text()); typ = (idx / "type").read_text().strip()
            s = (idx / "size").read_text().strip()
            kb = float(s.rstrip("K")) if s.endswith("K") else float(s.rstrip("M")) * 1024
            if lvl in (2, 3) and typ in ("Unified", "Data"):
                out[lvl] = kb / 1024.0
    except OSError:
        pass
    return out.get(2), out.get(3)


def timeit(fn, target=0.15):
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
    build = Path(tempfile.mkdtemp(prefix="spectra_cache_"))
    so = build / "lib.so"
    subprocess.run(["g++", "-O3", "-mavx2", "-std=c++17", "-shared", "-fPIC",
                    str(SRC), "-o", str(so)], check=True)
    lib = ctypes.CDLL(str(so))
    lib.spectra_sparse_ternary_gemv.restype = None
    lib.spectra_sparse_ternary_gemv.argtypes = [I8, I32, ctypes.c_int, U8, I32, ctypes.c_int,
                                                ctypes.c_int, ctypes.c_int, I8]
    rng = np.random.default_rng(11)
    idx = np.ascontiguousarray(np.arange(NA, dtype=np.int32))
    X = np.ascontiguousarray(rng.integers(-127, 128, size=(NA, HIDDEN)).astype(np.int8))

    out_dims = [128, 512, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    rows = []
    for od in out_dims:
        W = rng.integers(-1, 2, size=(od, HIDDEN)).astype(np.int8)
        packed = np.ascontiguousarray(pack_ternary(W)[0])     # whole matrix, vectorized
        mult = np.full(od, 1000, dtype=np.int32)
        Y = np.zeros((NA, od), dtype=np.int8)
        t = timeit(lambda: lib.spectra_sparse_ternary_gemv(
            X.ctypes.data_as(I8), idx.ctypes.data_as(I32), NA,
            packed.ctypes.data_as(U8), mult.ctypes.data_as(I32), SHIFT,
            HIDDEN, od, Y.ctypes.data_as(I8)))
        wbytes_mb = packed_size_bytes(od * HIDDEN) / (1024 * 1024)
        gops = 2 * NA * od * HIDDEN / t / 1e9
        rows.append({"out_dim": od, "weight_mb": wbytes_mb, "gops": gops, "latency_ms": t * 1e3})
        print(f"[cache] W={wbytes_mb:8.3f} MB  {gops:7.2f} GOP/s  {t*1e3:8.3f} ms")
        del W, packed, Y

    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / "bench_cache_residency.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # stash real cache sizes for the renderer
    l2, l3 = host_caches()
    hp = DATA / "host.json"
    host = json.loads(hp.read_text()) if hp.exists() else {}
    host.update({"l2_mb": l2, "l3_mb": l3, "packed_core_mb": packed_size_bytes(5_600_000) / (1024 * 1024)})
    hp.write_text(json.dumps(host, indent=2))
    print(f"[done] L2={l2} MB L3={l3} MB -> {DATA/'bench_cache_residency.csv'}")


if __name__ == "__main__":
    main()
