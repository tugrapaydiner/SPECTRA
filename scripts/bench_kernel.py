#!/usr/bin/env python3
"""Real microbenchmark of the SPECTRA AVX2 ternary GEMV kernel (NO synthetic data).

Compiles the production kernel deploy/cpp_sparse_kernel/spectra_kernel.cpp twice --
with -mavx2 (SIMD path) and scalar -- loads both via ctypes, verifies the AVX2
output is BIT-EXACT to the scalar reference, then measures, on this host's CPU:

  * weight-stationary recursion: effective throughput (GOP/s of int8 MACs) and
    arithmetic intensity as a function of the recursion reuse factor K. This is the
    memory-bound -> compute-bound transition the whole design rests on.
  * SIMD scaling: AVX2 vs scalar throughput and speedup vs hidden width.

Outputs measured CSVs to <out>/.  Reproduce on the target device:
    python scripts/bench_kernel.py --out assets/data
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
sys.path.insert(0, str(ROOT))
from deploy.pack_ternary import packed_size_bytes, pack_ternary  # real packer
from eval.roofline import arithmetic_intensity                   # exact AI math

SRC = ROOT / "deploy" / "cpp_sparse_kernel" / "spectra_kernel.cpp"
SHIFT = 15
I8 = ctypes.POINTER(ctypes.c_int8)
U8 = ctypes.POINTER(ctypes.c_uint8)
I32 = ctypes.POINTER(ctypes.c_int32)


def compile_kernel(avx2: bool, builddir: Path) -> Path:
    out = builddir / f"libspectra_{'avx2' if avx2 else 'scalar'}.so"
    flags = ["g++", "-O3", "-std=c++17", "-shared", "-fPIC"]
    if avx2:
        flags.append("-mavx2")
    subprocess.run(flags + [str(SRC), "-o", str(out)], check=True)
    return out


def load(lib_path: Path):
    lib = ctypes.CDLL(str(lib_path))
    lib.spectra_weight_stationary_gemv.restype = None
    lib.spectra_weight_stationary_gemv.argtypes = [
        I8, U8, I32, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, I8]
    return lib


def make_inputs(out_dim: int, hidden: int, K: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    W = rng.integers(-1, 2, size=(out_dim, hidden)).astype(np.int8)
    packed = np.concatenate([pack_ternary(W[o])[0] for o in range(out_dim)]).astype(np.uint8)
    X = rng.integers(-127, 128, size=(K, hidden)).astype(np.int8)
    mult = np.full(out_dim, 1000, dtype=np.int32)
    Y = np.zeros((K, out_dim), dtype=np.int8)
    return np.ascontiguousarray(packed), np.ascontiguousarray(X), mult, Y


def call_ws(lib, packed, X, mult, K, hidden, out_dim, Y):
    lib.spectra_weight_stationary_gemv(
        X.ctypes.data_as(I8), packed.ctypes.data_as(U8), mult.ctypes.data_as(I32),
        SHIFT, K, hidden, out_dim, Y.ctypes.data_as(I8))


def timeit(fn, target=0.30):
    fn()  # warmup
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/data")
    args = ap.parse_args()
    out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    build = Path(tempfile.mkdtemp(prefix="spectra_kbuild_"))
    avx2 = load(compile_kernel(True, build))
    scalar = load(compile_kernel(False, build))

    # --- correctness gate: AVX2 must equal the scalar oracle, bit-for-bit ----
    packed, X, mult, Yv = make_inputs(256, 256, 8, seed=1)
    _, _, _, Ys = make_inputs(256, 256, 8, seed=1)
    call_ws(avx2, packed, X, mult, 8, 256, 256, Yv)
    call_ws(scalar, packed, X, mult, 8, 256, 256, Ys)
    bit_exact = bool(np.array_equal(Yv, Ys))
    print(f"[correctness] AVX2 == scalar (bit-exact): {bit_exact}")
    if not bit_exact:
        raise SystemExit("AVX2 path diverged from scalar reference -- aborting bench")

    host = {
        "cpu": platform.processor() or platform.machine(),
        "uname": platform.platform(),
        "bit_exact": bit_exact,
    }
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                host["cpu"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass

    # --- (A) weight-stationary: throughput + AI vs recursion reuse K --------
    out_dim, hidden = 512, 512
    Ks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    rows_ws = []
    peak = 0.0
    for K in Ks:
        packed, X, mult, Y = make_inputs(out_dim, hidden, K, seed=2)
        for name, lib in (("avx2", avx2), ("scalar", scalar)):
            t = timeit(lambda l=lib, p=packed, x=X, m=mult, k=K, Y=Y:
                       call_ws(l, p, x, m, k, hidden, out_dim, Y))
            macs = K * out_dim * hidden
            gops = 2 * macs / t / 1e9                      # int8 MAC -> 2 ops
            ai = arithmetic_intensity(out_dim, hidden, weight_bits=2, reuse=K)
            rows_ws.append({"K": K, "build": name, "hidden": hidden, "out_dim": out_dim,
                            "latency_ms": t * 1e3, "gops": gops, "arithmetic_intensity": ai})
            if name == "avx2":
                peak = max(peak, gops)
            print(f"[ws] K={K:<4} {name:<6} {gops:8.2f} GOP/s  AI={ai:8.1f}  {t*1e3:8.3f} ms")

    # --- (B) SIMD scaling: AVX2 vs scalar GOP/s + speedup vs hidden width ----
    Kfix, out_dim = 64, 512
    hiddens = [64, 128, 256, 512, 1024, 2048]
    rows_simd = []
    for hidden in hiddens:
        packed, X, mult, Y = make_inputs(out_dim, hidden, Kfix, seed=3)
        g = {}
        for name, lib in (("avx2", avx2), ("scalar", scalar)):
            t = timeit(lambda l=lib, p=packed, x=X, m=mult, h=hidden, Y=Y:
                       call_ws(l, p, x, m, Kfix, h, out_dim, Y))
            g[name] = 2 * Kfix * out_dim * hidden / t / 1e9
        speedup = g["avx2"] / g["scalar"] if g["scalar"] else float("nan")
        rows_simd.append({"hidden": hidden, "gops_avx2": g["avx2"], "gops_scalar": g["scalar"],
                          "speedup": speedup})
        print(f"[simd] hidden={hidden:<5} avx2={g['avx2']:7.2f}  scalar={g['scalar']:7.2f}  x{speedup:5.2f}")

    host["peak_avx2_gops"] = peak
    (out / "host.json").write_text(json.dumps(host, indent=2))
    with open(out / "bench_weight_stationary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_ws[0].keys())); w.writeheader(); w.writerows(rows_ws)
    with open(out / "bench_simd_scaling.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_simd[0].keys())); w.writeheader(); w.writerows(rows_simd)
    print(f"\n[done] host={host['cpu']!r} peak_avx2={peak:.1f} GOP/s -> wrote CSVs to {out}")


if __name__ == "__main__":
    main()
