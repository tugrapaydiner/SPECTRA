#!/usr/bin/env python3
"""Microbenchmark the checked native kernel with an auditable M13 convention.

The K-input weight-stationary call consumes a *precomputed* [K,H] matrix in one
invocation. It demonstrates reuse inside that kernel call; it does not establish
sequential recurrent cache residency. Throughput and arithmetic intensity both use
1 MAC = 2 arithmetic operations.
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

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT))
from deploy.pack_ternary import pack_ternary_rows
from deploy.torch_kernel import cpu_supports_avx2
from eval.roofline import arithmetic_intensity, precomputed_input_reuse_traffic

SRC = ROOT / "deploy" / "cpp_sparse_kernel" / "spectra_kernel.cpp"
SHIFT = 15
I8 = ctypes.POINTER(ctypes.c_int8); U8 = ctypes.POINTER(ctypes.c_uint8)
I32 = ctypes.POINTER(ctypes.c_int32); SZ = ctypes.c_size_t
BASE_FLAGS = ["g++", "-O3", "-std=c++17", "-shared", "-fPIC"]


def compile_kernel(avx2: bool, builddir: Path) -> tuple[Path, list[str]]:
    out = builddir / f"libspectra_{'avx2' if avx2 else 'scalar'}.so"
    flags = list(BASE_FLAGS) + (["-mavx2"] if avx2 else [])
    subprocess.run(flags + [str(SRC), "-o", str(out)], check=True)
    return out, flags


def load(path: Path):
    lib = ctypes.CDLL(str(path)); lib.spectra_weight_stationary_gemv.restype = ctypes.c_int
    lib.spectra_weight_stationary_gemv.argtypes = [I8,SZ,U8,SZ,I32,SZ,ctypes.c_int,
        ctypes.c_int,ctypes.c_int,ctypes.c_int,I8,SZ]
    return lib


def make_inputs(out_dim: int, hidden: int, K: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    W = rng.integers(-1, 2, size=(out_dim, hidden), dtype=np.int8)
    packed, _ = pack_ternary_rows(W)
    X = rng.integers(-128, 128, size=(K, hidden), dtype=np.int16).astype(np.int8)
    mult = np.full(out_dim, 1000, dtype=np.int32)
    Y = np.zeros((K, out_dim), dtype=np.int8)
    return np.ascontiguousarray(packed), np.ascontiguousarray(X), mult, Y


def call_ws(lib, packed, X, mult, K, hidden, out_dim, Y):
    status = lib.spectra_weight_stationary_gemv(
        X.ctypes.data_as(I8), X.size, packed.ctypes.data_as(U8), packed.size,
        mult.ctypes.data_as(I32), mult.size, SHIFT, K, hidden, out_dim,
        Y.ctypes.data_as(I8), Y.size)
    if status != 0:
        raise RuntimeError(f"native weight-stationary status={status}")


def timeit(fn, target: float = 0.30) -> tuple[float, int]:
    fn(); reps = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(reps): fn()
        dt = time.perf_counter() - t0
        if dt >= target: return dt / reps, reps
        reps = max(reps * 2, int(reps * target / max(dt, 1e-9)) + 1)


def _cpu_model() -> str:
    value = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/data")
    ap.add_argument("--target-seconds", type=float, default=0.30)
    args = ap.parse_args()
    if args.target_seconds <= 0: raise ValueError("--target-seconds must be positive")
    out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not cpu_supports_avx2():
        raise SystemExit("AVX2 comparison requested but CPU does not advertise AVX2")

    build = Path(tempfile.mkdtemp(prefix="spectra_kbuild_"))
    avx_path, avx_flags = compile_kernel(True, build); scalar_path, scalar_flags = compile_kernel(False, build)
    avx2, scalar = load(avx_path), load(scalar_path)

    packed, X, mult, Yv = make_inputs(256, 257, 8, seed=1)
    _, _, _, Ys = make_inputs(256, 257, 8, seed=1)
    call_ws(avx2, packed, X, mult, 8, 257, 256, Yv); call_ws(scalar, packed, X, mult, 8, 257, 256, Ys)
    bit_exact = bool(np.array_equal(Yv, Ys))
    if not bit_exact: raise SystemExit("AVX2 diverged from scalar")

    host = {
        "cpu": _cpu_model(), "uname": platform.platform(), "bit_exact": bit_exact,
        "operation_convention": "1_MAC_equals_2_arithmetic_operations",
        "workload_label": "precomputed_input_reuse",
        "traffic_scope": "kernel_external_first_touch_logical_bytes_not_measured_dram",
        "cache_residency_established": False, "bandwidth_bottleneck_established": False,
        "compiler": "g++", "compile_flags_avx2": avx_flags, "compile_flags_scalar": scalar_flags,
    }

    out_dim = hidden = 512; rows_ws: list[dict] = []; peak = 0.0
    for K in [1,2,4,8,16,32,64,128,256]:
        packed, X, mult, Y = make_inputs(out_dim, hidden, K, seed=2)
        traffic = precomputed_input_reuse_traffic(out_dim, hidden, weight_bits=2, reuse=K)
        for name, lib in (("avx2", avx2), ("scalar", scalar)):
            t, reps = timeit(lambda l=lib,p=packed,x=X,m=mult,k=K,Y=Y:
                             call_ws(l,p,x,m,k,hidden,out_dim,Y), args.target_seconds)
            macs = K * out_dim * hidden; operations = 2 * macs; gops = operations / t / 1e9
            ai = arithmetic_intensity(out_dim, hidden, weight_bits=2, reuse=K)
            row = {
                "workload": "precomputed_input_reuse", "K": K, "build": name,
                "hidden": hidden, "out_dim": out_dim, "latency_ms": t * 1e3,
                "timing_repetitions": reps, "macs": macs, "operations": operations,
                "gops": gops, "arithmetic_intensity_ops_per_byte": ai,
                "packed_weight_bytes": traffic.packed_weight_bytes,
                "input_activation_bytes": traffic.input_activation_bytes,
                "output_activation_bytes": traffic.output_activation_bytes,
                "requant_parameter_bytes": traffic.requant_parameter_bytes,
                "external_first_touch_logical_bytes": traffic.total_external_first_touch_bytes,
                "decoded_row_scratch_bytes": traffic.decoded_row_scratch_bytes,
                "traffic_scope": traffic.traffic_scope,
                "cache_residency_established": False, "bandwidth_bottleneck_established": False,
            }
            rows_ws.append(row)
            if name == "avx2": peak = max(peak, gops)
            print(f"[precomputed K] K={K:<4} {name:<6} {gops:8.2f} GOP/s AI={ai:8.2f} op/B")

    Kfix, out_dim = 64, 512; rows_simd: list[dict] = []
    for hidden in [64,128,256,512,1024,2048]:
        packed, X, mult, Y = make_inputs(out_dim, hidden, Kfix, seed=3); g = {}
        for name, lib in (("avx2", avx2), ("scalar", scalar)):
            t, reps = timeit(lambda l=lib,p=packed,x=X,m=mult,h=hidden,Y=Y:
                             call_ws(l,p,x,m,Kfix,h,out_dim,Y), args.target_seconds)
            macs = Kfix * out_dim * hidden; operations = 2 * macs
            g[name] = operations / t / 1e9
        speed = g["avx2"] / g["scalar"]
        rows_simd.append({"workload": "precomputed_input_reuse", "K": Kfix,
            "hidden": hidden, "gops_avx2": g["avx2"], "gops_scalar": g["scalar"],
            "speedup": speed, "operation_convention": "1_MAC_equals_2_arithmetic_operations"})

    host["peak_avx2_gops_precomputed_input_reuse"] = peak
    (out / "host.json").write_text(json.dumps(host, indent=2, sort_keys=True) + "\n")
    with open(out / "bench_weight_stationary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_ws[0].keys())); w.writeheader(); w.writerows(rows_ws)
    with open(out / "bench_simd_scaling.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_simd[0].keys())); w.writeheader(); w.writerows(rows_simd)


if __name__ == "__main__": main()
