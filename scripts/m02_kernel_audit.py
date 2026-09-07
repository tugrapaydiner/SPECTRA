#!/usr/bin/env python3
"""Milestone 02 before/after native arithmetic audit.

Compiles the pre-M02 and current kernel source both scalar and AVX2, reproduces
the INT8_MIN x -1 mismatch in the old AVX2 path, verifies the corrected path,
and records a small dot-kernel performance comparison. This is intentionally a
microbenchmark, not a publication performance claim.
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

I8 = C.POINTER(C.c_int8)
U8 = C.POINTER(C.c_uint8)

WRAPPER = r'''
#include <cstdint>
#include <cstddef>
#include "SOURCE_PATH"
extern "C" int32_t m02_dot_packed(const int8_t* x, const uint8_t* w, int h) {
  return ternary_dot(x, w, h);
}
extern "C" int32_t m02_dot_decoded(const int8_t* x, const int8_t* w, int h) {
  return ternary_dot_decoded(x, w, h);
}
extern "C" int8_t m02_requant(int32_t acc, int32_t mult, int shift) {
  return requantize_i32(acc, mult, shift);
}
extern "C" int64_t m02_bench_packed(
    const int8_t* X, int rows, const uint8_t* w, int h, int reps) {
  volatile int64_t sum = 0;
  for (int i = 0; i < reps; ++i) sum += ternary_dot(X + (size_t)(i % rows) * h, w, h);
  return sum;
}
extern "C" int64_t m02_bench_decoded(
    const int8_t* X, int rows, const int8_t* w, int h, int reps) {
  volatile int64_t sum = 0;
  for (int i = 0; i < reps; ++i) sum += ternary_dot_decoded(X + (size_t)(i % rows) * h, w, h);
  return sum;
}
'''


def pack_row(w: np.ndarray) -> np.ndarray:
    codes = np.zeros(w.size, dtype=np.uint8)
    codes[w == 1] = 1
    codes[w == -1] = 2
    out = np.zeros((w.size + 3) // 4, dtype=np.uint8)
    for i, c in enumerate(codes):
        out[i // 4] |= int(c) << (2 * (i % 4))
    return out


def compile_variant(source: Path, avx2: bool, build_dir: Path, tag: str):
    wrapper = build_dir / f"wrapper_{tag}_{'avx2' if avx2 else 'scalar'}.cpp"
    wrapper.write_text(WRAPPER.replace("SOURCE_PATH", str(source.resolve())), encoding="utf-8")
    so = build_dir / f"{tag}_{'avx2' if avx2 else 'scalar'}.so"
    cmd = ["g++", "-O3", "-std=c++17", "-shared", "-fPIC"]
    if avx2:
        cmd.append("-mavx2")
    full_cmd = cmd + [str(wrapper), "-o", str(so)]
    proc = subprocess.run(full_cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        print("[m02-audit] compiler command failed:", " ".join(full_cmd))
        if proc.stdout:
            print("[m02-audit] compiler stdout:\n" + proc.stdout)
        if proc.stderr:
            print("[m02-audit] compiler stderr:\n" + proc.stderr)
        raise subprocess.CalledProcessError(
            proc.returncode, full_cmd, output=proc.stdout, stderr=proc.stderr
        )
    lib = C.CDLL(str(so))
    lib.m02_dot_packed.argtypes = [I8, U8, C.c_int]
    lib.m02_dot_packed.restype = C.c_int32
    lib.m02_dot_decoded.argtypes = [I8, I8, C.c_int]
    lib.m02_dot_decoded.restype = C.c_int32
    lib.m02_requant.argtypes = [C.c_int32, C.c_int32, C.c_int]
    lib.m02_requant.restype = C.c_int8
    lib.m02_bench_packed.argtypes = [I8, C.c_int, U8, C.c_int, C.c_int]
    lib.m02_bench_packed.restype = C.c_int64
    lib.m02_bench_decoded.argtypes = [I8, C.c_int, I8, C.c_int, C.c_int]
    lib.m02_bench_decoded.restype = C.c_int64
    return lib


def case(lib):
    x = np.zeros(32, dtype=np.int8); x[0] = -128
    w = np.zeros(32, dtype=np.int8); w[0] = -1
    packed = pack_row(w)
    acc_p = int(lib.m02_dot_packed(x.ctypes.data_as(I8), packed.ctypes.data_as(U8), 32))
    acc_d = int(lib.m02_dot_decoded(x.ctypes.data_as(I8), w.ctypes.data_as(I8), 32))
    return {
        "packed_acc": acc_p,
        "decoded_acc": acc_d,
        "packed_requant": int(lib.m02_requant(acc_p, 1, 7)),
        "decoded_requant": int(lib.m02_requant(acc_d, 1, 7)),
    }


def bench(lib, decoded: bool, X, packed, w, h: int, reps: int, rounds: int = 5):
    fn = lib.m02_bench_decoded if decoded else lib.m02_bench_packed
    ptr = w.ctypes.data_as(I8) if decoded else packed.ctypes.data_as(U8)
    vals = []
    checksum = None
    for _ in range(rounds):
        t0 = time.perf_counter_ns()
        checksum = int(fn(X.ctypes.data_as(I8), X.shape[0], ptr, h, reps))
        vals.append((time.perf_counter_ns() - t0) / reps)
    return statistics.median(vals), checksum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    build_dir = Path(tempfile.mkdtemp(prefix="spectra_m02_audit_"))
    before_s = compile_variant(Path(args.before), False, build_dir, "before")
    before_v = compile_variant(Path(args.before), True, build_dir, "before")
    after_s = compile_variant(Path(args.after), False, build_dir, "after")
    after_v = compile_variant(Path(args.after), True, build_dir, "after")

    report = {
        "reported_case": {
            "expected": {"acc": 128, "requant_mult": 1, "shift": 7, "requant": 1},
            "before_scalar": case(before_s),
            "before_avx2": case(before_v),
            "after_scalar": case(after_s),
            "after_avx2": case(after_v),
        }
    }
    assert report["reported_case"]["before_scalar"]["packed_requant"] == 1
    assert report["reported_case"]["before_avx2"]["packed_requant"] == -1
    assert report["reported_case"]["before_avx2"]["decoded_requant"] == -1
    assert report["reported_case"]["after_scalar"]["packed_requant"] == 1
    assert report["reported_case"]["after_avx2"]["packed_requant"] == 1
    assert report["reported_case"]["after_avx2"]["decoded_requant"] == 1

    rng = np.random.default_rng(7)
    h, rows, reps = 512, 32, 20000
    X = rng.integers(-128, 128, size=(rows, h), dtype=np.int16).astype(np.int8)
    w = rng.integers(-1, 2, size=h, dtype=np.int8)
    packed = pack_row(w)
    perf = {}
    for decoded in (False, True):
        name = "decoded" if decoded else "packed"
        b, cb = bench(before_v, decoded, X, packed, w, h, reps)
        a, ca = bench(after_v, decoded, X, packed, w, h, reps)
        perf[name] = {
            "before_ns_per_dot": b,
            "after_ns_per_dot": a,
            "after_over_before": a / b,
            "before_checksum": cb,
            "after_checksum": ca,
        }
    report["microbenchmark"] = {"hidden": h, "rows": rows, "reps": reps, **perf}
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
