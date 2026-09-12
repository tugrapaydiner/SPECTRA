"""Auditable six-arm CPU runtime comparison; no learned-quality superiority claim."""
from __future__ import annotations

import argparse
from collections import defaultdict
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from data import sudoku
from deploy import m10_native
from deploy.m10_artifact import load_cpu_artifact
from deploy.m10_runtime import CPURecursiveRuntime
from deploy.validated_runtime import ValidatedCPURecursiveRuntime
from spectra.blocked_runtime import BlockedCPURecursiveRuntime, load_extension
from spectra.inference import predict_final

ARMS = ("checked_trace", "checked_final", "validated_trace", "validated_final", "blocked_trace", "blocked_final")
CLASSES = {"checked": CPURecursiveRuntime, "validated": ValidatedCPURecursiveRuntime, "blocked": BlockedCPURecursiveRuntime}
CONFIG = {"rounds": 7, "warmups": 2, "arm_order_seed": 916028,
          "memory_cases": ["untrained-d64-s16:b4:i0", "trained-d64-s4:b1:i0"],
          "scope": "warm inference plus independent validity check; not time per successful solve"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")


def read(path):
    return json.loads(Path(path).read_text())


def setup_cpu():
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    cpu_model = platform.processor() or "unavailable"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        cpu_model = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                          if line.startswith("model name") and ":" in line), cpu_model)
    compiler_command = os.environ.get("CXX", "c++")
    try:
        compiler = subprocess.run([compiler_command, "--version"], capture_output=True,
                                  text=True, timeout=10).stdout.splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        compiler = "unavailable"
    return {"python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
            "cpu_model": cpu_model, "compiler_command": compiler_command, "compiler": compiler,
            "python_compiler": platform.python_compiler(),
            "platform": platform.platform(), "cpu_threads": torch.get_num_threads(),
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "cuda_used": False, "energy_joules": None}


def source_files():
    paths = [ROOT / "pyproject.toml"]
    for directory in ["spectra", "deploy", "model", "data", "train", "eval", "common", "scripts"]:
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.suffix in {".py", ".cpp", ".h"})
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def load_fixtures(directory):
    manifest = read(directory / "SHA256.json")
    for name, expected in manifest.items():
        path = directory / name
        if path.resolve().parent != directory.resolve() or sha(path) != expected:
            raise ValueError("fixture digest/path mismatch: " + name)
    fixtures = read(directory / "fixtures.json")
    if fixtures["schema"] != "spectra.runtime_fixtures.v1": raise ValueError("unknown fixture schema")
    cases = {}
    for fixture in fixtures["fixtures"]:
        if sha(directory / fixture["artifact"]) != fixture["sha256"]: raise ValueError("artifact digest mismatch")
        for item in fixtures["inputs"]:
            if sha(directory / item["path"]) != item["sha256"]: raise ValueError("input digest mismatch")
            name = f"{fixture['name']}:b{item['batch']}:i{item['index']}"
            if name in cases: raise ValueError("duplicate fixture case")
            cases[name] = {"fixture": fixture, "input": item}
    return cases


def retained_bytes(value):
    seen = {}
    def visit(obj):
        if isinstance(obj, torch.Tensor):
            storage = obj.untyped_storage()
            seen[(str(obj.device), storage.data_ptr())] = storage.nbytes()
        elif dataclasses.is_dataclass(obj):
            for field in dataclasses.fields(obj): visit(getattr(obj, field.name))
        elif isinstance(obj, dict):
            for item in obj.values(): visit(item)
        elif isinstance(obj, (tuple, list)):
            for item in obj: visit(item)
    visit(value)
    return sum(seen.values())


def prediction(runtime, mode, x):
    return runtime.forward(x) if mode.endswith("_trace") else predict_final(runtime, x)


def output_key(value, mode):
    halt = value.step_outputs[-1]["halt_logit"] if mode.endswith("_trace") else value.halt_logit
    return {"logits": digest(value.logits), "answer": digest(value.answer), "halt": digest(halt)}


def validity(x, answer):
    xx, yy = x.numpy().reshape(-1, 4, 4), answer.numpy().reshape(-1, 4, 4)
    return [bool(sudoku.is_solved(y, 2) and sudoku.respects_clues(p, y, 2)) for p, y in zip(xx, yy, strict=True)]


def case_input(directory, case):
    return torch.from_numpy(np.load(directory / case["input"]["path"], allow_pickle=False)).long()


def analyze(rows, cases, rounds):
    expected = {(case, arm, i) for case in cases for arm in ARMS for i in range(rounds)}
    seen = set(); grouped = defaultdict(list); keys = {}; valid = {}
    for row in rows:
        key = (row["case"], row["arm"], row["round"])
        if key not in expected or key in seen: raise ValueError("duplicate or unexpected timing cell")
        if type(row["round"]) is not int: raise ValueError("invalid round")
        seen.add(key)
        for field in ("api_ns", "complete_ns"):
            if type(row[field]) is not int or row[field] <= 0: raise ValueError("invalid timing")
        if row["complete_ns"] < row["api_ns"]: raise ValueError("check time missing")
        name = row["case"]
        if name in keys and keys[name] != row["outputs"]: raise ValueError("nonidentical final output bits")
        if name in valid and valid[name] != row["valid"]: raise ValueError("validity differs")
        keys[name], valid[name] = row["outputs"], row["valid"]
        depth = cases[name]["fixture"]["depth"]
        count = 13 * depth if row["arm"].endswith("trace") else 12 * depth + 1
        if row["native_calls"] != count: raise ValueError("wrong native work count")
        if row["halt_calls"] != (depth if row["arm"].endswith("trace") else 1): raise ValueError("wrong head count")
        grouped[(name, row["arm"])].append(row)
    if seen != expected: raise ValueError("missing timing cells")
    cells = {}
    for name in cases:
        arms = {}
        for arm in ARMS:
            rr = grouped[(name, arm)]
            times = [r["complete_ns"] for r in rr]
            arms[arm] = {"median_complete_ns": statistics.median(times),
                         "p95_complete_ns": float(np.quantile(times, .95)),
                         "median_api_ns": statistics.median(r["api_ns"] for r in rr),
                         "returned_tensor_bytes": rr[0]["returned_tensor_bytes"]}
            if len({r["returned_tensor_bytes"] for r in rr}) != 1: raise ValueError("inconsistent returned storage")
        fastest = min(ARMS[:4], key=lambda a: arms[a]["median_complete_ns"])
        cells[name] = {"arms": arms, "valid": valid[name], "trained": cases[name]["fixture"]["trained"],
                       "fastest_existing_arm": fastest,
                       "blocked_final_over_fastest_existing": arms["blocked_final"]["median_complete_ns"] / arms[fastest]["median_complete_ns"],
                       "blocked_final_over_validated_final": arms["blocked_final"]["median_complete_ns"] / arms["validated_final"]["median_complete_ns"]}
    totals = {arm: sum(cell["arms"][arm]["median_complete_ns"] for cell in cells.values()) for arm in ARMS}
    fastest_total = sum(cell["arms"][cell["fastest_existing_arm"]]["median_complete_ns"] for cell in cells.values())
    return {"schema": "spectra.integrated_runtime.v1", "calls": len(rows), "cases": len(cases),
            "artifacts": len({c["fixture"]["name"] for c in cases.values()}),
            "exact_final_bits": True, "cells": cells,
            "ratio_of_summed_case_medians": {
                "blocked_final_over_validated_final": totals["blocked_final"] / totals["validated_final"],
                "blocked_final_over_checked_trace": totals["blocked_final"] / totals["checked_trace"],
                "blocked_final_over_fastest_existing_per_case": totals["blocked_final"] / fastest_total},
            "regressed_case_count_vs_fastest_existing": sum(c["blocked_final_over_fastest_existing"] > 1 for c in cells.values()),
            "scope": CONFIG["scope"], "repetitions_are_not_independent_artifacts": True,
            "quality_improvement_claimed": False, "energy_measured": False}


def worker(directory, case, arm):
    """Fresh-process preparation, first-call and absolute peak RSS observation."""
    import psutil
    import resource
    setup_cpu()
    # Both cached native modules are loaded for every arm before the RSS baseline.
    # This is NOT a first import or native-compilation benchmark.
    m10_native.load_extension(); load_extension()
    process = psutil.Process()
    rss_baseline = process.memory_info().rss
    peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter_ns(); artifact = load_cpu_artifact(directory / case["fixture"]["artifact"])
    artifact_load_ns = time.perf_counter_ns() - start
    x = case_input(directory, case)
    start = time.perf_counter_ns(); runtime = CLASSES[arm.split("_")[0]](artifact)
    construct_ns = time.perf_counter_ns() - start
    rss_pre_call = process.memory_info().rss
    start = time.perf_counter_ns(); result = prediction(runtime, arm, x)
    api_ns = time.perf_counter_ns() - start
    is_valid = validity(x, result.answer)
    complete_ns = time.perf_counter_ns() - start
    key = output_key(result, arm); storage = retained_bytes(result)
    del result
    for _ in range(8):
        result = prediction(runtime, arm, x)
        if output_key(result, arm) != key: raise AssertionError("fresh worker replay mismatch")
        del result
    multiplier = 1 if sys.platform == "darwin" else 1024
    return {"case": f"{case['fixture']['name']}:b{case['input']['batch']}:i{case['input']['index']}",
            "arm": arm, "artifact_load_ns": artifact_load_ns, "runtime_construction_ns": construct_ns,
            "first_api_ns": api_ns, "first_inference_plus_check_ns": complete_ns,
            "rss_baseline_bytes": rss_baseline, "rss_before_first_call_bytes": rss_pre_call,
            "rss_after_calls_bytes": process.memory_info().rss,
            "process_peak_before_setup_bytes": peak_before * multiplier,
            "process_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * multiplier,
            "additional_owned_weight_bytes": runtime.backend_report().get("additional_owned_weight_bytes", 0),
            "returned_tensor_bytes": storage, "outputs": key, "valid": is_valid,
            "memory_scope": "absolute fresh-process peak, including imports and both preloaded cached extensions; not inference-only allocation"}


def profile_case(directory, case, out):
    """Inclusive stage instrumentation; never used as uninstrumented timing."""
    x = case_input(directory, case); summaries = {}
    for backend, cls in CLASSES.items():
        runtime = cls(load_cpu_artifact(directory / case["fixture"]["artifact"]))
        totals = defaultdict(int); calls = defaultdict(int)
        for name in ("_linear", "_a8", "_rms_norm", "_halt", "encode_input", "decode"):
            original = getattr(runtime, name)
            def timed(*args, _name=name, _method=original, **kwargs):
                start = time.perf_counter_ns()
                try: return _method(*args, **kwargs)
                finally:
                    totals[_name] += time.perf_counter_ns() - start; calls[_name] += 1
            setattr(runtime, name, timed)
        start = time.perf_counter_ns()
        for _ in range(10): runtime.forward(x)
        total_ns = time.perf_counter_ns() - start
        summaries[backend] = {"instrumented_total_ns": total_ns, "stage_ns": dict(totals), "stage_calls": dict(calls)}
    write(out / "profile.json", {"case": case["fixture"]["name"], "scope": "instrumented trace; stage timers inclusive; not benchmark speedup", "backends": summaries})


def run(directory, out):
    out.mkdir(parents=True, exist_ok=False)
    environment = setup_cpu(); cases = load_fixtures(directory)
    sources = source_files()
    write(out / "freeze.json", {"config": CONFIG, "arms": ARMS, "cases": cases,
          "fixture_manifest_sha256": sha(directory / "SHA256.json"), "source_sha256": sources,
          "environment": environment, "frozen_before_measurement_unix_ns": time.time_ns(),
          "identity_scope": "content hashes; this does not assert external executable preregistration"})
    with tarfile.open(out / "execution_source.tar.gz", "w:gz") as archive:
        for name in sources: archive.add(ROOT / name, arcname=name)
    setup = {}
    for name, fn in [("historical", m10_native.load_extension), ("blocked", load_extension)]:
        start = time.perf_counter_ns(); fn(); setup[name + "_load_or_build_ns"] = time.perf_counter_ns() - start
    write(out / "native_setup.json", setup)
    rng = random.Random(CONFIG["arm_order_seed"]); rows = []
    with (out / "rows.jsonl").open("x") as stream:
        for name, case in cases.items():
            x = case_input(directory, case)
            engines = {arm: CLASSES[arm.split("_")[0]](load_cpu_artifact(directory / case["fixture"]["artifact"])) for arm in ARMS}
            expected = None
            for arm in ARMS:
                for _ in range(CONFIG["warmups"]):
                    result = prediction(engines[arm], arm, x)
                    key = output_key(result, arm)
                    if expected is not None and key != expected: raise AssertionError("warmup bits differ")
                    expected = key
                    del result
            for round_id in range(CONFIG["rounds"]):
                order = list(ARMS); rng.shuffle(order)
                for position, arm in enumerate(order):
                    start = time.perf_counter_ns(); result = prediction(engines[arm], arm, x)
                    api_ns = time.perf_counter_ns() - start
                    is_valid = validity(x, result.answer)
                    complete_ns = time.perf_counter_ns() - start
                    key = output_key(result, arm)
                    if key != expected: raise AssertionError("measured bits differ")
                    row = {"case": name, "arm": arm, "round": round_id, "order": position,
                           "api_ns": api_ns, "complete_ns": complete_ns, "outputs": key,
                           "valid": is_valid, "returned_tensor_bytes": retained_bytes(result),
                           "native_calls": result.work["native_linear_calls"],
                           "halt_calls": result.work["halt_head_fp32_calls"]}
                    rows.append(row); stream.write(json.dumps(row, allow_nan=False) + "\n"); stream.flush()
                    del result
            print(name, "complete", flush=True)
    summary = analyze(rows, cases, CONFIG["rounds"]); write(out / "summary.json", summary)
    # Only the declared representative shapes receive isolated process memory runs.
    for name in CONFIG["memory_cases"]:
        if name not in cases: raise ValueError("missing declared memory case")
        for arm in ARMS:
            start = time.perf_counter_ns()
            command = [sys.executable, str(Path(__file__).resolve()), "worker", "--fixtures", str(directory.resolve()), "--case", name, "--arm", arm]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
            if completed.returncode: raise RuntimeError(completed.stderr)
            observation = json.loads(completed.stdout)
            observation["subprocess_wall_ns"] = time.perf_counter_ns() - start
            write(out / ("memory-" + name.replace(":", "-") + "-" + arm + ".json"), observation)
    profile_case(directory, cases[CONFIG["memory_cases"][-1]], out)
    write(out / "SHA256.json", {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps(summary["ratio_of_summed_case_medians"], indent=2))


def verify(directory, out, replay):
    for name, expected in read(out / "SHA256.json").items():
        if (out / name).resolve().parent != out.resolve() or sha(out / name) != expected:
            raise ValueError("measurement file digest/path mismatch: " + name)
    frozen = read(out / "freeze.json")
    if sha(directory / "SHA256.json") != frozen["fixture_manifest_sha256"]: raise ValueError("fixtures changed")
    cases = load_fixtures(directory)
    if cases != frozen["cases"] or tuple(frozen["arms"]) != ARMS: raise ValueError("matrix changed")
    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines()]
    summary = analyze(rows, cases, frozen["config"]["rounds"])
    if summary != read(out / "summary.json"): raise ValueError("summary not reproducible")
    expected = {(r["case"], r["arm"]): (r["outputs"], r["valid"]) for r in rows}
    for name in frozen["config"]["memory_cases"]:
        for arm in ARMS:
            record = read(out / ("memory-" + name.replace(":", "-") + "-" + arm + ".json"))
            if record["case"] != name or record["arm"] != arm:
                raise ValueError("isolated memory cell mismatch")
            if (record["outputs"], record["valid"]) != expected[(name, arm)]:
                raise ValueError("isolated worker outputs differ from timed outputs")
            for field in ("process_peak_bytes", "rss_baseline_bytes", "returned_tensor_bytes"):
                if type(record[field]) is not int or record[field] < 0:
                    raise ValueError("invalid isolated memory observation")
    replays = 0
    if replay:
        setup_cpu()
        if source_files() != frozen["source_sha256"]: raise ValueError("replay source differs from frozen executable")
        expected = {(r["case"], r["arm"]): (r["outputs"], r["valid"]) for r in rows}
        for name, case in cases.items():
            x = case_input(directory, case)
            for arm in ARMS:
                runtime = CLASSES[arm.split("_")[0]](load_cpu_artifact(directory / case["fixture"]["artifact"]))
                result = prediction(runtime, arm, x)
                if (output_key(result, arm), validity(x, result.answer)) != expected[(name, arm)]:
                    raise ValueError("execution replay mismatch")
                replays += 1
    result = {"status": "PASS", "observations": len(rows), "replayed_unique_executions": replays,
              "scope": "output/analysis replay does not recreate historical timings"}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "verify", "worker"])
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--case"); parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.command == "worker":
        cases = load_fixtures(args.fixtures)
        print(json.dumps(worker(args.fixtures, cases[args.case], args.arm)))
    elif args.out is None:
        parser.error("--out is required for run/verify")
    elif args.command == "run": run(args.fixtures, args.out)
    else: verify(args.fixtures, args.out, args.replay)
