"""Run the declared comparison after a published freeze; retain every row."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.focused_evaluation.study import (
    ARMS, MEMORY_ARMS, ROUNDS, SEARCH_SEEDS, append, call, case_specs,
    check_freeze, generate, require,
)


def load_native():
    import pysat
    from pysat.solvers import Solver
    require(pysat.__version__ == "1.9.dev15", "python-sat version must be 1.9.dev15")
    return pysat


def memory_worker(arm, trace):
    case = json.load(sys.stdin)
    load_native()  # Common interpreter/native-import baseline for every arm.
    gc.collect()
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if trace:
        tracemalloc.start()
    result = call(case, arm, SEARCH_SEEDS[0])
    peak = tracemalloc.get_traced_memory()[1] if trace else None
    if trace:
        tracemalloc.stop()
    print(json.dumps({"id": case["id"], "arm": arm, "status": result["status"],
                      "python_peak_bytes": peak, "baseline_rss_kib": before,
                      "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))


def run_memory(out, cases):
    # A fresh small relay prevents the large timing supervisor's RSS from setting
    # the worker's inherited high-water floor. Its child measures a new process.
    relay = "import subprocess,sys; raise SystemExit(subprocess.call(sys.argv[1:]))"
    with (out/"memory.jsonl").open("x") as stream:
        for case in cases:
            if case["index"] >= 2:
                continue
            for arm in MEMORY_ARMS:
                base = [sys.executable, "-I", "-S", "-c", relay, sys.executable,
                        "-I", str(Path(__file__).resolve()), "worker", "--arm", arm]
                rss = subprocess.run(base, input=json.dumps(case), capture_output=True,
                                     text=True, timeout=60, check=True)
                row = json.loads(rss.stdout)
                if arm != "glucose4":
                    traced = subprocess.run(base+["--trace"], input=json.dumps(case),
                                            capture_output=True, text=True, timeout=60, check=True)
                    row["python_peak_bytes"] = json.loads(traced.stdout)["python_peak_bytes"]
                append(stream, row)


def run(out, freeze_path):
    freeze = json.loads(freeze_path.read_text())
    check_freeze(freeze)
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            == freeze["commit"], "run from the published frozen commit")
    pysat = load_native()
    import pysolvers
    import pysat.solvers
    require(sys.platform == "linux", "RSS units and affinity require Linux")
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    out.mkdir(parents=True, exist_ok=False)
    lock = {"freeze": freeze, "python": sys.version, "platform": platform.platform(),
            "python_sat": pysat.__version__, "affinity": sorted(os.sched_getaffinity(0)),
            "native_sources": {str(Path(module.__file__).name): hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                               for module in (pysat, pysat.solvers, pysolvers)},
            "cpu": next(s.split(":", 1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines()
                        if s.startswith("model name")), "status": "RUN_STARTED"}
    (out/"LOCK.json").write_text(json.dumps(lock, indent=2)+"\n")
    cases = [generate(spec) for spec in case_specs()]
    require(len({c["sha256"] for c in cases}) == len(cases), "duplicate generated formula")
    with (out/"cases.jsonl").open("x") as stream:
        for case in cases:
            append(stream, case)
    try:
        with (out/"rows.jsonl").open("x") as stream:
            for round_id in range(ROUNDS):
                for i, case in enumerate(cases):
                    for slot, seed in enumerate(SEARCH_SEEDS):
                        offset = (i+round_id+slot) % len(ARMS)
                        order = ARMS[offset:]+ARMS[:offset]
                        if (i+round_id+slot) % 2:
                            order = order[::-1]
                        for arm in order:
                            cpu_start, wall_start = time.process_time_ns(), time.perf_counter_ns()
                            result = call(case, arm, seed)
                            wall_ns, cpu_ns = time.perf_counter_ns()-wall_start, time.process_time_ns()-cpu_start
                            append(stream, {"id": case["id"], "sha256": case["sha256"], "arm": arm,
                                            "seed": seed, "round": round_id, "wall_ns": wall_ns,
                                            "cpu_ns": cpu_ns, "result": result})
                print(f"timing round {round_id+1}/{ROUNDS} complete", flush=True)
        run_memory(out, cases)
        files = ["LOCK.json", "cases.jsonl", "rows.jsonl", "memory.jsonl"]
        manifest = {"status": "COMPLETE", "files": {name: {
            "bytes": (out/name).stat().st_size,
            "sha256": hashlib.sha256((out/name).read_bytes()).hexdigest()} for name in files}}
        (out/"MANIFEST.json").write_text(json.dumps(manifest, indent=2)+"\n")
        print("timing and memory inventories complete", flush=True)
    except BaseException as error:
        (out/"FAILURE.json").write_text(json.dumps({"type": type(error).__name__, "message": str(error)}, indent=2)+"\n")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--out", type=Path, required=True)
    run_parser.add_argument("--freeze", type=Path, required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--arm", choices=MEMORY_ARMS, required=True)
    worker.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.command == "worker":
        memory_worker(args.arm, args.trace)
    else:
        run(args.out.resolve(), args.freeze.resolve())


if __name__ == "__main__":
    main()
