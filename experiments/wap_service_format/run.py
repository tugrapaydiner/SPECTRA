"""Compile hash-bound WAP service cases and measure complete packed sessions.

This is an exposed-data deployment study, not a second holdout.  Process startup,
imports, and native-library loading are outside the complete timer exactly as in
the frozen study.  The complete timer includes packed-file read and SHA-256
verification, restricted data-only decode, fresh preparation, every query, full
original witness checking, diagnostics, and disposal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
import time
import zlib

from experiments.wap_service_format.format import (
    load_packed_case, pack_directory, read_manifest, receipt_by_name,
)
from experiments.wap_support.controls import run_prevalidated_session

ARMS = ("scc", "none", "minicard", "cadical195")
ORDERS = ("forward", "reverse", "shuffle")
SCHEDULE_SEED = 0x51A7E2D5
ADDRESS_SPACE_BYTES = 4 * 1024**3
DEADLINE_SECONDS = 60


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(path, (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode())


def machine() -> dict:
    status = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("Threads:", "VmRSS:", "VmHWM:")):
            key, value = line.split(":", 1)
            status[key] = int(value.split()[0])
    cpu = Path("/proc/cpuinfo").read_text(errors="replace")
    model = next((line.split(":", 1)[1].strip() for line in cpu.splitlines()
                  if line.startswith("model name")), "unknown")
    return {
        "platform": platform.platform(), "python": sys.version, "cpu_model": model,
        "affinity": sorted(os.sched_getaffinity(0)), "pid": os.getpid(),
        "address_space_limit_bytes": ADDRESS_SPACE_BYTES,
        "threads": status.get("Threads", -1), "vmrss_kib": status.get("VmRSS", -1),
        "vmhwm_kib": status.get("VmHWM", -1),
        "numerical_frameworks_loaded": [name for name in ("numpy", "torch", "scipy", "sklearn")
                                         if name in sys.modules],
    }


def schedule(manifest: dict) -> list[tuple[str, str, str, int]]:
    jobs = []
    for row in sorted(manifest["cases"], key=lambda value: value["packed_name"]):
        for arm in ARMS:
            for round_index, order in enumerate(ORDERS):
                seed = (SCHEDULE_SEED ^ int(row["case_sha256"][:16], 16)
                        ^ (round_index << 20) ^ sum(map(ord, arm))) & ((1 << 64) - 1)
                jobs.append((row["packed_name"], arm, order, seed))
    random.Random(SCHEDULE_SEED).shuffle(jobs)
    return jobs


def worker(case_path: Path, expected_sha256: str, runtime_path: Path, arm: str,
           order: str, order_seed: int, destination: Path) -> None:
    resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_SPACE_BYTES, ADDRESS_SPACE_BYTES))
    affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(affinity)})
    from spectra.cnf.quotient_query import QuotientRuntime
    runtime = QuotientRuntime(runtime_path)
    if arm in ("minicard", "cadical195"):
        import pysat.solvers  # competitor library load excluded, as in the frozen endpoint
    complete_start = time.perf_counter_ns()
    load_start = complete_start
    loaded = load_packed_case(case_path, expected_sha256=expected_sha256)
    load_ns = time.perf_counter_ns() - load_start
    result = run_prevalidated_session(
        loaded.case, runtime, arm, order_name=order, order_seed=order_seed,
    )
    complete_ns = time.perf_counter_ns() - complete_start
    compressed = zlib.compress(result.output_bytes, 1)
    atomic_bytes(destination / "outputs.bin.zlib", compressed)
    metadata = result.metadata()
    metadata.update({
        "schema": "spectra.wap_service_format.session.v1",
        "complete_ns": complete_ns,
        "packed_load_ns": load_ns,
        "packed_file_sha256": loaded.file_sha256,
        "packed_payload_sha256": loaded.payload_sha256,
        "packed_bytes": loaded.packed_bytes,
        "payload_bytes": loaded.payload_bytes,
        "output_file": "outputs.bin.zlib",
        "output_compression_level": 1,
        "output_compressed_bytes": len(compressed),
        "output_compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "machine": machine(),
    })
    atomic_json(destination / "session.json", metadata)


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_manifest(root: Path) -> None:
    manifest = {str(path.relative_to(root)): sha256(path)
                for path in sorted(root.rglob("*"))
                if path.is_file() and path.name != "MANIFEST.json"}
    atomic_json(root / "MANIFEST.json", manifest)


def run_matrix(packed_root: Path, runtime_path: Path, output: Path) -> None:
    manifest_path = packed_root / "MANIFEST.json"
    manifest = read_manifest(manifest_path)
    receipts = receipt_by_name(manifest)
    jobs = schedule(manifest)
    schedule_path = output / "schedule.json"
    environment_path = output / "environment.json"
    if output.exists():
        if not schedule_path.is_file() or not environment_path.is_file():
            raise FileExistsError("existing output is not a resumable packed study")
        if json.loads(schedule_path.read_text()) != [list(x) for x in jobs]:
            raise ValueError("existing packed schedule differs")
    else:
        output.mkdir(parents=True)
        (output / "sessions").mkdir()
        atomic_json(schedule_path, jobs)
        environment = machine()
        environment.update({
            "schema": "spectra.wap_service_format.environment.v1",
            "runtime_library_sha256": sha256(runtime_path),
            "packed_manifest_sha256": sha256(manifest_path),
            "arms": list(ARMS), "orders": list(ORDERS),
        })
        try:
            import pysat
            environment["python_sat_version"] = pysat.__version__
        except Exception:
            environment["python_sat_version"] = None
        atomic_json(environment_path, environment)
    rows_path = output / "sessions.jsonl"
    rows = ([json.loads(line) for line in rows_path.read_text().splitlines() if line]
            if rows_path.exists() else [])
    completed = {row.get("session_id") for row in rows}
    if len(completed) != len(rows):
        raise ValueError("duplicate session id in resumable packed study")
    for number, (packed_name, arm, order, seed) in enumerate(jobs):
        session_id = f"{number:04d}-{packed_name.removesuffix('.spwap')}-{arm}-{order}"
        if session_id in completed:
            continue
        destination = output / "sessions" / session_id
        if destination.exists():
            import shutil
            shutil.rmtree(destination)
        destination.mkdir()
        receipt = receipts[packed_name]
        command = [
            sys.executable, "-B", "-m", "experiments.wap_service_format.run", "worker",
            "--case", str(packed_root / packed_name),
            "--expected-sha256", receipt.packed_file_sha256,
            "--runtime", str(runtime_path), "--arm", arm, "--order", order,
            "--order-seed", str(seed), "--destination", str(destination),
        ]
        start = time.perf_counter_ns()
        try:
            process = subprocess.run(command, capture_output=True, text=True,
                                     timeout=DEADLINE_SECONDS)
        except subprocess.TimeoutExpired as exc:
            cold_wall_ns = time.perf_counter_ns() - start
            failure = {"schema": "spectra.wap_service_format.failure.v1", "status": "TIMEOUT",
                       "session_id": session_id, "packed_name": packed_name, "arm": arm,
                       "order": order, "order_seed": seed, "cold_wall_ns": cold_wall_ns,
                       "stdout": exc.stdout or "", "stderr": exc.stderr or ""}
            atomic_json(destination / "failure.json", failure)
            row = {"session_id": session_id, "packed_name": packed_name, "arm": arm,
                   "order": order, "order_seed": seed, "status": "TIMEOUT",
                   "cold_wall_ns": cold_wall_ns,
                   "failure_sha256": sha256(destination / "failure.json")}
            atomic_json(destination / "outer.json", row); append_jsonl(rows_path, row)
            rows.append(row); write_manifest(output)
            raise RuntimeError("packed service session timed out: " + session_id) from exc
        cold_wall_ns = time.perf_counter_ns() - start
        if process.returncode:
            failure = {"schema": "spectra.wap_service_format.failure.v1", "status": "ERROR",
                       "session_id": session_id, "packed_name": packed_name, "arm": arm,
                       "order": order, "order_seed": seed, "cold_wall_ns": cold_wall_ns,
                       "returncode": process.returncode, "stdout": process.stdout,
                       "stderr": process.stderr}
            atomic_json(destination / "failure.json", failure)
            row = {"session_id": session_id, "packed_name": packed_name, "arm": arm,
                   "order": order, "order_seed": seed, "status": "ERROR",
                   "cold_wall_ns": cold_wall_ns,
                   "failure_sha256": sha256(destination / "failure.json")}
            atomic_json(destination / "outer.json", row); append_jsonl(rows_path, row)
            rows.append(row); write_manifest(output)
            raise RuntimeError(f"packed service session failed {session_id}:\n{process.stderr}")
        session_path = destination / "session.json"
        metadata = json.loads(session_path.read_text())
        row = {"session_id": session_id, "packed_name": packed_name, "arm": arm,
               "order": order, "order_seed": seed, "status": "COMPLETE",
               "cold_wall_ns": cold_wall_ns, "complete_ns": metadata["complete_ns"],
               "packed_load_ns": metadata["packed_load_ns"],
               "session_metadata_sha256": sha256(session_path),
               "output_compressed_sha256": metadata["output_compressed_sha256"]}
        atomic_json(destination / "outer.json", row); append_jsonl(rows_path, row)
        rows.append(row); completed.add(session_id)
        atomic_json(output / "progress.json", {"complete": False, "rows": len(rows),
                                                 "scheduled": len(jobs), "last": row})
        print(f"{len(rows)}/{len(jobs)} {session_id} {row['complete_ns']/1e6:.3f} ms", flush=True)
    by_id = {row["session_id"]: row for row in rows}
    ordered = [by_id[f"{i:04d}-{name.removesuffix('.spwap')}-{arm}-{order}"]
               for i, (name, arm, order, _seed) in enumerate(jobs)]
    atomic_json(output / "sessions.json", ordered)
    atomic_json(output / "progress.json", {"complete": True, "rows": len(ordered),
                                             "scheduled": len(jobs), "last": ordered[-1]})
    write_manifest(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("pack")
    pack.add_argument("--cases", type=Path, required=True)
    pack.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--packed", type=Path, required=True)
    run.add_argument("--runtime", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    work = sub.add_parser("worker")
    work.add_argument("--case", type=Path, required=True)
    work.add_argument("--expected-sha256", required=True)
    work.add_argument("--runtime", type=Path, required=True)
    work.add_argument("--arm", choices=ARMS, required=True)
    work.add_argument("--order", choices=ORDERS, required=True)
    work.add_argument("--order-seed", type=int, required=True)
    work.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "pack":
        manifest = pack_directory(args.cases.resolve(), args.out.resolve())
        print(json.dumps({"cases": len(manifest["cases"]), "out": str(args.out.resolve())},
                         sort_keys=True))
    elif args.command == "run":
        run_matrix(args.packed.resolve(), args.runtime.resolve(), args.out.resolve())
    else:
        worker(args.case.resolve(), args.expected_sha256, args.runtime.resolve(),
               args.arm, args.order, args.order_seed, args.destination.resolve())


if __name__ == "__main__":
    main()
