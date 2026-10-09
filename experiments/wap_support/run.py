"""Build fixed WAP cases and execute isolated complete solver sessions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import sys
import time
import traceback
import subprocess
import zlib

from experiments.wap_support.controls import ARMS, run_session
from experiments.wap_support.workload import WorkloadCase, build_case

ORDERS = ("forward", "reverse", "shuffle")
ADDRESS_SPACE_BYTES = 4 * 1024**3
SESSION_DEADLINE_SECONDS = 60
SCHEDULE_SEED = 0xC011AB1E


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(payload)
    os.replace(temp, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(path, (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode())


def build_cases(graphs: Path, inventory_path: Path, output: Path, roles: set[str]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "graphs").mkdir()
    (output / "cases").mkdir()
    (output / "generators").mkdir()
    inventory = json.loads(inventory_path.read_text())
    selected = [(name, record) for name, record in sorted(inventory["graphs"].items())
                if record["role"] in roles]
    if not selected:
        raise ValueError("no graph matches requested roles")
    receipts = []
    for name, record in selected:
        source = graphs / name
        if not source.is_file():
            raise FileNotFoundError(source)
        target = output / "graphs" / name
        target.write_bytes(source.read_bytes())
        start = time.perf_counter_ns()
        case, generator = build_case(target, graph_name=name,
                                     expected_git_blob_sha1=record["git_blob_sha1"])
        construction_ns = time.perf_counter_ns() - start
        case_path = output / "cases" / f"{name}.json"
        generator_path = output / "generators" / f"{name}.json"
        case.to_json(case_path)
        generator["construction_ns"] = construction_ns
        generator["graph_role"] = record["role"]
        generator["graph_bytes"] = len(target.read_bytes())
        generator["graph_sha256"] = sha256(target)
        atomic_json(generator_path, generator)
        receipts.append({
            "name": name, "role": record["role"], "case_sha256": case.case_sha256,
            "case_file_sha256": sha256(case_path), "generator_file_sha256": sha256(generator_path),
            "graph_sha256": sha256(target), "graph_git_blob_sha1": record["git_blob_sha1"],
            "construction_ns": construction_ns,
        })
        print(json.dumps(receipts[-1], sort_keys=True), flush=True)
    atomic_json(output / "case_inventory.json", {
        "schema": "spectra.wap_support.case_inventory.v1",
        "upstream_repository": inventory["repository"],
        "upstream_commit": inventory["commit"],
        "inventory_sha256": sha256(inventory_path),
        "roles": sorted(roles), "cases": receipts,
    })
    write_manifest(output)


def _machine() -> dict:
    cpu = Path("/proc/cpuinfo").read_text(errors="replace")
    model = next((line.split(":", 1)[1].strip() for line in cpu.splitlines()
                  if line.startswith("model name")), "unknown")
    status = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("Threads:", "VmRSS:", "VmHWM:")):
            key, value = line.split(":", 1)
            status[key] = int(value.split()[0])
    return {
        "platform": platform.platform(), "python": sys.version, "cpu_model": model,
        "affinity": sorted(os.sched_getaffinity(0)), "pid": os.getpid(),
        "address_space_limit_bytes": ADDRESS_SPACE_BYTES,
        "threads": status.get("Threads", -1),
        "vmrss_kib": status.get("VmRSS", -1),
        "vmhwm_kib": status.get("VmHWM", -1),
    }


def run_worker(case_path: Path, arm: str, runtime_path: Path,
               order: str, order_seed: int, destination: Path) -> None:
    resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_SPACE_BYTES, ADDRESS_SPACE_BYTES))
    affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(affinity)})
    from spectra.cnf.quotient_query import QuotientRuntime
    runtime = QuotientRuntime(runtime_path)
    if arm in ("minicard", "cadical195", "cache_minicard"):
        import pysat.solvers  # preload the competitor library outside warm-session timing
    complete_start = time.perf_counter_ns()
    case = WorkloadCase.from_json(case_path)
    result = run_session(case, runtime, arm, order_name=order, order_seed=order_seed)
    complete_ns = time.perf_counter_ns() - complete_start
    compressed = zlib.compress(result.output_bytes, level=9)
    output_name = "outputs.bin.zlib"
    atomic_bytes(destination / output_name, compressed)
    metadata = result.metadata()
    metadata.update({
        "complete_ns": complete_ns,
        "output_file": output_name,
        "output_compressed_bytes": len(compressed),
        "output_compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "machine": _machine(),
    })
    atomic_json(destination / "session.json", metadata)


def schedule(cases: list[Path]) -> list[tuple[str, str, str, int]]:
    jobs = []
    for case_path in cases:
        case = WorkloadCase.from_json(case_path)
        for arm in ARMS:
            for round_index, order in enumerate(ORDERS):
                seed = (SCHEDULE_SEED ^ int(case.case_sha256[:16], 16)
                        ^ (round_index << 20) ^ sum(map(ord, arm))) & ((1 << 64) - 1)
                jobs.append((case_path.name, arm, order, seed))
    random.Random(SCHEDULE_SEED).shuffle(jobs)
    return jobs


def _append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def run_matrix(cases_dir: Path, runtime_path: Path, output: Path, *, max_new: int | None = None) -> None:
    if output.exists():
        if not (output / "schedule.json").is_file():
            raise FileExistsError("existing output is not a resumable study")
    else:
        output.mkdir(parents=True)
        (output / "sessions").mkdir()
    cases = sorted(cases_dir.glob("*.json"))
    if not cases:
        raise ValueError("no case files")
    jobs = schedule(cases)
    schedule_path = output / "schedule.json"
    if schedule_path.exists():
        if json.loads(schedule_path.read_text()) != [list(x) for x in jobs]:
            raise ValueError("existing schedule differs")
    else:
        atomic_json(schedule_path, jobs)
        environment = _machine()
        environment["runtime_library_sha256"] = sha256(runtime_path)
        receipt = runtime_path.parent / "build.json"
        environment["runtime_build_receipt_sha256"] = sha256(receipt) if receipt.is_file() else None
        try:
            import pysat
            environment["python_sat_version"] = pysat.__version__
        except Exception:
            environment["python_sat_version"] = None
        atomic_json(output / "environment.json", environment)
    lookup = {path.name: path for path in cases}
    rows_path = output / "sessions.jsonl"
    rows = []
    if rows_path.exists():
        rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    completed = {row["session_id"] for row in rows}
    added = 0
    for number, (case_name, arm, order, seed) in enumerate(jobs):
        session_id = f"{number:04d}-{case_name.removesuffix('.json')}-{arm}-{order}"
        if session_id in completed:
            continue
        destination = output / "sessions" / session_id
        if destination.exists():
            import shutil
            shutil.rmtree(destination)
        destination.mkdir()
        command = [sys.executable, "-m", "experiments.wap_support.run", "worker",
                   "--case", str(lookup[case_name]), "--arm", arm,
                   "--runtime", str(runtime_path), "--order", order,
                   "--order-seed", str(seed), "--destination", str(destination)]
        cold_start = time.perf_counter_ns()
        completed_process = subprocess.run(command, capture_output=True, text=True,
                                           timeout=SESSION_DEADLINE_SECONDS)
        cold_wall_ns = time.perf_counter_ns() - cold_start
        if completed_process.returncode:
            raise RuntimeError(f"session failed {session_id}:\n{completed_process.stdout}\n{completed_process.stderr}")
        metadata_path = destination / "session.json"
        if not metadata_path.is_file():
            raise RuntimeError("worker did not publish session metadata")
        metadata = json.loads(metadata_path.read_text())
        row = {
            "session_id": session_id, "case_file": case_name, "arm": arm,
            "order": order, "order_seed": seed, "cold_wall_ns": cold_wall_ns,
            "complete_ns": metadata["complete_ns"],
            "session_metadata_sha256": sha256(metadata_path),
            "output_compressed_sha256": metadata["output_compressed_sha256"],
        }
        atomic_json(destination / "outer.json", row)
        _append_jsonl(rows_path, row)
        rows.append(row); completed.add(session_id); added += 1
        print(f"{len(rows)}/{len(jobs)} {session_id} {row['complete_ns']/1e6:.3f} ms", flush=True)
        if max_new is not None and added >= max_new:
            break
    require_count = len(jobs)
    if len(rows) != require_count:
        print(f"checkpointed incomplete matrix {len(rows)}/{require_count}", flush=True)
        return
    # Canonical order, independent of resume order.
    by_id = {row["session_id"]: row for row in rows}
    ordered = [by_id[f"{i:04d}-{case.removesuffix('.json')}-{arm}-{order}"]
               for i, (case, arm, order, _seed) in enumerate(jobs)]
    atomic_json(output / "sessions.json", ordered)
    write_manifest(output)

def write_manifest(root: Path) -> None:
    manifest = {str(path.relative_to(root)): sha256(path)
                for path in sorted(root.rglob("*"))
                if path.is_file() and path.name != "MANIFEST.json"}
    atomic_json(root / "MANIFEST.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-cases")
    build.add_argument("--graphs", type=Path, required=True)
    build.add_argument("--inventory", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--roles", nargs="+", required=True)
    run = sub.add_parser("run")
    run.add_argument("--cases", type=Path, required=True)
    run.add_argument("--runtime", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--max-new", type=int)
    worker = sub.add_parser("worker")
    worker.add_argument("--case", type=Path, required=True)
    worker.add_argument("--arm", required=True)
    worker.add_argument("--runtime", type=Path, required=True)
    worker.add_argument("--order", required=True)
    worker.add_argument("--order-seed", type=int, required=True)
    worker.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build-cases":
        build_cases(args.graphs.resolve(), args.inventory.resolve(), args.out.resolve(), set(args.roles))
    elif args.command == "run":
        run_matrix(args.cases.resolve(), args.runtime.resolve(), args.out.resolve(), max_new=args.max_new)
    else:
        run_worker(args.case.resolve(), args.arm, args.runtime.resolve(), args.order,
                   args.order_seed, args.destination.resolve())

if __name__ == "__main__":
    main()
