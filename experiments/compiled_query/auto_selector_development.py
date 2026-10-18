"""Paired oracle study for the exposed automatic-selector development panel.

This file measures only already exposed synthetic cases.  It is not confirmation.
The selector may be tuned from this output, but confirmation must use the separately
frozen prospective panel in AUTO_SELECTOR_V3_PROTOCOL_20261018.md.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback

from .cases import Case, canonical, digest
from .exact import ExactRuntime, pack_session_packet, verify_session_output
from .sessions import decode_certified_output
from .transfer_panel import FAMILIES, make_case, seed_for

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "spectra.auto_selector_development.v1"
CLASSIFICATION = "exposed synthetic selector-development panel; not confirmation"
COUNTS = (1, 63, 64, 65, 256, 1024)
MODES = ("raw", "scc", "sparse", "route", "auto")
EXPLICIT = MODES[:-1]
REPEATS = 5
WARMUPS = 2
REPLICAS = tuple(range(6))


def _write_once(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return digest(payload)


def _put(path: Path, value: object) -> str:
    return _write_once(path, canonical(value))


def _gmean(values: list[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _sources() -> dict[str, str]:
    paths = (
        ROOT / "spectra" / "exact.py",
        ROOT / "spectra" / "exact_problem.py",
        ROOT / "spectra" / "_native" / "exact_binary.cpp",
        Path(__file__),
        Path(__file__).with_name("AUTO_SELECTOR_V3_PROTOCOL_20261018.md"),
        Path(__file__).with_name("transfer_panel.py"),
    )
    return {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in paths}


def freeze(destination: Path, library: Path) -> dict:
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    library = library.resolve()
    receipt = library.with_name("build.json")
    library_bytes = library.read_bytes()
    receipt_bytes = receipt.read_bytes()
    build = json.loads(receipt_bytes)
    if build.get("library_sha256") != digest(library_bytes):
        raise ValueError("native library differs from its build receipt")
    if build.get("source_sha256") != digest((ROOT / "spectra/_native/exact_binary.cpp").read_bytes()):
        raise ValueError("native source differs from its build receipt")

    cases = []
    for family in FAMILIES:
        for replica in REPLICAS:
            case = make_case(family, replica)
            payload = case.payload()
            sha = _write_once(destination / "cases" / f"{case.name}.json", payload)
            cases.append({
                "name": case.name,
                "family": family,
                "replica": replica,
                "n": case.problem.n,
                "edges": len(case.problem.edges),
                "pins": sum(map(len, case.queries)),
                "case_sha256": sha,
            })

    copied_library = destination / "native" / library.name
    copied_receipt = destination / "native" / "build.json"
    _write_once(copied_library, library_bytes)
    _write_once(copied_receipt, receipt_bytes)
    manifest = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_tree": subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True
        ).strip(),
        "sources": _sources(),
        "library": str(copied_library.relative_to(destination)),
        "library_sha256": digest(library_bytes),
        "receipt_sha256": digest(receipt_bytes),
        "cases": cases,
        "counts": COUNTS,
        "modes": MODES,
        "repeats": REPEATS,
        "warmups": WARMUPS,
        "deadline_seconds": 120,
        "address_space_bytes": 2 << 30,
        "native_bytes": 512 << 20,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    manifest_sha = _put(destination / "manifest.json", manifest)
    _write_once(destination / "manifest.sha256", (manifest_sha + "\n").encode())
    return manifest


def _manifest(root: Path, *, executing: bool = False) -> dict:
    payload = (root / "manifest.json").read_bytes()
    if digest(payload) != (root / "manifest.sha256").read_text().strip():
        raise ValueError("manifest hash differs")
    manifest = json.loads(payload)
    if (manifest.get("schema") != SCHEMA
            or tuple(manifest.get("counts", ())) != COUNTS
            or tuple(manifest.get("modes", ())) != MODES
            or manifest.get("repeats") != REPEATS
            or manifest.get("warmups") != WARMUPS):
        raise ValueError("selector-development schedule differs")
    expected = {(family, replica) for family in FAMILIES for replica in REPLICAS}
    actual = {(item["family"], item["replica"]) for item in manifest["cases"]}
    if actual != expected or len(manifest["cases"]) != len(expected):
        raise ValueError("selector-development case roster differs")
    if executing and manifest["sources"] != _sources():
        raise ValueError("executing source differs from frozen source")
    return manifest


def _record_name(count: int, repeat: int, mode: str) -> str:
    return f"{count}-{repeat:02d}-{mode}.json"


def worker(root: Path, case_index: int) -> None:
    import resource

    manifest = _manifest(root, executing=True)
    resource.setrlimit(resource.RLIMIT_AS, (manifest["address_space_bytes"],) * 2)
    if hasattr(os, "sched_getaffinity"):
        cpus = sorted(os.sched_getaffinity(0))
        if cpus:
            os.sched_setaffinity(0, {cpus[0]})

    item = manifest["cases"][case_index]
    case_path = root / "cases" / f"{item['name']}.json"
    case_payload = case_path.read_bytes()
    if digest(case_payload) != item["case_sha256"]:
        raise ValueError("case hash differs")
    case = Case.decode(case_payload)
    library = root / manifest["library"]
    if digest(library.read_bytes()) != manifest["library_sha256"]:
        raise ValueError("native library hash differs")
    if digest(library.with_name("build.json").read_bytes()) != manifest["receipt_sha256"]:
        raise ValueError("native receipt hash differs")
    runtime = ExactRuntime(library)

    folder = root / "runs" / case.name
    folder.mkdir(parents=True, exist_ok=False)
    output_files: dict[str, str] = {}
    packet_files: dict[str, dict] = {}
    records: dict[str, str] = {}
    rng = random.Random(seed_for("auto-selector-development/" + case.name))
    gc.disable()
    try:
        for count in COUNTS:
            for repeat in range(REPEATS):
                order = list(range(count))
                if repeat % 3 == 1:
                    order.reverse()
                elif repeat % 3 == 2:
                    random.Random(seed_for(
                        f"auto-selector-development/{case.name}/{count}/{repeat}"
                    )).shuffle(order)
                rows = tuple(case.queries[index] for index in order)
                packet = pack_session_packet(case.problem, rows)
                packet_sha = digest(packet)
                if packet_sha not in packet_files:
                    _write_once(folder / "packets" / f"{packet_sha}.sbc1", packet)
                    packet_files[packet_sha] = {"count": count, "order": order}

                mode_order = list(MODES)
                rng.shuffle(mode_order)
                outputs_this_repeat: dict[str, str] = {}
                decisions_this_repeat: dict[str, tuple[int, ...]] = {}
                for mode in mode_order:
                    record = {
                        "case": case.name,
                        "family": item["family"],
                        "count": count,
                        "repeat": repeat,
                        "mode": mode,
                        "packet_sha256": packet_sha,
                    }
                    try:
                        if repeat == 0:
                            for _ in range(WARMUPS):
                                runtime.solve_session_packet(
                                    packet, mode=mode,
                                    max_bytes=manifest["native_bytes"],
                                )
                        started = time.perf_counter_ns()
                        answer = runtime.solve_session_packet(
                            packet, mode=mode,
                            max_bytes=manifest["native_bytes"],
                        )
                        elapsed = time.perf_counter_ns() - started
                        if not verify_session_output(packet, answer.output):
                            raise AssertionError("independent session replay failed")
                        rows_out = decode_certified_output(
                            answer.output, case.problem.n, count
                        )
                        decisions = tuple(code for code, _, _ in rows_out)
                        output_sha = digest(answer.output)
                        if output_sha not in output_files:
                            _write_once(
                                folder / "outputs" / f"{output_sha}.scq2",
                                answer.output,
                            )
                            output_files[output_sha] = mode
                        outputs_this_repeat[mode] = output_sha
                        decisions_this_repeat[mode] = decisions
                        record.update({
                            "status": "COMPLETE",
                            "complete_ns": elapsed,
                            "output_sha256": output_sha,
                            "output_bytes": len(answer.output),
                            "sat": answer.sat_count,
                            "unsat": answer.unsat_count,
                            "proof_bytes": answer.proof_bytes,
                            "selected_mode": answer.info.get("selected_mode", mode),
                            "auto_policy": answer.info.get("auto_policy"),
                            "auto_policy_reason": answer.info.get("auto_policy_reason"),
                            "info": answer.info,
                        })
                    except Exception as error:
                        record.update({
                            "status": "ERROR",
                            "error": repr(error),
                            "traceback": traceback.format_exc(),
                        })
                    name = _record_name(count, repeat, mode)
                    records[name] = _put(folder / "records" / name, record)

                complete_modes = {
                    mode for mode in MODES
                    if decisions_this_repeat.get(mode) is not None
                }
                if complete_modes == set(MODES):
                    decisions = {decisions_this_repeat[mode] for mode in MODES}
                    if len(decisions) != 1:
                        raise AssertionError("explicit and automatic decisions disagree")
                    auto_record = json.loads(
                        (folder / "records" / _record_name(count, repeat, "auto")).read_bytes()
                    selected = auto_record["selected_mode"]
                    if selected not in EXPLICIT:
                        raise AssertionError("auto reported an unknown explicit engine")
                    if outputs_this_repeat["auto"] != outputs_this_repeat[selected]:
                        raise AssertionError("auto output differs from its reported engine")
    finally:
        gc.enable()

    packets_sha = _put(folder / "packets.json", packet_files)
    worker_record = {
        "case": case.name,
        "manifest_sha256": digest((root / "manifest.json").read_bytes()),
        "records": records,
        "packets_sha256": packets_sha,
        "output_count": len(output_files),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    _put(folder / "worker.json", worker_record)


def execute(root: Path, library: Path) -> None:
    manifest = freeze(root, library)
    for case_index, item in enumerate(manifest["cases"]):
        case_folder = root / "runs" / item["name"]
        controller_path = case_folder / "controller.json"
        command = [
            sys.executable, "-m", "experiments.compiled_query.auto_selector_development",
            "_worker", "--root", str(root.resolve()), "--case", str(case_index),
        ]
        started = time.perf_counter_ns()
        status = "COMPLETE"
        case_folder.mkdir(parents=True, exist_ok=False)
        # The worker owns the final case directory. Remove the controller shell
        # before spawning so creation remains fail-closed and atomic per case.
        case_folder.rmdir()
        stdout_path = root / "controller" / f"{item['name']}.stdout.log"
        stderr_path = root / "controller" / f"{item['name']}.stderr.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=stdout, stderr=stderr,
                start_new_session=True,
                env={
                    **os.environ,
                    "PYTHONHASHSEED": "0",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                },
            )
            try:
                code = process.wait(timeout=manifest["deadline_seconds"])
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
                status = "TIMEOUT"
        if code and status == "COMPLETE":
            status = "CRASH"
        worker_path = root / "runs" / item["name"] / "worker.json"
        controller = {
            "case": item["name"],
            "status": status,
            "returncode": code,
            "cold_wall_ns": time.perf_counter_ns() - started,
            "worker_sha256": digest(worker_path.read_bytes()) if worker_path.exists() else None,
            "stdout_sha256": digest(stdout_path.read_bytes()),
            "stderr_sha256": digest(stderr_path.read_bytes()),
        }
        controller_path.parent.mkdir(parents=True, exist_ok=True)
        _put(controller_path, controller)
        print(item["name"], status, flush=True)


def audit(root: Path) -> dict:
    manifest = _manifest(root)
    failures: list[dict] = []
    cells: list[dict] = []
    selected_reasons = Counter()
    count_ratios: dict[int, list[float]] = defaultdict(list)
    family_ratios: dict[str, list[float]] = defaultdict(list)
    selected_fastest = selected_tied = total_cells = 0

    for item in manifest["cases"]:
        folder = root / "runs" / item["name"]
        controller_path = folder / "controller.json"
        if not controller_path.exists():
            failures.append({"case": item["name"], "reason": "MISSING_CONTROLLER"})
            continue
        controller = json.loads(controller_path.read_bytes())
        if controller.get("status") != "COMPLETE":
            failures.append({"case": item["name"], "reason": controller.get("status")})
            continue
        worker_path = folder / "worker.json"
        if digest(worker_path.read_bytes()) != controller.get("worker_sha256"):
            raise ValueError("worker hash differs")
        worker_record = json.loads(worker_path.read_bytes())
        expected = {
            _record_name(count, repeat, mode)
            for count in COUNTS for repeat in range(REPEATS) for mode in MODES
        }
        if set(worker_record["records"]) != expected:
            raise ValueError("worker timing roster differs")

        by_cell: dict[tuple[int, str], list[dict]] = defaultdict(list)
        for name, expected_sha in worker_record["records"].items():
            path = folder / "records" / name
            if digest(path.read_bytes()) != expected_sha:
                raise ValueError("timing-record hash differs")
            row = json.loads(path.read_bytes())
            if row.get("status") != "COMPLETE":
                failures.append({
                    "case": item["name"],
                    "record": name,
                    "reason": row.get("status"),
                })
                continue
            if type(row.get("complete_ns")) is not int or row["complete_ns"] <= 0:
                raise ValueError("invalid timing")
            by_cell[row["count"], row["mode"]].append(row)

        for count in COUNTS:
            if any(len(by_cell[count, mode]) != REPEATS for mode in MODES):
                failures.append({
                    "case": item["name"], "count": count,
                    "reason": "INCOMPLETE_CELL",
                })
                continue
            medians = {
                mode: statistics.median(row["complete_ns"] for row in by_cell[count, mode])
                for mode in MODES
            }
            oracle_mode = min(EXPLICIT, key=lambda mode: medians[mode])
            oracle_ns = medians[oracle_mode]
            tied = {
                mode for mode in EXPLICIT
                if medians[mode] <= 1.02 * oracle_ns
            }
            auto_rows = by_cell[count, "auto"]
            selections = {row["selected_mode"] for row in auto_rows}
            reasons = {row["auto_policy_reason"] for row in auto_rows}
            if len(selections) != 1 or len(reasons) != 1:
                raise ValueError("auto selection changed across identical geometry")
            selected = next(iter(selections))
            reason = next(iter(reasons))
            ratio = medians["auto"] / oracle_ns
            total_cells += 1
            if selected == oracle_mode:
                selected_fastest += 1
            if selected in tied:
                selected_tied += 1
            selected_reasons[selected + ":" + str(reason)] += 1
            count_ratios[count].append(ratio)
            family_ratios[item["family"]].append(ratio)
            cells.append({
                "case": item["name"],
                "family": item["family"],
                "replica": item["replica"],
                "count": count,
                "medians_ns": medians,
                "oracle_mode": oracle_mode,
                "oracle_ns": oracle_ns,
                "oracle_ties_2pct": sorted(tied),
                "auto_selected_mode": selected,
                "auto_policy_reason": reason,
                "auto_oracle_ratio": ratio,
                "selected_is_fastest": selected == oracle_mode,
                "selected_within_2pct_tie": selected in tied,
            })

    ratios = [cell["auto_oracle_ratio"] for cell in cells]
    summary = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "functional_complete": not failures and len(cells) == len(manifest["cases"]) * len(COUNTS),
        "case_count": len(manifest["cases"]),
        "cell_count": len(cells),
        "failures": failures,
        "overall_auto_oracle_geomean": _gmean(ratios) if ratios else None,
        "worst_auto_oracle_ratio": max(ratios) if ratios else None,
        "worst_cell": max(cells, key=lambda cell: cell["auto_oracle_ratio"]) if cells else None,
        "count_geomeans": {
            str(count): _gmean(values) for count, values in sorted(count_ratios.items())
        },
        "family_geomeans": {
            family: _gmean(values) for family, values in sorted(family_ratios.items())
        },
        "selected_actual_fastest_fraction": selected_fastest / total_cells if total_cells else None,
        "selected_within_2pct_tie_fraction": selected_tied / total_cells if total_cells else None,
        "selection_reasons": dict(sorted(selected_reasons.items())),
    }
    payload = {"summary": summary, "cells": cells}
    (root / "audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["functional_complete"]:
        raise SystemExit("selector development audit is incomplete")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--library", type=Path, required=True)
    worker_parser = sub.add_parser("_worker")
    worker_parser.add_argument("--root", type=Path, required=True)
    worker_parser.add_argument("--case", type=int, required=True)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        execute(args.root.resolve(), args.library.resolve())
    elif args.command == "_worker":
        worker(args.root.resolve(), args.case)
    else:
        audit(args.root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
