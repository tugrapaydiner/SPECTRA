"""Proof-carrying bounded repair on a pinned ITC-2007 examination instance.

This is a transfer study for the installed ``spectra.exact`` API, not a claim to
solve the complete capacitated timetabling objective.  It starts from a published
feasible timetable, freezes exams involved in unsupported hard constraints, gives
other exams one deterministic duration-compatible alternative period, and solves
only the shared-student/EXCLUSION conflict core.  Every result is replayed from the
stored SBC1 packet and SCQ2 evidence without trusting the native index.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from typing import Iterable

from spectra.exact import (
    ExactRuntime,
    build_exact_runtime,
    decode_certified_output,
    pack_validated_session_packet,
    verify_session_output,
)
from spectra.exact_proof import decode_proof
from spectra.itc2007_exam import ITCExamInstance, parse_itc2007_exam

SCHEMA = "spectra.itc2007.exam_repair_transfer.v2"
DATASET_REPOSITORY = "davidvanbulck/ITC-2007-CETT"
DATASET_COMMIT = "ed2d4c0cf5302b77a8b321cd0380de0bb579d538"
DEFAULT_INSTANCE = "examtt-instances/Instances/set9.exam"
DEFAULT_SOLUTION = "InitMethods/SolStore/set9_OBSIFeas_0.sol"
MODES = ("auto", "raw", "scc", "route")
WARMUPS = 8
REPEATS = 32
MAX_INPUT_BYTES = 64 << 20
SOURCE_PATHS = (
    Path("pyproject.toml"), Path("README.md"),
    Path("spectra/exact.py"), Path("spectra/exact_problem.py"),
    Path("spectra/exact_proof.py"), Path("spectra/exam_repair.py"),
    Path("spectra/itc2007_exam.py"),
    Path("spectra/_native/exact_binary.cpp"),
    Path("tests/public/test_exam_repair.py"),
    Path("tests/public/test_itc2007_exam.py"),
    Path("scripts/check_exam_repair_installation.py"),
    Path("docs/EXAM_REPAIR.md"), Path("docs/README.md"),
    Path("experiments/compiled_query/exact.py"),
    Path("experiments/compiled_query/exact_native.cpp"),
    Path("experiments/compiled_query/test_exact.py"),
    Path("experiments/compiled_query/itc_exam_transfer.py"),
    Path("experiments/compiled_query/itc_exam_suite.py"),
    Path("experiments/compiled_query/test_itc_exam_transfer.py"),
    Path("experiments/compiled_query/test_itc_exam_suite.py"),
    Path("experiments/compiled_query/ITC_EXAM_TRANSFER_PROTOCOL.md"),
    Path("experiments/compiled_query/ITC_EXAM_SUITE_PROTOCOL.md"),
    Path("experiments/compiled_query/ITC_EXAM_TRANSFER_REVALIDATION_20261019.md"),
    Path("experiments/compiled_query/AUTO_SELECTOR_V10_REVALIDATION_20261019.md"),
    Path(".github/workflows/itc-exam-suite.yml"),
)

def timing_schedule(warmups: int, repeats: int) -> str:
    return (
        f"{warmups} warmups per mode; {repeats} measured repetitions "
        "rotate modes by repetition"
    )


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode()


def write_once(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
    return digest(data)


def deterministic_word(seed: int, *parts: int) -> int:
    payload = seed.to_bytes(8, "little", signed=False)
    for part in parts:
        if type(part) is not int or part < 0:
            raise ValueError("deterministic schedule parts must be nonnegative exact integers")
        payload += part.to_bytes(8, "little", signed=False)
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def read_bound(path: Path, expected: str | None = None,
               *, max_bytes: int = MAX_INPUT_BYTES) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"{path} exceeds the bounded read limit")
    if expected is not None and digest(data) != expected:
        raise ValueError(f"{path} hash differs")
    return data


def parse_solution(text: str, instance: ITCExamInstance) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if type(text) is not str:
        raise ValueError("solution must be an exact string")
    periods: list[int] = []
    rooms: list[int] = []
    for line_number, raw in enumerate(text.replace("\r\n", "\n").replace("\r", "\n").split("\n"), 1):
        line = raw.strip()
        if not line:
            continue
        fields = tuple(value.strip() for value in line.split(","))
        if len(fields) != 2 or any(not value.isascii() or not value.isdigit() for value in fields):
            raise ValueError(f"line {line_number}: solution row must be period,room")
        period, room = map(int, fields)
        if period >= len(instance.periods) or room >= len(instance.rooms):
            raise ValueError(f"line {line_number}: solution index outside instance")
        periods.append(period)
        rooms.append(room)
    if len(periods) != len(instance.exams):
        raise ValueError("solution row count differs from exam count")
    return tuple(periods), tuple(rooms)


def check_full_baseline(instance: ITCExamInstance, periods: tuple[int, ...],
                        rooms: tuple[int, ...]) -> None:
    """Check the published timetable's hard constraints before deriving repairs."""
    if len(periods) != len(instance.exams) or len(rooms) != len(instance.exams):
        raise ValueError("baseline geometry differs")
    roster: dict[int, list[int]] = defaultdict(list)
    for exam_index, exam in enumerate(instance.exams):
        period = periods[exam_index]
        room = rooms[exam_index]
        if instance.periods[period].duration < exam.duration:
            raise ValueError(f"baseline exam {exam_index} exceeds period duration")
        for student in exam.students:
            roster[student].append(exam_index)
        if not 0 <= room < len(instance.rooms):
            raise ValueError("baseline room outside bank")
    for student, exams in roster.items():
        seen: set[int] = set()
        for exam in exams:
            if periods[exam] in seen:
                raise ValueError(f"baseline repeats student {student} in one period")
            seen.add(periods[exam])
    for left, right in instance.exclusions:
        if periods[left] == periods[right]:
            raise ValueError("baseline violates an EXCLUSION constraint")
    for left, right in instance.coincidences:
        if periods[left] != periods[right]:
            raise ValueError("baseline violates an EXAM_COINCIDENCE constraint")
    for later, earlier in instance.after:
        if periods[later] <= periods[earlier]:
            raise ValueError("baseline violates an AFTER constraint")
    usage: dict[tuple[int, int], int] = defaultdict(int)
    occupants: dict[tuple[int, int], list[int]] = defaultdict(list)
    for exam_index, exam in enumerate(instance.exams):
        key = (periods[exam_index], rooms[exam_index])
        usage[key] += len(exam.students)
        occupants[key].append(exam_index)
    for (period, room), count in usage.items():
        if count > instance.rooms[room].capacity:
            raise ValueError(f"baseline room capacity exceeded at period {period}, room {room}")
    for exam in instance.room_exclusive:
        if len(occupants[periods[exam], rooms[exam]]) != 1:
            raise ValueError("baseline violates ROOM_EXCLUSIVE")


def protected_exams(instance: ITCExamInstance) -> set[int]:
    protected = set(instance.room_exclusive)
    for left, right in (*instance.after, *instance.coincidences):
        protected.add(left)
        protected.add(right)
    return protected


def derive_candidates(instance: ITCExamInstance, baseline_periods: tuple[int, ...],
                      *, seed: int) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Choose one deterministic alternative while retaining the baseline witness."""
    protected = protected_exams(instance)
    rows: list[tuple[int, ...]] = []
    flexible: list[int] = []
    max_room = max((room.capacity for room in instance.rooms), default=0)
    for exam_index, exam in enumerate(instance.exams):
        base = baseline_periods[exam_index]
        if exam_index in protected:
            rows.append((base,))
            continue
        alternatives = [
            period for period, record in enumerate(instance.periods)
            if period != base and record.duration >= exam.duration
            and len(exam.students) <= max_room
        ]
        if not alternatives:
            rows.append((base,))
            continue
        # Hash-shuffled choice avoids concentrating all repairs in the earliest
        # periods while remaining fixed by the frozen source/solution identity.
        alternatives.sort(key=lambda period: hashlib.sha256(
            f"{seed}/{exam_index}/{period}".encode("ascii")).digest())
        alternate = alternatives[0]
        rows.append((base, alternate))
        flexible.append(exam_index)
    return tuple(rows), tuple(flexible)


def generate_queries(core, baseline_periods: tuple[int, ...], flexible: tuple[int, ...],
                     *, target: int, seed: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    if type(target) is not int or not 1 <= target <= 100_000:
        raise ValueError("query target outside [1, 100000]")
    model = core.model
    names = model.period_ids
    exam_names = model.exam_ids
    alternatives: dict[int, str] = {}
    for exam in flexible:
        base_name = names[baseline_periods[exam]]
        alternatives[exam] = next(name for name in model.candidates[exam] if name != base_name)
    queries: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()

    def add(query: tuple[tuple[int, int], ...]) -> None:
        if query not in seen:
            seen.add(query)
            queries.append(query)

    add(())
    for period in names:
        add(model.freeze_event(closed_periods=(period,)))
    for exam in flexible:
        add(model.freeze_event(fixed=((exam_names[exam], alternatives[exam]),)))
    attempts = 0
    while len(queries) < target and attempts < target * 200:
        attempts += 1
        if not flexible:
            break
        maximum = min(12, len(flexible))
        width = (2 + deterministic_word(seed, attempts, 0) % (maximum - 1)
                 if maximum >= 2 else 1)
        ranked = sorted(
            flexible,
            key=lambda exam: (deterministic_word(seed, attempts, 1, exam), exam),
        )
        chosen = tuple(sorted(ranked[:width]))
        fixed = tuple((exam_names[exam], alternatives[exam]) for exam in chosen)
        if attempts % 3 == 0:
            closed = (names[deterministic_word(seed, attempts, 2) % len(names)],)
            query = model.freeze_event(closed_periods=closed, fixed=fixed)
        else:
            query = model.freeze_event(fixed=fixed)
        add(query)
    if len(queries) < target:
        raise ValueError(f"could create only {len(queries)} distinct repair events")
    return tuple(queries[:target])


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), text=True).strip()


def source_inventory() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {str(path): digest((root / path).read_bytes()) for path in SOURCE_PATHS}


def validate_source_commit(commit: str, inventory: dict[str, str]) -> str:
    if (type(commit) is not str or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)):
        raise ValueError("source_commit must be a full lowercase commit SHA")
    resolved = _git("rev-parse", "--verify", commit + "^{commit}")
    if resolved != commit:
        raise ValueError("source_commit must identify the commit itself")
    for path, expected in inventory.items():
        try:
            data = subprocess.check_output(("git", "show", f"{commit}:{path}"))
        except subprocess.CalledProcessError as error:
            raise ValueError(f"source commit does not contain {path}") from error
        if digest(data) != expected:
            raise ValueError(f"working source differs from source_commit at {path}")
    return _git("rev-parse", commit + "^{tree}")


def freeze(root: Path, instance_path: Path, solution_path: Path, *, query_count: int,
           license_path: Path | None = None, source_commit: str | None = None,
           dataset_instance_path: str = DEFAULT_INSTANCE,
           dataset_solution_path: str = DEFAULT_SOLUTION,
           evaluation_role: str = "development") -> dict:
    root = root.resolve()
    if root.exists():
        raise FileExistsError(root)
    if license_path is None:
        raise ValueError("freeze requires the upstream redistribution license")
    if (type(dataset_instance_path) is not str or not dataset_instance_path
            or type(dataset_solution_path) is not str or not dataset_solution_path):
        raise ValueError("dataset paths must be nonempty exact strings")
    if evaluation_role not in ("development", "holdout", "replication"):
        raise ValueError("evaluation_role must be development, holdout or replication")
    instance_bytes = read_bound(instance_path)
    solution_bytes = read_bound(solution_path)
    license_bytes = read_bound(license_path)
    instance = parse_itc2007_exam(instance_bytes.decode("utf-8"))
    baseline_periods, baseline_rooms = parse_solution(solution_bytes.decode("utf-8"), instance)
    check_full_baseline(instance, baseline_periods, baseline_rooms)
    seed = int.from_bytes(hashlib.sha256(instance_bytes + b"\0" + solution_bytes).digest()[:8], "little")
    if len(instance.periods) > 64:
        raise ValueError("selected ITC instance exceeds the exact API's 64-period palette")
    candidates, flexible = derive_candidates(instance, baseline_periods, seed=seed)
    core = instance.conflict_core(candidates, acknowledge_ignored_constraints=True)
    if not core.model.problem.check(bytes(baseline_periods)):
        raise ValueError("published baseline does not witness the derived conflict core")
    queries = generate_queries(core, baseline_periods, flexible, target=query_count, seed=seed)
    packet = pack_validated_session_packet(core.model.problem, queries)
    inventory = source_inventory()
    commit = source_commit or _git("rev-parse", "HEAD")
    source_tree: str | None = None
    if source_commit is None:
        changed = _git("status", "--porcelain", "--", *(str(path) for path in SOURCE_PATHS))
        if changed:
            raise ValueError("commit the transfer sources before freezing evidence")
        source_tree = validate_source_commit(commit, inventory)
    elif type(source_commit) is not str or not source_commit:
        raise ValueError("source_commit fixture identity must be a nonempty string")

    state = {
            "baseline_periods": baseline_periods,
            "baseline_rooms": baseline_rooms,
            "candidate_periods": candidates,
            "flexible_exams": flexible,
            "protected_exams": tuple(sorted(protected_exams(instance))),
            "seed": seed,
    }
    staging = root.with_name(root.name + ".partial")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        artifacts = {
            "inputs/instance.exam": write_once(staging / "inputs/instance.exam", instance_bytes),
            "inputs/baseline.sol": write_once(staging / "inputs/baseline.sol", solution_bytes),
            "inputs/UPSTREAM_LICENSE.txt": write_once(
                staging / "inputs/UPSTREAM_LICENSE.txt", license_bytes
            ),
            "candidates.json": write_once(staging / "candidates.json", canonical_json(state)),
            "queries.json": write_once(staging / "queries.json", canonical_json(queries)),
            "session.sbc1": write_once(staging / "session.sbc1", packet),
        }
        manifest = {
            "schema": SCHEMA,
            "classification": (
                "published ITC-2007 input and feasible schedule; deterministic bounded "
                "conflict-core transfer, not full timetabling or outside-team replication"
            ),
            "source_commit": commit,
            "source_tree": source_tree,
            "source_inventory": inventory,
            "dataset_repository": DATASET_REPOSITORY,
            "dataset_commit": DATASET_COMMIT,
            "dataset_instance_path": dataset_instance_path,
            "dataset_solution_path": dataset_solution_path,
            "evaluation_role": evaluation_role,
            "instance_sha256": digest(instance_bytes),
            "solution_sha256": digest(solution_bytes),
            "upstream_acknowledgment": (
                "This product includes software/data distributed by David Van Bulck, "
                "Dries Goossens, and Andrea Schaerf, Ghent University, Belgium "
                "(https://github.com/davidvanbulck/ITC-2007-CETT.git)."
            ),
            "upstream_license_sha256": digest(license_bytes),
            "model": core.info,
            "candidate_policy": (
                "published baseline; freeze AFTER/coincidence/room-exclusive exams; "
                "one hash-selected duration/individual-room-compatible alternative otherwise"
            ),
            "query_policy": (
                "base, every single-period closure, every single-exam move, then seeded "
                "multi-move and closure-plus-move events"
            ),
            "query_count": len(queries),
            "modes": MODES,
            "warmups": WARMUPS,
            "repeats": REPEATS,
            "timing_schedule": timing_schedule(WARMUPS, REPEATS),
            "timing_scope": (
                "ready immutable SBC1 through construction, decisions, full evidence, "
                "independent native check, SCQ2 materialization and disposal"
            ),
            "artifacts": artifacts,
        }
        manifest_bytes = canonical_json(manifest)
        write_once(staging / "manifest.json", manifest_bytes)
        write_once(staging / "manifest.sha256", (digest(manifest_bytes) + "\n").encode())
        staging.rename(root)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_manifest(root: Path) -> dict:
    expected = (root / "manifest.sha256").read_text().strip()
    manifest_bytes = read_bound(root / "manifest.json", expected)
    manifest = json.loads(manifest_bytes)
    if (manifest.get("schema") != SCHEMA or tuple(manifest.get("modes", ())) != MODES
            or type(manifest.get("warmups")) is not int or manifest["warmups"] < 0
            or type(manifest.get("repeats")) is not int or manifest["repeats"] < 1
            or type(manifest.get("query_count")) is not int or manifest["query_count"] < 1
            or manifest.get("timing_schedule")
            != timing_schedule(manifest["warmups"], manifest["repeats"])):
        raise ValueError("transfer manifest schema, modes or schedule differ")
    if (manifest.get("dataset_repository") != DATASET_REPOSITORY
            or manifest.get("dataset_commit") != DATASET_COMMIT
            or type(manifest.get("dataset_instance_path")) is not str
            or not manifest["dataset_instance_path"]
            or type(manifest.get("dataset_solution_path")) is not str
            or not manifest["dataset_solution_path"]
            or manifest.get("evaluation_role") not in
            ("development", "holdout", "replication")):
        raise ValueError("transfer dataset identity differs")
    inventory = manifest.get("source_inventory")
    if type(inventory) is not dict or inventory != source_inventory():
        raise ValueError("transfer source inventory differs")
    if manifest.get("source_tree") is not None:
        if validate_source_commit(manifest.get("source_commit"), inventory) != manifest["source_tree"]:
            raise ValueError("transfer source tree differs")
    required = {"inputs/instance.exam", "inputs/baseline.sol", "candidates.json",
                "queries.json", "session.sbc1", "inputs/UPSTREAM_LICENSE.txt"}
    inventory = set(manifest.get("artifacts", ()))
    if inventory != required:
        raise ValueError("transfer manifest artifact inventory differs")
    for name, checksum in manifest["artifacts"].items():
        read_bound(root / name, checksum)
    if manifest["artifacts"]["inputs/instance.exam"] != manifest.get("instance_sha256"):
        raise ValueError("manifest instance identity differs")
    if manifest["artifacts"]["inputs/baseline.sol"] != manifest.get("solution_sha256"):
        raise ValueError("manifest solution identity differs")
    if (manifest["artifacts"]["inputs/UPSTREAM_LICENSE.txt"]
            != manifest.get("upstream_license_sha256")):
        raise ValueError("manifest upstream-license identity differs")
    return manifest


def reconstruct(root: Path, manifest: dict):
    instance_bytes = read_bound(
        root / "inputs/instance.exam", manifest["artifacts"]["inputs/instance.exam"]
    )
    solution_bytes = read_bound(
        root / "inputs/baseline.sol", manifest["artifacts"]["inputs/baseline.sol"]
    )
    instance = parse_itc2007_exam(instance_bytes.decode("utf-8"))
    baseline_periods, baseline_rooms = parse_solution(solution_bytes.decode("utf-8"), instance)
    check_full_baseline(instance, baseline_periods, baseline_rooms)
    seed = int.from_bytes(
        hashlib.sha256(instance_bytes + b"\0" + solution_bytes).digest()[:8], "little"
    )
    candidates, flexible = derive_candidates(instance, baseline_periods, seed=seed)
    expected_state = {
        "baseline_periods": baseline_periods,
        "baseline_rooms": baseline_rooms,
        "candidate_periods": candidates,
        "flexible_exams": flexible,
        "protected_exams": tuple(sorted(protected_exams(instance))),
        "seed": seed,
    }
    state = json.loads(read_bound(
        root / "candidates.json", manifest["artifacts"]["candidates.json"]
    ))
    if canonical_json(state) != canonical_json(expected_state):
        raise ValueError("stored candidate policy differs from original inputs")
    core = instance.conflict_core(candidates, acknowledge_ignored_constraints=True)
    if core.info != manifest.get("model"):
        raise ValueError("stored model receipt differs from reconstructed inputs")
    queries = tuple(tuple(tuple(pin) for pin in row) for row in json.loads(read_bound(
        root / "queries.json", manifest["artifacts"]["queries.json"]
    )))
    expected_queries = generate_queries(
        core, baseline_periods, flexible,
        target=manifest["query_count"], seed=seed,
    )
    if queries != expected_queries:
        raise ValueError("stored query bank differs from deterministic policy")
    packet = read_bound(root / "session.sbc1", manifest["artifacts"]["session.sbc1"])
    if pack_validated_session_packet(core.model.problem, queries) != packet:
        raise ValueError("stored packet differs from reconstructed original inputs")
    if not core.model.problem.check(bytes(baseline_periods)):
        raise ValueError("stored baseline no longer witnesses the reconstructed core")
    return instance, state, core, queries


def run(root: Path) -> dict:
    root = root.resolve()
    manifest = load_manifest(root)
    reconstruct(root, manifest)
    if (root / "run.json").exists() or (root / "outputs").exists() or (root / "native-build").exists():
        raise FileExistsError("transfer execution already exists")
    packet = read_bound(root / "session.sbc1", manifest["artifacts"]["session.sbc1"])
    build = root / "native-build"
    library = build_exact_runtime(build)
    runtime = ExactRuntime(library)
    states: dict[str, dict[str, object]] = {
        mode: {"elapsed_ns": [], "output": None, "metadata": None} for mode in MODES
    }
    execution_order: list[list[str]] = []

    def execute(mode: str, *, measured: bool) -> None:
        start = time.perf_counter_ns()
        result = runtime.solve_session_packet(packet, mode=mode)
        elapsed = time.perf_counter_ns() - start
        if not verify_session_output(packet, result.output):
            raise AssertionError(f"{mode} output failed standard-library replay")
        current = {
            "query_count": result.query_count,
            "sat_count": result.sat_count,
            "unsat_count": result.unsat_count,
            "proof_bytes": result.proof_bytes,
            "visited": result.visited,
            "arcs_scanned": result.arcs_scanned,
            "selected_mode": result.info.get("selected_mode", mode),
            "bank_query_count": result.info.get("bank_query_count"),
            "bank_pin_count": result.info.get("bank_pin_count"),
            "bank_kernel": result.info.get("bank_kernel"),
            "bank_tiles": result.info.get("bank_tiles"),
            "auto_policy": result.info.get("auto_policy"),
            "auto_policy_reason": result.info.get("auto_policy_reason"),
        }
        if (current["query_count"] != manifest["query_count"]
                or current["sat_count"] + current["unsat_count"] != manifest["query_count"]
                or current["bank_query_count"] != manifest["query_count"]
                or any(type(current[name]) is not int or current[name] < 0
                       for name in ("proof_bytes", "visited", "arcs_scanned",
                                    "bank_pin_count", "bank_tiles"))
                or current["selected_mode"] not in ("raw", "scc", "route")
                or current["bank_kernel"] not in ("serial", "tile64")
                or current["auto_policy"] not in ("explicit", "structural-v10")
                or type(current["auto_policy_reason"]) is not str
                or (mode == "auto" and current["auto_policy"] != "structural-v10")
                or (mode != "auto" and (current["selected_mode"] != mode
                                         or current["auto_policy"] != "explicit"))):
            raise AssertionError(f"{mode} returned inconsistent metadata")
        state = states[mode]
        if state["output"] is None:
            state["output"], state["metadata"] = result.output, current
        elif state["output"] != result.output or state["metadata"] != current:
            raise AssertionError(f"{mode} result changed across identical calls")
        if measured:
            elapsed_rows = state["elapsed_ns"]
            assert isinstance(elapsed_rows, list)
            elapsed_rows.append(elapsed)

    for _ in range(manifest["warmups"]):
        for mode in MODES:
            execute(mode, measured=False)
    for repeat in range(manifest["repeats"]):
        offset = repeat % len(MODES)
        order = (*MODES[offset:], *MODES[:offset])
        execution_order.append(list(order))
        for mode in order:
            execute(mode, measured=True)

    records: dict[str, object] = {
        "schema": SCHEMA + ".run",
        "manifest_sha256": (root / "manifest.sha256").read_text().strip(),
        "build_receipt_sha256": digest((build / "build.json").read_bytes()),
        "warmups": manifest["warmups"],
        "execution_order": execution_order,
        "modes": {},
    }
    for mode in MODES:
        state = states[mode]
        output = state["output"]
        metadata = state["metadata"]
        elapsed = state["elapsed_ns"]
        assert isinstance(output, bytes) and isinstance(metadata, dict) and isinstance(elapsed, list)
        if len(elapsed) != manifest["repeats"]:
            raise AssertionError(f"{mode} timing count differs")
        output_name = f"outputs/{mode}.scq2"
        output_hash = write_once(root / output_name, output)
        records["modes"][mode] = {
            **metadata,
            "elapsed_ns": elapsed,
            "output": output_name,
            "output_sha256": output_hash,
        }
    run_bytes = canonical_json(records)
    write_once(root / "run.json", run_bytes)
    write_once(root / "run.sha256", (digest(run_bytes) + "\n").encode())
    return records


def audit(root: Path) -> dict:
    root = root.resolve()
    manifest = load_manifest(root)
    run_hash = (root / "run.sha256").read_text().strip()
    record = json.loads(read_bound(root / "run.json", run_hash))
    if record.get("schema") != SCHEMA + ".run" or set(record.get("modes", ())) != set(MODES):
        raise ValueError("run schema or mode inventory differs")
    if record.get("manifest_sha256") != (root / "manifest.sha256").read_text().strip():
        raise ValueError("run belongs to a different frozen manifest")
    expected_order = []
    for repeat in range(manifest["repeats"]):
        offset = repeat % len(MODES)
        expected_order.append(list((*MODES[offset:], *MODES[:offset])))
    if (record.get("warmups") != manifest.get("warmups")
            or record.get("execution_order") != expected_order):
        raise ValueError("run warmup or rotated timing schedule differs")
    packet = read_bound(root / "session.sbc1", manifest["artifacts"]["session.sbc1"])
    instance, state, core, queries = reconstruct(root, manifest)
    receipt = read_bound(root / "native-build/build.json")
    if digest(receipt) != record.get("build_receipt_sha256"):
        raise ValueError("native build receipt hash differs")
    decisions: tuple[int, ...] | None = None
    summaries = {}
    sample = None
    for mode in MODES:
        item = record["modes"][mode]
        if type(item) is not dict or item.get("output") != f"outputs/{mode}.scq2":
            raise ValueError(f"{mode} output location differs")
        elapsed = item.get("elapsed_ns")
        if (type(elapsed) is not list or len(elapsed) != manifest["repeats"]
                or any(type(value) is not int or value <= 0 for value in elapsed)):
            raise ValueError(f"{mode} timing schedule differs")
        output = read_bound(root / item["output"], item["output_sha256"])
        if not verify_session_output(packet, output):
            raise ValueError(f"{mode} output fails replay")
        rows = decode_certified_output(output, core.model.problem.n, len(queries))
        current = tuple(code for code, _, _ in rows)
        sat_count = sum(current)
        unsat_count = len(current) - sat_count
        proof_bytes = sum(len(proof) for code, _labels, proof in rows if not code)
        selected = item.get("selected_mode")
        policy = item.get("auto_policy")
        reason = item.get("auto_policy_reason")
        if (item.get("query_count") != len(queries)
                or item.get("bank_query_count") != len(queries)
                or item.get("sat_count") != sat_count
                or item.get("unsat_count") != unsat_count
                or item.get("proof_bytes") != proof_bytes
                or any(type(item.get(name)) is not int or item[name] < 0
                       for name in ("visited", "arcs_scanned", "bank_pin_count", "bank_tiles"))
                or item.get("bank_kernel") not in ("serial", "tile64")
                or selected not in ("raw", "scc", "route")
                or policy not in ("explicit", "structural-v2", "structural-v3", "structural-v4", "structural-v5", "structural-v6", "structural-v10")
                or type(reason) is not str
                or (mode == "auto" and policy not in ("structural-v2", "structural-v3", "structural-v4", "structural-v5", "structural-v6", "structural-v10"))
                or (mode != "auto" and (selected != mode or policy != "explicit"))):
            raise ValueError(f"{mode} metadata differs from certified output")
        if decisions is None:
            decisions = current
        elif decisions != current:
            raise ValueError("engine modes disagree on SAT/UNSAT decisions")
        kinds = Counter()
        for index, (code, _labels, proof) in enumerate(rows):
            if not code:
                kind, _, _ = decode_proof(proof, core.model.problem.n)
                kinds[str(kind)] += 1
                if sample is None:
                    sample = {
                        "query_index": index,
                        "mode": mode,
                        "explanation": core.model.explain_unsat(queries[index], proof),
                    }
        summaries[mode] = {
            "median_elapsed_ns": statistics.median(elapsed),
            "minimum_elapsed_ns": min(elapsed),
            "maximum_elapsed_ns": max(elapsed),
            "sat": sat_count,
            "unsat": unsat_count,
            "proof_kinds": dict(kinds),
            "selected_mode": selected,
            "bank_query_count": item["bank_query_count"],
            "bank_pin_count": item["bank_pin_count"],
            "bank_kernel": item["bank_kernel"],
            "bank_tiles": item["bank_tiles"],
            "auto_policy": policy,
            "auto_policy_reason": reason,
            "output_sha256": item["output_sha256"],
        }
    result = {
        "schema": SCHEMA + ".audit",
        "functional_complete": decisions is not None and len(decisions) == manifest["query_count"],
        "source_commit": manifest["source_commit"],
        "dataset_commit": manifest["dataset_commit"],
        "dataset_instance_path": manifest["dataset_instance_path"],
        "dataset_solution_path": manifest["dataset_solution_path"],
        "evaluation_role": manifest["evaluation_role"],
        "instance_sha256": manifest["instance_sha256"],
        "solution_sha256": manifest["solution_sha256"],
        "model": core.info,
        "baseline_valid_full_itc_hard_constraints": True,
        "flexible_exams": len(state["flexible_exams"]),
        "protected_exams": len(state["protected_exams"]),
        "queries": len(queries),
        "warmups": manifest["warmups"],
        "repeats": manifest["repeats"],
        "timing_schedule": manifest["timing_schedule"],
        "modes": summaries,
        "auto_vs_raw_median_ratio": (
            summaries["auto"]["median_elapsed_ns"] / summaries["raw"]["median_elapsed_ns"]
        ),
        "sample_unsat": sample,
        "limitations": [
            "conflict-core bounded repair, not complete ITC-2007 timetabling",
            "room reassignment and soft objective are not solved",
            "project-owned execution, not outside-team replication",
            "ready-packet timing excludes candidate generation, packing and compilation",
        ],
    }
    audit_bytes = canonical_json(result)
    audit_path = root / "audit.json"
    if audit_path.exists():
        expected = (root / "audit.sha256").read_text().strip()
        existing = read_bound(audit_path, expected)
        if existing != audit_bytes:
            raise ValueError("existing audit differs from independent replay")
    else:
        write_once(audit_path, audit_bytes)
        write_once(root / "audit.sha256", (digest(audit_bytes) + "\n").encode())
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "run", "audit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--instance", type=Path)
    parser.add_argument("--solution", type=Path)
    parser.add_argument("--queries", type=int, default=1024)
    parser.add_argument("--license", type=Path)
    parser.add_argument("--dataset-instance-path", default=DEFAULT_INSTANCE)
    parser.add_argument("--dataset-solution-path", default=DEFAULT_SOLUTION)
    parser.add_argument(
        "--evaluation-role", choices=("development", "holdout", "replication"),
        default="development",
    )
    args = parser.parse_args(argv)
    if args.command == "freeze":
        if args.instance is None or args.solution is None:
            parser.error("freeze requires --instance and --solution")
        value = freeze(
            args.root, args.instance, args.solution, query_count=args.queries,
            license_path=args.license,
            dataset_instance_path=args.dataset_instance_path,
            dataset_solution_path=args.dataset_solution_path,
            evaluation_role=args.evaluation_role,
        )
    elif args.command == "run":
        value = run(args.root)
    else:
        value = audit(args.root)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
