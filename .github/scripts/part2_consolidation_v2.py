from __future__ import annotations

from pathlib import Path
from textwrap import dedent, indent


ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise AssertionError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def patch_exact_docstrings() -> None:
    policy = """        The default ``auto`` mode uses frozen structural policy v10. Banks of
        at most eight requests on graphs above 512 vertices take the exact raw
        path directly, and fixed contradictions stay raw. Dense high-pin banks
        shorter than 64 requests stay raw. Below 96 vertices, dense high-pin
        banks stay raw only when SCC compression is present; uncompressed long
        banks may use SCC. At exactly 64 requests, dense uncompressed graphs
        whose edge count is at least four times the vertex count stay raw.
        Other feasible banks of at least 64 requests use SCC; banks from eight
        requests use SCC earlier when SCCs compress the literal graph by at least
        fourfold. Selection uses immutable graph/request geometry only, never
        answers or measured timings. ``info['selected_mode']`` and
        ``info['auto_policy_reason']`` expose the decision. Explicit modes keep
        their meaning; the automatic policy is a frozen structural dispatcher.
"""
    marker = (
        "        their meaning; the automatic policy is a frozen structural dispatcher.\n"
    )
    for filename in ("spectra/exact.py", "experiments/compiled_query/exact.py"):
        text = read(filename)
        start = text.index(
            "        The default ``auto`` mode uses structural policy"
        )
        end = text.index(marker, start) + len(marker)
        write(filename, text[:start] + policy + text[end:])


def patch_itc_transfer() -> None:
    filename = "experiments/compiled_query/itc_exam_transfer.py"
    text = read(filename)
    text = replace_once(
        text,
        'SCHEMA = "spectra.itc2007.exam_repair_transfer.v1"\n',
        'SCHEMA = "spectra.itc2007.exam_repair_transfer.v2"\n',
        label="ITC transfer schema",
    )
    text = replace_once(
        text,
        "WARMUPS = 1\nREPEATS = 5\n",
        "WARMUPS = 8\nREPEATS = 32\n",
        label="ITC timing constants",
    )
    text = replace_once(
        text,
        '    Path("experiments/compiled_query/ITC_EXAM_SUITE_PROTOCOL.md"),\n'
        '    Path(".github/workflows/itc-exam-suite.yml"),\n',
        '    Path("experiments/compiled_query/ITC_EXAM_SUITE_PROTOCOL.md"),\n'
        '    Path("experiments/compiled_query/ITC_EXAM_TRANSFER_REVALIDATION_20261019.md"),\n'
        '    Path("experiments/compiled_query/AUTO_SELECTOR_V10_REVALIDATION_20261019.md"),\n'
        '    Path(".github/workflows/itc-exam-suite.yml"),\n',
        label="ITC source inventory",
    )
    marker = "def digest(data: bytes) -> str:\n"
    helper = dedent(
        """\
        def timing_schedule(warmups: int, repeats: int) -> str:
            return (
                f"{warmups} warmups per mode; {repeats} measured repetitions "
                "rotate modes by repetition"
            )


        def digest(data: bytes) -> str:
        """
    )
    text = replace_once(text, marker, helper, label="ITC timing schedule helper")
    text = replace_once(
        text,
        '"timing_schedule": "one warmup per mode; measured modes rotate by repetition",\n',
        '"timing_schedule": timing_schedule(WARMUPS, REPEATS),\n',
        label="ITC frozen timing schedule",
    )
    text = replace_once(
        text,
        'or manifest.get("timing_schedule")\n'
        '            != "one warmup per mode; measured modes rotate by repetition"):\n',
        'or manifest.get("timing_schedule")\n'
        '            != timing_schedule(manifest["warmups"], manifest["repeats"])):\n',
        label="ITC timing schedule validation",
    )
    write(filename, text)

    test_file = "experiments/compiled_query/test_itc_exam_transfer.py"
    test = read(test_file)
    test = replace_once(
        test,
        '    assert manifest["warmups"] == 0 and manifest["repeats"] == 2\n',
        '    assert manifest["warmups"] == 0 and manifest["repeats"] == 2\n'
        '    assert manifest["timing_schedule"] == transfer.timing_schedule(0, 2)\n',
        label="ITC timing test",
    )
    write(test_file, test)


def patch_itc_suite() -> None:
    filename = "experiments/compiled_query/itc_exam_suite.py"
    text = read(filename)
    text = replace_once(
        text,
        "    summaries = []\n    holdout_failures = []\n",
        "    summaries = []\n"
        "    selector_failures = []\n"
        "    holdout_failures = []\n"
        "    development_failures = []\n",
        label="ITC failure buckets",
    )
    text = replace_once(
        text,
        '        if not gate_pass:\n'
        '            holdout_failures.append(item["name"])\n',
        '        if not gate_pass:\n'
        '            selector_failures.append(item["name"])\n'
        '            if item["role"] == "holdout":\n'
        '                holdout_failures.append(item["name"])\n'
        '            else:\n'
        '                development_failures.append(item["name"])\n',
        label="ITC role-aware failure classification",
    )
    text = replace_once(
        text,
        '            "holdout_gate_pass": gate_pass,\n',
        '            "selector_guard_pass": gate_pass,\n'
        '            "holdout_gate_pass": (\n'
        '                gate_pass if item["role"] == "holdout" else None\n'
        '            ),\n',
        label="ITC case gate labels",
    )
    text = replace_once(
        text,
        '        "holdout_gate_pass": not holdout_failures,\n'
        '        "selector_guard_pass": not holdout_failures,\n'
        '        "holdout_failures": holdout_failures,\n',
        '        "holdout_gate_pass": not holdout_failures,\n'
        '        "selector_guard_pass": not selector_failures,\n'
        '        "selector_failures": selector_failures,\n'
        '        "holdout_failures": holdout_failures,\n'
        '        "development_failures": development_failures,\n',
        label="ITC suite gate labels",
    )
    text = replace_once(
        text,
        '            "the gate limits auto-policy overhead relative to the explicit raw control",\n',
        '            "the selector guard covers every case; holdout failures are reported separately",\n'
        '            "the gate limits auto-policy overhead relative to the explicit raw control",\n',
        label="ITC interpretation",
    )
    write(filename, text)

    test_file = "experiments/compiled_query/test_itc_exam_suite.py"
    test = read(test_file)
    test = replace_once(
        test,
        '    assert result["holdout_gate_pass"]\n'
        '    assert result["holdout_failures"] == []\n',
        '    assert result["holdout_gate_pass"]\n'
        '    assert result["selector_guard_pass"]\n'
        '    assert result["selector_failures"] == []\n'
        '    assert result["holdout_failures"] == []\n'
        '    assert result["development_failures"] == []\n'
        '    assert result["cases"][0]["selector_guard_pass"]\n'
        '    assert result["cases"][0]["holdout_gate_pass"] is None\n'
        '    assert all(\n'
        '        case["holdout_gate_pass"] is True\n'
        '        for case in result["cases"][1:]\n'
        '    )\n',
        label="ITC audit contract",
    )
    write(test_file, test)


def patch_selector_benchmark() -> None:
    filename = "experiments/compiled_query/auto_selector_benchmark.py"
    text = read(filename)
    text = replace_once(
        text,
        '"""Frozen development and confirmation benchmark for structural auto policy v10."""',
        '"""Position-balanced, calibrated benchmark for structural auto policy v10."""',
        label="benchmark module identity",
    )
    text = replace_once(
        text,
        'WARMUPS = 2\n'
        'SCHEMA = "spectra.auto_selector_benchmark.v7"\n'
        'POLICY = "structural-v10"\n',
        'WARMUPS = 2\n'
        'MIN_SAMPLE_NS = 1_000_000\n'
        'MAX_BATCH_SIZE = 256\n'
        'SCHEMA = "spectra.auto_selector_benchmark.v8"\n'
        'POLICY = "structural-v10"\n',
        label="benchmark sampling constants",
    )
    marker = "def run(library: Path, panel: str, destination: Path, repeats: int) -> dict:\n"
    helpers = dedent(
        """\
        def _time_batch(
            runtime: ExactRuntime,
            problem: ExactProblem,
            rows,
            mode: str,
            batch_size: int,
        ):
            if batch_size < 1:
                raise ValueError("batch size must be positive")
            current = b""
            answer = None
            started = time.perf_counter_ns()
            for _ in range(batch_size):
                current = pack_session_packet(problem, rows)
                answer = runtime.solve_session_packet(current, mode=mode)
            elapsed = time.perf_counter_ns() - started
            if answer is None:
                raise AssertionError("timed batch produced no answer")
            return elapsed, current, answer


        def _calibrate_batch(
            runtime: ExactRuntime,
            problem: ExactProblem,
            rows,
            mode: str,
        ) -> int:
            batch_size = 1
            for _ in range(8):
                elapsed, current, answer = _time_batch(
                    runtime, problem, rows, mode, batch_size
                )
                if not verify_session_output(current, answer.output):
                    raise AssertionError("calibration output replay failed")
                if elapsed >= MIN_SAMPLE_NS or batch_size >= MAX_BATCH_SIZE:
                    return batch_size
                target = math.ceil(
                    batch_size * MIN_SAMPLE_NS / max(elapsed, 1)
                )
                batch_size = min(
                    MAX_BATCH_SIZE, max(batch_size + 1, target)
                )
            return batch_size


        def run(library: Path, panel: str, destination: Path, repeats: int) -> dict:
        """
    )
    text = replace_once(text, marker, helpers, label="benchmark batch helpers")
    text = replace_once(
        text,
        '    if repeats < 9 or repeats % 2 == 0:\n'
        '        raise ValueError("odd repeats >= 9 required")\n',
        '    if repeats < 16 or repeats % len(MODES) != 0:\n'
        '        raise ValueError("repeats must be a multiple of four and at least 16")\n',
        label="benchmark balanced repeat contract",
    )

    start = text.index("                samples = {mode: [] for mode in MODES}\n")
    end = text.index("                record = {\n", start)
    timing = indent(
        dedent(
            """\
            batch_sizes = {
                mode: _calibrate_batch(runtime, problem, rows, mode)
                for mode in MODES
            }
            samples = {mode: [] for mode in MODES}
            hashes = {mode: set() for mode in MODES}
            execution_order = []
            schedule = random.Random(seed_for(
                "SPECTRA-auto-selector-v10-schedule",
                f"{panel}/{case.name}/{count}",
            ))
            base_order = list(MODES)
            for repetition in range(repeats):
                offset = repetition % len(MODES)
                if offset == 0:
                    schedule.shuffle(base_order)
                order = (*base_order[offset:], *base_order[:offset])
                execution_order.append(list(order))
                for mode in order:
                    elapsed, current, answer = _time_batch(
                        runtime, problem, rows, mode, batch_sizes[mode]
                    )
                    samples[mode].append(elapsed / batch_sizes[mode])
                    hashes[mode].add(hashlib.sha256(answer.output).hexdigest())
                    if answer.output != first[mode].output:
                        raise AssertionError(
                            "timed output differs from the verified warm result"
                        )
                    if not verify_session_output(current, answer.output):
                        raise AssertionError("timed output replay failed")
            if any(len(values) != 1 for values in hashes.values()):
                raise AssertionError("output bytes changed across repetitions")
            medians = {
                mode: statistics.median(values)
                for mode, values in samples.items()
            }
            oracle = min(("raw", "scc", "route"), key=medians.get)
            selection_ratios = [
                selected_ns / oracle_ns
                for selected_ns, oracle_ns in zip(
                    samples[selected], samples[oracle]
                )
            ]
            realized_ratios = [
                auto_ns / oracle_ns
                for auto_ns, oracle_ns in zip(
                    samples["auto"], samples[oracle]
                )
            ]
            auto_raw_ratios = [
                auto_ns / raw_ns
                for auto_ns, raw_ns in zip(
                    samples["auto"], samples["raw"]
                )
            ]
            """
        ),
        " " * 16,
    )
    text = text[:start] + timing + text[end:]
    text = replace_once(
        text,
        '                    "selection_regret": medians[selected] / medians[oracle],\n'
        '                    "realized_regret": medians["auto"] / medians[oracle],\n'
        '                    "auto_raw_ratio": medians["auto"] / medians["raw"],\n'
        '                    "samples_ns": samples,\n',
        '                    "batch_sizes": batch_sizes,\n'
        '                    "execution_order": execution_order,\n'
        '                    "selection_regret": statistics.median(selection_ratios),\n'
        '                    "realized_regret": statistics.median(realized_ratios),\n'
        '                    "auto_raw_ratio": statistics.median(auto_raw_ratios),\n'
        '                    "independent_median_realized_regret": (\n'
        '                        medians["auto"] / medians[oracle]\n'
        '                    ),\n'
        '                    "samples_ns_per_call": samples,\n',
        label="benchmark paired metrics",
    )
    text = replace_once(
        text,
        '        "repeats": repeats,\n',
        '        "repeats": repeats,\n'
        '        "sampling": {\n'
        '            "method": (\n'
        '                "position-balanced median of paired calibrated-batch "\n'
        '                "per-call samples"\n'
        '            ),\n'
        '            "minimum_sample_ns": MIN_SAMPLE_NS,\n'
        '            "maximum_batch_size": MAX_BATCH_SIZE,\n'
        '        },\n',
        label="benchmark sampling receipt",
    )
    text = replace_once(
        text,
        '    parser.add_argument("--repeats", type=int, default=17)\n',
        '    parser.add_argument("--repeats", type=int, default=32)\n',
        label="benchmark CLI repeats",
    )
    write(filename, text)


def patch_validation_workflows() -> None:
    expression = "${" + "{ github.sha }}"
    workflows = (
        (
            ".github/workflows/part2-auto-selector-validation.yml",
            "auto-selector-v10-validation",
            "part2-auto-selector-validation",
            ".github/workflows/auto-selector-v10-validation.yml",
        ),
        (
            ".github/workflows/part2-itc-balanced.yml",
            "auto-selector-v10-itc-balanced",
            "part2-itc-balanced",
            ".github/workflows/auto-selector-v10-itc-balanced.yml",
        ),
    )
    for workflow, old_name, new_name, old_path in workflows:
        text = read(workflow)
        text = replace_once(
            text,
            f"name: {old_name}",
            f"name: {new_name}",
            label=f"{workflow} name",
        )
        text = replace_once(
            text,
            "branches: [performance/automatic-engine-selection-v10]",
            "branches: [performance/automatic-engine-selection-final]",
            label=f"{workflow} branch",
        )
        text = replace_once(
            text,
            f"paths: [{old_path}]",
            "paths: [.github/part2-validation-trigger]",
            label=f"{workflow} trigger",
        )
        text = replace_once(
            text,
            f"group: {old_name}",
            f"group: {new_name}",
            label=f"{workflow} concurrency",
        )
        text = replace_once(
            text,
            "SOURCE_COMMIT: 79520b8340ed9e16f8eda836b03d5ac1ede409e8",
            "SOURCE_COMMIT: " + expression,
            label=f"{workflow} source env",
        )
        text = text.replace(
            "'source_commit': '79520b8340ed9e16f8eda836b03d5ac1ede409e8'",
            "'source_commit': '" + expression + "'",
        )
        text = text.replace("--repeats 31", "--repeats 32")
        write(workflow, text)


def write_revalidation_docs() -> None:
    write(
        "experiments/compiled_query/AUTO_SELECTOR_V10_REVALIDATION_20261019.md",
        dedent(
            """\
            # Part 2 automatic selector v10 — consolidation revalidation

            Date: 2026-10-19

            This branch is the single canonical continuation of Part 2. It was
            rebuilt from `main` plus the v10 selector source, tests, protocols,
            and benchmark tools. All pre-consolidation heads remain reachable
            from the dedicated archive commit.

            ## Corrections made before revalidation

            - Public and research API documentation now describes the executable
              v10 policy and its 96-vertex boundary consistently.
            - ITC development failures and true holdout failures are reported
              separately; the all-case selector guard remains strict.
            - The confirmation benchmark keeps every original gate threshold and
              case. It replaces independent sub-10-microsecond medians with
              calibrated batches, complete four-position timing blocks, and
              paired per-repetition regret ratios.
            - The original failed evidence remains archived. It is not
              overwritten, discarded, or relabelled as a pass.

            ## Merge rule

            Draft PR #71 remains unmerged until the canonical source passes the
            complete x86-64 and ARM64 confirmation matrix, WAP guard, paired
            overhead guard, ITC suite, balanced ITC correction, public/native/
            package/platform regressions, and source-identity checks. A focused
            pass never substitutes for a failed full matrix.
            """
        ),
    )
    write(
        "experiments/compiled_query/ITC_EXAM_TRANSFER_REVALIDATION_20261019.md",
        dedent(
            """\
            # ITC exam-transfer timing amendment — 2026-10-19

            The original protocol used one warmup and five measured repetitions.
            That schedule was retained in the archive together with its failed
            set9 development result. Both prospective holdouts passed.

            Revalidation uses eight warmups and 32 measured repetitions. The four
            modes rotate through complete four-position blocks, so each mode
            occupies every timing position equally. No selector rule, dataset,
            query bank, expected mode, gate threshold, timing boundary, output,
            or proof requirement changes. The transfer evidence schema is bumped
            from v1 to v2 so amended evidence cannot be confused with the original
            run.

            This amendment corrects measurement power and position balance; it
            does not waive the strict all-case selector guard or the separately
            reported holdout gate.
            """
        ),
    )


def cleanup_bootstrap_files() -> None:
    for relative in (
        ".github/workflows/part2-consolidation-patch.yml",
        ".github/workflows/part2-consolidation-patch-v2.yml",
        ".github/scripts/part2_consolidation_v2.py",
    ):
        path = ROOT / relative
        if path.exists():
            path.unlink()


def main() -> None:
    patch_exact_docstrings()
    patch_itc_transfer()
    patch_itc_suite()
    patch_selector_benchmark()
    patch_validation_workflows()
    write_revalidation_docs()
    cleanup_bootstrap_files()


if __name__ == "__main__":
    main()
