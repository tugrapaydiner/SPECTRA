"""Small, standard-library command interface. Research remains an optional extra."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import sys
from . import __version__
from ._json import DEFAULT_MAX_BYTES, load_file

EXACT_FILE_MAX_BYTES = 64 << 20


def _read_binary(path: Path, *, max_bytes: int) -> bytes:
    if type(max_bytes) is not int or not 1 <= max_bytes <= sys.maxsize - 1:
        raise ValueError("max-bytes must be a positive bounded integer")
    with path.open("rb") as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"{path} exceeds max-bytes")
    return data


def _emit(value: dict, destination: Path | None) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if destination is None:
        sys.stdout.write(text)
    else:
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spectra", description="Checked research tools; exact answers, explicit limits.")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    from .svm_stream import add_parser
    add_parser(commands)
    commands.add_parser("doctor", help="Report installation without importing optional ML libraries")
    cnf = commands.add_parser("cnf", help="Classical Boolean CNF tools")
    actions = cnf.add_subparsers(dest="action", required=True)
    solve_parser = actions.add_parser("solve", help="Bounded search; a capped failure is UNKNOWN, not UNSAT")
    solve_parser.add_argument("input", type=Path)
    solve_parser.add_argument("--seed", type=int, default=0)
    solve_parser.add_argument("--max-flips", type=int, default=1024)
    solve_parser.add_argument("--out", type=Path)
    solve_parser.add_argument("--backend", choices=["compact", "indexed", "deductive", "focused"], default="compact",
                              help="focused uses variable age; deductive handles unit/2-SAT constraints; compact is the historical default")
    check_parser = actions.add_parser("check", help="Check a complete Boolean witness against original clauses")
    check_parser.add_argument("input", type=Path)
    check_parser.add_argument("witness", type=Path, help="JSON object with a Boolean-list witness field")
    exact = commands.add_parser("exact", help="Proof-carrying exact binary-domain tools")
    exact_actions = exact.add_subparsers(dest="action", required=True)
    exact_build = exact_actions.add_parser("build", help="Explicitly compile the bundled ABI-3 native runtime")
    exact_build.add_argument("directory", type=Path)
    exact_build.add_argument("--compiler")
    exact_verify = exact_actions.add_parser("verify", help="Replay an SBC1 -> SCQ2 result without native code")
    exact_verify.add_argument("packet", type=Path)
    exact_verify.add_argument("output", type=Path)
    exact_verify.add_argument("--max-bytes", type=int, default=EXACT_FILE_MAX_BYTES)
    evidence = commands.add_parser("evidence", help="Verify a portable hash/size manifest, not scientific validity")
    evidence.add_argument("root", type=Path)
    evidence.add_argument("manifest", type=Path)
    for json_parser in (check_parser, evidence):
        json_parser.add_argument("--max-json-bytes", type=int, default=DEFAULT_MAX_BYTES,
                                 help="maximum witness/manifest bytes (default: 16777216)")
    args = parser.parse_args(argv)
    try:
        if args.command == "svm":
            from .svm_stream import execute
            _emit(execute(args), None)
        elif args.command == "doctor":
            from .exact import EXACT_API_VERSION, EXACT_NATIVE_ABI, SOURCE as EXACT_SOURCE
            _emit({"version": __version__, "python": platform.python_version(),
                   "torch_installed": importlib.util.find_spec("torch") is not None,
                   "cnf_requires_torch": False,
                   "exact_api_version": EXACT_API_VERSION,
                   "exact_native_abi": EXACT_NATIVE_ABI,
                   "exact_source_packaged": EXACT_SOURCE.is_file()}, None)
        elif args.command == "evidence":
            from .evidence import verify
            _emit(verify(args.root, load_file(args.manifest, max_bytes=args.max_json_bytes)), None)
        elif args.command == "exact":
            from .exact import (EXACT_API_VERSION, EXACT_NATIVE_ABI, build_exact_runtime,
                                decode_certified_output, unpack_session_packet,
                                verify_session_output)
            if args.action == "build":
                library = build_exact_runtime(args.directory, compiler=args.compiler)
                record = json.loads(library.with_name("build.json").read_text(encoding="utf-8"))
                _emit({"status": "BUILT", "library": str(library),
                       "build_receipt": str(library.with_name("build.json")),
                       "exact_api_version": EXACT_API_VERSION,
                       "native_abi": EXACT_NATIVE_ABI,
                       "source_sha256": record["source_sha256"],
                       "library_sha256": record["library_sha256"]}, None)
            else:
                packet = _read_binary(args.packet, max_bytes=args.max_bytes)
                output = _read_binary(args.output, max_bytes=args.max_bytes)
                problem, rows = unpack_session_packet(packet)
                decoded = decode_certified_output(output, problem.n, len(rows))
                verified = verify_session_output(packet, output)
                _emit({"verified": verified, "queries": len(decoded),
                       "sat": sum(code == 1 for code, _, _ in decoded),
                       "unsat": sum(code == 0 for code, _, _ in decoded),
                       "exact_api_version": EXACT_API_VERSION,
                       "native_abi": EXACT_NATIVE_ABI}, None)
                return 0 if verified else 1
        else:
            from .cnf import read_dimacs, solve, solve_indexed, solve_deductive, solve_focused
            with args.input.open(encoding="utf-8") as stream:
                problem = read_dimacs(stream)
            if args.action == "solve":
                if args.out is not None and args.out.exists():
                    raise FileExistsError(f"refusing to replace {args.out}")
                solver = {"compact": solve, "indexed": solve_indexed, "deductive": solve_deductive, "focused": solve_focused}[args.backend]
                result = solver(problem, seed=args.seed, max_flips=args.max_flips).record()
                result["backend"] = args.backend
                result["timing_scope"] = "cold_search_including_preparation"
                result["formula_sha256"] = problem.sha256()
                _emit(result, args.out)
            else:
                record = load_file(args.witness, max_bytes=args.max_json_bytes)
                if type(record) is not dict or type(record.get("witness")) is not list:
                    raise ValueError("expected a JSON object containing a Boolean-list witness")
                if "formula_sha256" in record and record["formula_sha256"] != problem.sha256():
                    raise ValueError("witness report belongs to a different formula")
                violated = problem.violated(tuple(record["witness"]))
                _emit({"valid": not violated, "unsatisfied": list(violated),
                       "formula_sha256": problem.sha256()}, None)
                return 0 if not violated else 1
    except (OSError, ValueError, TypeError, RuntimeError, NotImplementedError) as error:
        print(f"spectra: {error}", file=sys.stderr)
        return 2
    return 0
