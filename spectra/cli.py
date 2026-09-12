"""Small, standard-library command interface. Research remains an optional extra."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import sys
from . import __version__


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
    commands.add_parser("doctor", help="Report installation without importing optional ML libraries")
    cnf = commands.add_parser("cnf", help="Classical Boolean CNF tools")
    actions = cnf.add_subparsers(dest="action", required=True)
    solve_parser = actions.add_parser("solve", help="Bounded search; a capped failure is UNKNOWN, not UNSAT")
    solve_parser.add_argument("input", type=Path)
    solve_parser.add_argument("--seed", type=int, default=0)
    solve_parser.add_argument("--max-flips", type=int, default=1024)
    solve_parser.add_argument("--out", type=Path)
    check_parser = actions.add_parser("check", help="Check a complete Boolean witness against original clauses")
    check_parser.add_argument("input", type=Path)
    check_parser.add_argument("witness", type=Path, help="JSON object with a Boolean-list witness field")
    evidence = commands.add_parser("evidence", help="Verify a portable hash/size manifest, not scientific validity")
    evidence.add_argument("root", type=Path)
    evidence.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            _emit({"version": __version__, "python": platform.python_version(),
                   "torch_installed": importlib.util.find_spec("torch") is not None,
                   "cnf_requires_torch": False}, None)
        elif args.command == "evidence":
            from .evidence import verify
            _emit(verify(args.root, json.loads(args.manifest.read_text(encoding="utf-8"))), None)
        else:
            from .cnf import read_dimacs, solve
            with args.input.open(encoding="utf-8") as stream:
                problem = read_dimacs(stream)
            if args.action == "solve":
                if args.out is not None and args.out.exists():
                    raise FileExistsError(f"refusing to replace {args.out}")
                result = solve(problem, seed=args.seed, max_flips=args.max_flips).record()
                result["formula_sha256"] = problem.sha256()
                _emit(result, args.out)
            else:
                record = json.loads(args.witness.read_text(encoding="utf-8"))
                if type(record) is not dict or type(record.get("witness")) is not list:
                    raise ValueError("expected a JSON object containing a Boolean-list witness")
                if "formula_sha256" in record and record["formula_sha256"] != problem.sha256():
                    raise ValueError("witness report belongs to a different formula")
                violated = problem.violated(tuple(record["witness"]))
                _emit({"valid": not violated, "unsatisfied": list(violated),
                       "formula_sha256": problem.sha256()}, None)
                return 0 if not violated else 1
    except (OSError, ValueError, TypeError) as error:
        print(f"spectra: {error}", file=sys.stderr)
        return 2
    return 0
