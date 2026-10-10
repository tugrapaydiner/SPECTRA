"""Replay every retained local-search path; never regenerate historical times."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.focused_evaluation.analyse import load
from experiments.focused_evaluation.study import call, require, semantic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    _, cases, rows, _ = load(args.evidence)
    lookup = {case["id"]: case for case in cases}
    count = 0
    for row in rows:
        if row["round"] != 0 or row["arm"] == "glucose4":
            continue
        result = call(lookup[row["id"]], row["arm"], row["seed"])
        require(semantic(result) == semantic(row["result"]), f"trajectory changed: {row['id']} / {row['arm']}")
        count += 1
    print(json.dumps({"status": "PASS", "local_paths_replayed": count,
                      "scope": "all four Python arms; historical timing and native paths not reproduced"}, indent=2))


if __name__ == "__main__":
    main()
