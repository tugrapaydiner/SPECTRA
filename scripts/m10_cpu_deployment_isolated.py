#!/usr/bin/env python3
"""Run M10 with a fresh, experiment-private Torch C++ extension build cache."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys


def _out_from_argv() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out", default="outputs/m10_cpu_deployment")
    args, _ = parser.parse_known_args()
    return Path(args.out)


def main() -> int:
    out = _out_from_argv()
    build_dir = out.parent / f"{out.name}_native_build_cache"
    existed_before = build_dir.exists()
    shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True, exist_ok=False)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(build_dir.resolve())

    # Import only after selecting a fresh build directory. m10_native.load_extension
    # is not called at module import, so the experiment's first timed call must
    # compile/load against this empty directory rather than a focused-test cache.
    from scripts.m10_cpu_deployment import main as experiment_main

    code = int(experiment_main())
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "isolated": True,
        "build_directory": str(build_dir.resolve()),
        "directory_existed_before_cleanup": bool(existed_before),
        "directory_empty_before_experiment_import": True,
        "purpose": "force cold source build/load timing independent of prior CI focused-test extension cache",
    }
    (out / "native_build_isolation.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
