"""Apply raw-tile64 patch using v1 exact replacements with corrected guards."""
from __future__ import annotations

from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
V1 = runpy.run_path(str(ROOT / "scripts/apply_raw_tile64.py"))
patch_native = V1["patch_native"]
patch_python = V1["patch_python"]
patch_tests = V1["patch_tests"]
replace_once = V1["replace_once"]


def patch_benchmark(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        """    sparse = tuple(
        (u, v)
        for u in range(n)
        for v in range(u + 1, n)
        if rng.random() < 0.012
    )
""",
        """    # Keep the random case bipartite so the base graph is satisfiable;
    # contradictions then come only from the disclosed request generator.
    sparse = tuple(
        (u, v)
        for u in range(n // 2)
        for v in range(n // 2, n)
        if rng.random() < 0.024
    )
""",
        f"{path}: sparse bipartite benchmark",
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    native_paths = [
        ROOT / "spectra/_native/exact_binary.cpp",
        ROOT / "experiments/compiled_query/exact_native.cpp",
    ]
    python_paths = [
        ROOT / "spectra/exact.py",
        ROOT / "experiments/compiled_query/exact.py",
    ]
    if native_paths[0].read_bytes() != native_paths[1].read_bytes():
        raise RuntimeError("native installed/research sources differ before patch")
    # The public and research Python wrappers intentionally use different problem
    # classes/imports, so patch their shared mode surfaces independently.
    for path in native_paths:
        patch_native(path)
    for path in python_paths:
        patch_python(path)
    patch_tests(ROOT / "experiments/compiled_query/test_exact.py")
    patch_benchmark(ROOT / "experiments/compiled_query/raw_tile64_benchmark.py")
    if native_paths[0].read_bytes() != native_paths[1].read_bytes():
        raise RuntimeError("native installed/research sources differ after patch")
    print("raw tile64 v2 patch applied with all exact replacement guards satisfied")


if __name__ == "__main__":
    main()
