"""Apply raw-tile64 v2 and correct the independent public replay import."""
from __future__ import annotations

from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
V2 = runpy.run_path(str(ROOT / "scripts/apply_raw_tile64_v2.py"))


def main() -> None:
    V2["main"]()
    path = ROOT / "experiments/compiled_query/raw_tile64_benchmark.py"
    text = path.read_text(encoding="utf-8")
    old = "from .exact import ExactRuntime, pack_session_packet, verify_session_output\n"
    new = (
        "from .exact import ExactRuntime, pack_session_packet\n"
        "from spectra.exact import verify_session_output\n"
    )
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"benchmark verifier import: expected one match, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("raw tile64 v3 benchmark verifier import applied")


if __name__ == "__main__":
    main()
