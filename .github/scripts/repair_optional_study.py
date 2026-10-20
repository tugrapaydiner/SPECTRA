from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    path = ROOT / "experiments/compiled_query/test_study.py"
    text = path.read_text(encoding="utf-8")
    before = "    path = tmp_path / 'input.json'; write_case(path, case)\n    preload(arm)\n"
    after = (
        "    path = tmp_path / 'input.json'; write_case(path, case)\n"
        "    if arm.startswith('cudd-'):\n"
        "        pytest.importorskip(\n"
        "            'dd.cudd', reason='Explicit optional backend dd.cudd not installed'\n"
        "        )\n"
        "    preload(arm)\n"
    )
    count = text.count(before)
    if count != 1:
        raise AssertionError(f"study preload boundary match count differs: {count}")
    path.write_text(text.replace(before, after, 1), encoding="utf-8")

    Path(__file__).unlink()


if __name__ == "__main__":
    main()
