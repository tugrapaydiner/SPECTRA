from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    path = ROOT / "experiments/compiled_query/itc_exam_transfer.py"
    text = path.read_text(encoding="utf-8")
    replacements = {
        '("explicit", "structural-v10")': '("explicit", "structural-v11")',
        'current["auto_policy"] != "structural-v10"': (
            'current["auto_policy"] != "structural-v11"'
        ),
    }
    for before, after in replacements.items():
        count = text.count(before)
        if count != 1:
            raise AssertionError(
                f"ITC policy contract match count differs for {before!r}: {count}"
            )
        text = text.replace(before, after, 1)
    path.write_text(text, encoding="utf-8")

    test_path = ROOT / "experiments/compiled_query/test_itc_exam_transfer.py"
    test = test_path.read_text(encoding="utf-8")
    marker = "def test_freeze_run_audit_and_tamper(tmp_path, monkeypatch):\n"
    regression = '''def test_current_auto_policy_identity_is_v11():
    source = Path(transfer.__file__).read_text(encoding="utf-8")
    assert '("explicit", "structural-v11")' in source
    assert 'current["auto_policy"] != "structural-v11"' in source
    assert 'current["auto_policy"] != "structural-v10"' not in source


'''
    if test.count(marker) != 1:
        raise AssertionError("ITC transfer test insertion point differs")
    if "def test_current_auto_policy_identity_is_v11" in test:
        raise AssertionError("ITC v11 regression test already exists")
    test_path.write_text(test.replace(marker, regression + marker, 1), encoding="utf-8")

    Path(__file__).unlink()


if __name__ == "__main__":
    main()
