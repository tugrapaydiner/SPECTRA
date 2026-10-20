from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise AssertionError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    path = ROOT / "experiments/compiled_query/itc_exam_transfer.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '("explicit", "structural-v10")',
        '("explicit", "structural-v11")',
        "ITC current accepted policy identities",
    )
    text = replace_once(
        text,
        'current["auto_policy"] != "structural-v10"',
        'current["auto_policy"] != "structural-v11"',
        "ITC current automatic policy identity",
    )
    historical = (
        '"explicit", "structural-v2", "structural-v3", "structural-v4", '
        '"structural-v5", "structural-v6", "structural-v10"'
    )
    historical_with_current = historical + ', "structural-v11"'
    text = replace_once(
        text,
        f"policy not in ({historical})",
        f"policy not in ({historical_with_current})",
        "ITC historical and current policy allowlist",
    )
    historical_auto = (
        '"structural-v2", "structural-v3", "structural-v4", '
        '"structural-v5", "structural-v6", "structural-v10"'
    )
    historical_auto_with_current = historical_auto + ', "structural-v11"'
    text = replace_once(
        text,
        f'(mode == "auto" and policy not in ({historical_auto}))',
        f'(mode == "auto" and policy not in ({historical_auto_with_current}))',
        "ITC automatic historical and current allowlist",
    )
    path.write_text(text, encoding="utf-8")

    test_path = ROOT / "experiments/compiled_query/test_itc_exam_transfer.py"
    test = test_path.read_text(encoding="utf-8")
    test = replace_once(
        test,
        '    assert result["modes"]["auto"]["auto_policy"] == "structural-v10"\n',
        '    assert result["modes"]["auto"]["auto_policy"] == "structural-v11"\n',
        "ITC end-to-end policy expectation",
    )
    marker = "def test_freeze_run_and_independent_audit(tmp_path, monkeypatch):\n"
    regression = '''def test_current_auto_policy_identity_is_v11():
    source = Path(transfer.__file__).read_text(encoding="utf-8")
    assert '("explicit", "structural-v11")' in source
    assert 'current["auto_policy"] != "structural-v11"' in source
    assert 'current["auto_policy"] != "structural-v10"' not in source
    assert '"structural-v10", "structural-v11"' in source


'''
    if "def test_current_auto_policy_identity_is_v11" in test:
        raise AssertionError("ITC v11 regression test already exists")
    test = replace_once(
        test,
        marker,
        regression + marker,
        "ITC transfer test insertion point",
    )
    test_path.write_text(test, encoding="utf-8")

    Path(__file__).unlink()


if __name__ == "__main__":
    main()
