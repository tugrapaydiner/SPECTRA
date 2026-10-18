from __future__ import annotations

from pathlib import Path

files = (
    Path(".github/workflows/validate-auto-selector-development.yml"),
    Path(".github/workflows/validate-auto-selector-wap.yml"),
    Path(".github/workflows/validate-auto-selector-v4-itc.yml"),
)

for path in files:
    text = path.read_text(encoding="utf-8")
    text = text.replace("structural-v5", "structural-v6")
    text = text.replace("AUTO_SELECTOR_V5_PROTOCOL_20261018.md",
                        "AUTO_SELECTOR_V6_PROTOCOL_20261018.md")
    text = text.replace("test_auto_selector_v5.py", "test_auto_selector_v6.py")
    text = text.replace("auto-selector-v5", "auto-selector-v6")
    text = text.replace("selector-v5", "selector-v6")
    if path.name == "validate-auto-selector-development.yml":
        text = text.replace("--repeats 17", "--repeats 31")
    if "structural-v5" in text or "AUTO_SELECTOR_V5" in text or "test_auto_selector_v5" in text:
        raise RuntimeError(f"stale v5 workflow identity remains in {path}")
    path.write_text(text, encoding="utf-8")
