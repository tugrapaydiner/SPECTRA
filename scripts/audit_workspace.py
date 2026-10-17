"""Audit the current active checkout without replaying retired evidence archives.

Historical trees are preserved at ``archive/research-history-20261016`` and their
original commits.  The active audit checks only files that can affect the current
package, tests, documentation, or CI.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
GUIDES = (
    "README.md", "CHANGELOG.md", "CONTRIBUTING.md", "SECURITY.md",
    "docs/README.md", "docs/DEVELOPMENT.md", "docs/STATUS.md",
    "docs/API.md", "docs/EXACT_API.md", "docs/EXAM_REPAIR.md", "docs/RELEASING.md",
    "docs/history/README.md", "results/README.md",
)


def _load_active_checker():
    path = ROOT / "scripts" / "check_active_tree.py"
    spec = importlib.util.spec_from_file_location("spectra_active_tree", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load active-tree checker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _without_fences(text: str) -> str:
    return re.sub(r"(?ms)^```[^\n]*\n.*?^```[^\n]*$", "", text)


def audit(root: Path = ROOT) -> dict:
    root = root.resolve(strict=True)
    active = _load_active_checker().check(root)
    links = external = 0
    for name in GUIDES:
        path = root / name
        if not path.is_file():
            raise ValueError(f"missing current guide: {name}")
        for link in re.findall(r"\[[^]\n]*\]\(([^)\s]+)\)", _without_fences(path.read_text(encoding="utf-8"))):
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc:
                external += 1
                continue
            target = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
            if not target.is_relative_to(root) or not target.exists():
                raise ValueError(f"broken or escaping current guide link: {name}: {link}")
            links += 1

    packaged = root / "spectra" / "_native" / "exact_binary.cpp"
    research = root / "experiments" / "compiled_query" / "exact_native.cpp"
    if packaged.read_bytes() != research.read_bytes():
        raise ValueError("packaged and research exact-native sources diverge")

    digest = hashlib.sha256(packaged.read_bytes()).hexdigest()
    return {
        "status": "PASS",
        "scope": "current active checkout; retired evidence is preserved in archive history",
        "active_tree": active,
        "current_guides": len(GUIDES),
        "local_links_checked": links,
        "external_links_not_fetched": external,
        "exact_native_sources_byte_identical": True,
        "exact_native_sha256": digest,
        "history_branch": "archive/research-history-20261016",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
