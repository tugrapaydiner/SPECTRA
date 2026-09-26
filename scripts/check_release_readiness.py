"""Offline release metadata and current-guide checks; not scientific approval.

Historical protocols are not rewritten or required to look like current guides.
Run audit_workspace.py separately with full Git history to verify retained bytes.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
GUIDES = (
    "README.md", "CHANGELOG.md", "CONTRIBUTING.md", "SECURITY.md",
    "docs/README.md", "docs/DEVELOPMENT.md", "docs/STATUS.md",
    "docs/API.md", "docs/RELEASING.md",
)


def versions(root: Path) -> str:
    """Read the two literal version fields without importing package/build code."""
    project = (root / "pyproject.toml").read_text(encoding="utf-8")
    sections = re.findall(r"(?ms)^\[project\]\s*\n(.*?)(?=^\[|\Z)", project)
    if len(sections) != 1:
        raise ValueError("expected exactly one [project] section")
    fields = re.findall(r"(?m)^version\s*=\s*([^\n]+)$", sections[0])
    if len(fields) != 1:
        raise ValueError("expected exactly one literal project version")
    version = ast.literal_eval(fields[0])
    tree = ast.parse((root / "spectra/__init__.py").read_text(encoding="utf-8"))
    values = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "__version__" for t in n.targets)]
    if not isinstance(version, str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[ab]|rc)?[0-9]*", version):
        raise ValueError("expected a supported numeric package version")
    if values != [version]:
        raise ValueError("pyproject.toml and spectra.__version__ disagree")
    return version


def _without_fences(text: str) -> str:
    return re.sub(r"(?ms)^```[^\n]*\n.*?^```[^\n]*$", "", text)


def anchors(path: Path) -> set[str]:
    text = _without_fences(path.read_text(encoding="utf-8"))
    found = set(re.findall(r'(?:id|name)=["\']([^"\']+)["\']', text))
    counts: dict[str, int] = {}
    for heading in re.findall(r"(?m)^#{1,6}\s+(.+?)\s*#*\s*$", text):
        heading = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", heading)
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        count = counts.get(slug, 0)
        found.add(f"{slug}-{count}" if count else slug)
        counts[slug] = count + 1
    return found


def check(root: Path = ROOT) -> dict:
    root = root.resolve(strict=True)
    version = versions(root)
    links = external = 0
    for name in GUIDES:
        path = root / name
        text = _without_fences(path.read_text(encoding="utf-8"))
        for link in re.findall(r"\[[^]\n]*\]\(([^)\s]+)\)", text):
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc:
                external += 1
                continue
            target = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
            if not target.is_relative_to(root) or not target.exists():
                raise ValueError(f"broken or escaping local guide link: {name}: {link}")
            if parsed.fragment and target.suffix == ".md" and unquote(parsed.fragment) not in anchors(target):
                raise ValueError(f"broken local heading anchor: {name}: {link}")
            links += 1
    for name in ("public-package.yml", "indexed-efficiency.yml"):
        text = (root / ".github/workflows" / name).read_text(encoding="utf-8")
        if re.search(r"spectra-[0-9]+\.[0-9]+\.[0-9]+[^\s]*\.whl", text):
            raise ValueError(f"hard-coded wheel version in {name}")
        if "check_current_installation.py" not in text:
            raise ValueError(f"current-wheel check missing from {name}")
    return {"status": "PASS", "version": version, "current_guides": len(GUIDES),
            "local_links_checked": links, "external_links_not_fetched": external,
            "scope": "offline metadata and navigation only; not release or scientific approval"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = check(args.root)
    text = json.dumps(result, indent=2) + "\n"
    if args.out is not None:
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
