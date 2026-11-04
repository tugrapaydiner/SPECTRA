"""Offline repository checks run before a release and in CI.

Checks that the package version is declared once and consistently, the test
extras pin the reference SAT wrapper, every local documentation link and heading
anchor resolves, every GitHub Actions step uses an immutable reference, and the
package workflow runs the installed-wheel checks.
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
    "docs/cli.md", "docs/cnf.md", "docs/development.md", "docs/exact.md",
    "docs/migration.md", "docs/sat.md", "docs/svm.md", "benchmarks/README.md",
)
WORKFLOW_REQUIREMENTS = {
    "package.yml": ("check_current_installation.py", "check_exact_installation.py",
                    "check_exam_repair_installation.py", "check_cnf_installation.py",
                    "check_svm_installation.py"),
}
ACTION_USE = re.compile(r"^\s*(?:-\s*)?(?:uses|\"uses\"|'uses'):\s*(.*?)\s*$")
REMOTE_ACTION = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+@[0-9a-f]{40}")
CONTAINER_ACTION = re.compile(r"docker://[^\s@]+@sha256:[0-9a-f]{64}")


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
    if not isinstance(version, str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:(?:a|b|rc)[0-9]+)?(?:\.dev[0-9]+)?", version):
        raise ValueError("expected a supported numeric package version")
    if values != [version]:
        raise ValueError("pyproject.toml and spectra.__version__ disagree")
    return version


def action_references(text: str, name: str) -> list[str]:
    """Require an immutable reference for every workflow ``uses:`` declaration.

    Local actions inherit the checked-out commit. Remote actions and reusable
    workflows need a full 40-character commit SHA; container actions need a digest.
    """
    references: list[str] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        match = ACTION_USE.fullmatch(line)
        if match is None:
            continue
        reference = match.group(1).split(" #", 1)[0].strip()
        if len(reference) >= 2 and reference[0] in "\"'" and reference[-1] == reference[0]:
            reference = reference[1:-1]
        if reference.startswith("./"):
            if ".." in Path(reference).parts or any(c.isspace() for c in reference):
                raise ValueError(f"unsafe local action: {name}:{line_number}: {reference}")
        elif not (REMOTE_ACTION.fullmatch(reference) or CONTAINER_ACTION.fullmatch(reference)):
            raise ValueError(f"action is not immutably pinned: {name}:{line_number}: {reference}")
        references.append(reference)
    return references


REQUIRED_TEST_DEPENDENCIES = {
    "pytest": None,
    "python-sat": "python-sat==1.9.dev15",
}


def _normalise_requirement_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
    if match is None:
        raise ValueError(f"invalid optional dependency requirement: {requirement!r}")
    return re.sub(r"[-_.]+", "-", match.group(1)).lower()


def optional_dependency_groups(root: Path) -> dict[str, list[str]]:
    """Read literal optional-dependency lists without importing build tooling."""
    project = (root / "pyproject.toml").read_text(encoding="utf-8")
    sections = re.findall(
        r"(?ms)^\[project\.optional-dependencies\]\s*\n(.*?)(?=^\[|\Z)",
        project,
    )
    if len(sections) != 1:
        raise ValueError("expected exactly one [project.optional-dependencies] section")
    declared = re.findall(r"(?m)^([A-Za-z0-9_-]+)\s*=", sections[0])
    assignments = re.findall(r"(?ms)^([A-Za-z0-9_-]+)\s*=\s*(\[[^]]*\])", sections[0])
    if len(assignments) != len(declared):
        raise ValueError("optional dependency groups must be literal string lists")
    groups: dict[str, list[str]] = {}
    for name, literal in assignments:
        value = ast.literal_eval(literal)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"optional dependency group {name!r} must be a literal string list")
        if name in groups:
            raise ValueError(f"duplicate optional dependency group: {name}")
        groups[name] = value
    return groups


def check_test_dependencies(root: Path) -> dict[str, int]:
    """Ensure the documented test/development installs can run public contracts."""
    groups = optional_dependency_groups(root)
    for group_name in ("test", "dev"):
        if group_name not in groups:
            raise ValueError(f"missing optional dependency group: {group_name}")
        by_name: dict[str, str] = {}
        for requirement in groups[group_name]:
            name = _normalise_requirement_name(requirement)
            if name in by_name:
                raise ValueError(f"duplicate dependency {name!r} in {group_name} extra")
            by_name[name] = requirement
        for name, exact in REQUIRED_TEST_DEPENDENCIES.items():
            if name not in by_name:
                raise ValueError(f"{group_name} extra is missing required public-test dependency: {name}")
            if exact is not None and re.sub(r"\s+", "", by_name[name]).lower() != exact.lower():
                raise ValueError(
                    f"{group_name} extra must pin {name} as {exact}; found {by_name[name]!r}"
                )
    return {
        "test_extra_requirements": len(groups["test"]),
        "dev_extra_requirements": len(groups["dev"]),
    }


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
    dependency_report = check_test_dependencies(root)
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
    workflows = root / ".github" / "workflows"
    actions = 0
    for path in sorted(workflows.glob("*.y*ml")):
        actions += len(action_references(path.read_text(encoding="utf-8"), path.name))
    for name, required in WORKFLOW_REQUIREMENTS.items():
        text = (workflows / name).read_text(encoding="utf-8")
        if re.search(r"spectra-[0-9]+\.[0-9]+\.[0-9]+[^\s]*\.whl", text):
            raise ValueError(f"hard-coded wheel version in {name}")
        for script in required:
            if script not in text:
                raise ValueError(f"required installed-wheel check missing from {name}: {script}")
    return {"status": "PASS", "version": version, "guides": len(GUIDES),
            **dependency_report, "workflow_actions_checked": actions,
            "local_links_checked": links, "external_links_not_fetched": external}


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
