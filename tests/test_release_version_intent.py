"""Ensure permanent CI cannot publish releases or mutate the repository.

The former version-intent contract exercised the retired
``indexed-efficiency.yml`` publication job.  The lean active tree has no
publication workflow, so its stronger current contract is simpler: every
permanent workflow is read-only and contains no release/package publication
command.  A future release workflow must introduce an explicit, separately
reviewed publication contract rather than inheriting a dead experiment job.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = ROOT / ".github" / "workflows"
EXPECTED_WORKFLOWS = {
    "cpu-regression.yml",
    "exact-public-api.yml",
    "general-cnf.yml",
    "public-package.yml",
    "repository-audit.yml",
    "svm-portability.yml",
    "wap-support-regression.yml",
}
FORBIDDEN_PUBLICATION_TEXT = (
    "gh release",
    "release create",
    "git push",
    "git tag",
    "twine upload",
    "pypa/gh-action-pypi-publish",
    "npm publish",
    "docker push",
)


def workflow_paths() -> list[Path]:
    return sorted(path for path in WORKFLOW_ROOT.glob("*.y*ml") if path.is_file())


def load_workflow(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert type(value) is dict, f"workflow is not a mapping: {path}"
    return value


def permission_maps(value):
    if type(value) is dict:
        for key, child in value.items():
            if key == "permissions":
                yield child
            yield from permission_maps(child)
    elif type(value) is list:
        for child in value:
            yield from permission_maps(child)


def test_permanent_workflow_inventory_is_explicit_and_small():
    observed = {path.name for path in workflow_paths()}
    assert observed == EXPECTED_WORKFLOWS


@pytest.mark.parametrize("path", workflow_paths(), ids=lambda path: path.name)
def test_permanent_workflows_have_no_write_permissions(path: Path):
    workflow = load_workflow(path)
    permissions = list(permission_maps(workflow))
    assert permissions, f"workflow must declare explicit permissions: {path}"
    for declared in permissions:
        assert declared not in ("write-all", "write"), path
        if type(declared) is dict:
            writable = {name: access for name, access in declared.items()
                        if str(access).lower() == "write"}
            assert not writable, f"write permission in {path}: {writable}"


@pytest.mark.parametrize("path", workflow_paths(), ids=lambda path: path.name)
def test_permanent_workflows_contain_no_publication_side_effect(path: Path):
    text = path.read_text(encoding="utf-8").lower()
    found = [token for token in FORBIDDEN_PUBLICATION_TEXT if token in text]
    assert not found, f"publication command in {path}: {found}"
