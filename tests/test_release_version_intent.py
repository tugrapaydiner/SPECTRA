"""Exercise the actual publication gate against disposable Git histories."""
from pathlib import Path
import os
import subprocess
import sys

import pytest
import yaml

pytestmark = pytest.mark.skipif(sys.version_info < (3, 11), reason="release job uses Python 3.13/tomllib")
ROOT = Path(__file__).resolve().parents[1]


def steps():
    workflow = yaml.safe_load((ROOT / ".github/workflows/indexed-efficiency.yml").read_text())
    return workflow["jobs"]["publish-versioned-evidence"]["steps"]


def run_guard(tmp_path, new_text, *, before_override=None):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()
    git("init", "-q")
    git("config", "user.email", "contract@example.invalid")
    git("config", "user.name", "Disposable Contract")
    (tmp_path / "pyproject.toml").write_text('[project]\nname="spectra"\nversion="0.7.1"\n')
    git("add", "pyproject.toml"); git("commit", "-qm", "base")
    before = git("rev-parse", "HEAD") if before_override is None else before_override
    (tmp_path / "pyproject.toml").write_text(new_text)
    gate = next(step for step in steps() if step.get("id") == "version_intent")
    output = tmp_path / "job-output.txt"
    env = dict(os.environ, BEFORE_SHA=before, GITHUB_OUTPUT=str(output))
    result = subprocess.run(["bash", "-c", gate["run"]], cwd=tmp_path,
                            env=env, text=True, capture_output=True, timeout=20)
    return result, output.read_text() if output.exists() else ""


@pytest.mark.parametrize("suffix", ["", '\n[tool.setuptools.package-data]\n"spectra._native"=["*.cpp"]\n', '\ndependencies=["example==0.7.2"]\n'])
def test_non_version_edits_do_not_request_publication(tmp_path, suffix):
    result, output = run_guard(tmp_path, '[project]\nname="spectra"\nversion="0.7.1"\n' + suffix)
    assert result.returncode == 0, result.stderr
    assert output == "changed=false\n"


def test_explicit_version_change_requests_guarded_publication(tmp_path):
    result, output = run_guard(tmp_path, '[project]\nname="spectra"\nversion="0.7.2"\n')
    assert result.returncode == 0, result.stderr
    assert output == "changed=true\n"


@pytest.mark.parametrize("before", ["0" * 40, "not-a-commit", "f" * 40])
def test_unknown_previous_version_fails_closed(tmp_path, before):
    result, output = run_guard(tmp_path, '[project]\nversion="0.7.2"\n', before_override=before)
    assert result.returncode != 0
    assert "changed=true" not in output


@pytest.mark.parametrize("text", ['[project]\nname="spectra"\n', '[project]\nversion=true\n', '[project]\nversion=""\n'])
def test_malformed_versions_fail_closed(tmp_path, text):
    result, output = run_guard(tmp_path, text)
    assert result.returncode != 0
    assert "changed=true" not in output


def test_every_release_side_effect_is_version_gated():
    sequence = steps()
    index = next(i for i, step in enumerate(sequence) if step.get("id") == "version_intent")
    for step in sequence[index + 1:]:
        assert step.get("if") == "steps.version_intent.outputs.changed == 'true'"
    publish = sequence[-1]["run"]
    assert "Refusing to overwrite an existing release" in publish
    assert "--draft" in publish and "--verify-tag" in publish
    assert sequence[0]["with"]["fetch-depth"] == 0
