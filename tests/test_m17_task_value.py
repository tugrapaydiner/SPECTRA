import pytest
import torch

from data.ancestry import digest
from eval.checkable_tasks import TaskSpec
from eval.verified_search import ValueTarget
from model.task_value import architecture, save_task_value, load_task_value
from model.typed_value import TypedStateValue
from scripts.m17_sources import safe_name

SPEC = TaskSpec("maze", 5, 5, 16, 8)
CORE, DATA = "1"*64, "2"*64


def checkpoint(tmp_path, target=ValueTarget.QUALITY):
    model = TypedStateValue(target, **architecture(SPEC)).eval()
    path = tmp_path/"value.pt"
    sha = save_task_value(path, model, spec=SPEC, core_sha256=CORE, training_manifest_sha256=DATA)
    return path, sha


def load(path, sha, **kwargs):
    return load_task_value(path, expected_sha256=sha, expected_core_sha256=kwargs.pop("core", CORE),
        expected_training_manifest_sha256=kwargs.pop("data", DATA), spec=kwargs.pop("spec", SPEC), **kwargs)


def test_task_value_strict_roundtrip_and_freeze(tmp_path):
    path, sha = checkpoint(tmp_path)
    model, contract = load(path, sha)
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert contract.transition_id == SPEC.transition_id
    assert contract.state_schema == SPEC.state_schema
    assert contract.target is ValueTarget.QUALITY
    contract.bind_selection(model_sha256=CORE, transition_id=SPEC.transition_id, state_schema=SPEC.state_schema)


@pytest.mark.parametrize("wrong", ["hash", "core", "data", "task"])
def test_task_value_rejects_wrong_bindings(tmp_path, wrong):
    path, sha = checkpoint(tmp_path)
    kwargs = {}
    if wrong == "hash": sha = "4"*64
    elif wrong in {"core", "data"}: kwargs[wrong] = "5"*64
    else: kwargs["spec"] = TaskSpec("maze", 7, 7, 16, 8)
    with pytest.raises(ValueError): load(path, sha, **kwargs)


def test_improvement_is_diagnostic_never_absolute_selector(tmp_path):
    path, sha = checkpoint(tmp_path, ValueTarget.IMPROVEMENT)
    with pytest.raises(ValueError, match="diagnostic"):
        load(path, sha)
    _, c = load(path, sha, diagnostic_improvement=True)
    with pytest.raises(ValueError, match="not an absolute"):
        c.bind_selection(model_sha256=CORE, transition_id=SPEC.transition_id, state_schema=SPEC.state_schema)


@pytest.mark.parametrize("mutation", ["extra", "nan", "dtype", "missing", "ancestry", "decode"])
def test_content_hash_is_not_a_substitute_for_format_validation(tmp_path, mutation):
    path, _ = checkpoint(tmp_path)
    payload = torch.load(path, weights_only=True)
    key = next(iter(payload["model_state"]))
    if mutation == "extra": payload["extra"] = "unrecognized"
    elif mutation == "nan": payload["model_state"][key].flatten()[0] = float("nan")
    elif mutation == "dtype": payload["model_state"][key] = payload["model_state"][key].double()
    elif mutation == "missing": del payload["model_state"][key]
    elif mutation == "ancestry": payload["data_ancestry"] = {}
    else: payload["task"]["decode_id"] = "different_decoder"
    torch.save(payload, path)
    with pytest.raises((ValueError, RuntimeError)):
        load(path, digest(path.read_bytes()))


def test_saving_does_not_overwrite_existing_checkpoint(tmp_path):
    path, _ = checkpoint(tmp_path)
    with pytest.raises(FileExistsError):
        save_task_value(path, TypedStateValue(ValueTarget.QUALITY, **architecture(SPEC)),
                        spec=SPEC, core_sha256=CORE, training_manifest_sha256=DATA)


@pytest.mark.parametrize("name", ["../a", "/a", "a/../../b", "a\\b", "", "./../a"])
def test_archive_paths_do_not_escape(name):
    with pytest.raises(ValueError): safe_name(name)


def test_canonical_tar_dot_prefix_is_normalized():
    assert safe_name("./experiment/models/test.pt") == "experiment/models/test.pt"
