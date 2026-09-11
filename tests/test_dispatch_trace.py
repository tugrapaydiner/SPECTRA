import pytest
import torch

from eval.checkable_tasks import SUDOKU_SHIFT
from scripts.m17_models import core_architecture, freeze
from model.trm import TRM
from scripts.probe_replay_dispatch import capture_first_cycle


def test_trace_is_read_only_and_removes_its_hooks():
    core = freeze(TRM(**core_architecture(SUDOKU_SHIFT)))
    inputs = torch.tensor([[0,2,3,4,3,0,1,2,2,1,0,3,4,3,2,0]])
    before = {k:v.clone() for k,v in core.state_dict().items()}
    trace = capture_first_cycle(core, inputs, SUDOKU_SHIFT)
    assert {'embedded','y','z','input','blocks.0.norm1_0'} <= set(trace)
    assert all(torch.equal(before[k],v) for k,v in core.state_dict().items())
    assert all(not m._forward_hooks for m in core.modules())
    trace['input'].zero_()
    assert inputs.count_nonzero() == 12
    assert all(not v.requires_grad for v in trace.values())


def test_trace_removes_hooks_even_when_traced_cycle_raises(monkeypatch):
    core = freeze(TRM(**core_architecture(SUDOKU_SHIFT)))
    original = core.recursive_cycle
    calls = 0
    def cycle(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('injected trace failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(core, 'recursive_cycle', cycle)
    with pytest.raises(RuntimeError, match='injected'):
        capture_first_cycle(core, torch.zeros(1,16,dtype=torch.long), SUDOKU_SHIFT)
    assert all(not m._forward_hooks for m in core.modules())


def test_trace_preserves_preexisting_hooks():
    core = freeze(TRM(**core_architecture(SUDOKU_SHIFT)))
    calls = []
    handle = core.token_embed.register_forward_hook(lambda *args: calls.append(1))
    try:
        capture_first_cycle(core, torch.zeros(1,16,dtype=torch.long), SUDOKU_SHIFT)
        assert len(calls) == 2
        assert handle.id in core.token_embed._forward_hooks
    finally:
        handle.remove()


@pytest.mark.parametrize("spec_name", ["sudoku_shift", "maze"])
def test_decomposed_attention_equals_native_without_changing_trace(spec_name):
    from scripts.probe_replay_dispatch import capture_attention_stages, FAMILY_SPECS
    spec = FAMILY_SPECS[spec_name][0]
    core = freeze(TRM(**core_architecture(spec)))
    trace = capture_first_cycle(core, torch.zeros(2, spec.height*spec.width, dtype=torch.long), spec)
    before = {k:v.clone() for k,v in trace.items()}
    stages = capture_attention_stages(core, trace)
    assert torch.equal(stages['mha_output'], trace['blocks.0.attn_0'])
    assert all(torch.equal(before[k],v) for k,v in trace.items())


def test_late_dispatch_selection_fails_closed():
    from scripts.probe_replay_dispatch import configure_dispatch
    with pytest.raises(RuntimeError, match="before importing"):
        configure_dispatch("original")
    with pytest.raises(ValueError, match="unknown"):
        configure_dispatch("typo")
