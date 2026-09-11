import pytest
from eval.historical_numerics import ordered_core
from eval.checkable_tasks import MAZE11
from model.trm import TRM
from scripts.m17_models import core_architecture, freeze


def test_maze_context_restores_all_overrides_and_preserves_global_functionals():
    import torch.nn.functional as functional
    before=functional.linear
    core=freeze(TRM(**core_architecture(MAZE11)))
    modules=[core.blocks[0].attn,core.blocks[0].ff[0],core.blocks[0].ff[2]]
    with pytest.raises(RuntimeError,match='injected'):
        with ordered_core(core,MAZE11):
            assert all('forward' in m.__dict__ for m in modules)
            assert functional.linear is before
            raise RuntimeError('injected')
    assert all('forward' not in m.__dict__ for m in modules)
    assert functional.linear is before


def test_maze_existing_custom_ff_rejected_without_partial_override():
    core=freeze(TRM(**core_architecture(MAZE11)))
    module=core.blocks[0].ff[2]
    module.forward=lambda x:x
    with pytest.raises(ValueError,match='already-customized'):
        with ordered_core(core,MAZE11):pass
    assert 'forward' not in core.blocks[0].attn.__dict__
    assert 'forward' not in core.blocks[0].ff[0].__dict__
    assert 'forward' in module.__dict__
