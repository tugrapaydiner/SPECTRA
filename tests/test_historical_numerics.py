"""Arithmetic oracles are independent of Torch/MKL reduction implementations."""
import ctypes
import ctypes.util

import numpy as np
import pytest
import torch

from eval.historical_numerics import extension, ordered_core
from eval.checkable_tasks import SUDOKU_SHIFT, MAZE11
from model.trm import TRM
from scripts.m17_models import core_architecture, freeze


def scalar_fma(a,b,initial=0.):
    f=ctypes.CDLL(ctypes.util.find_library('m')).fmaf
    f.argtypes=[ctypes.c_float]*3;f.restype=ctypes.c_float
    s=float(initial)
    for x,y in zip(a,b): s=f(float(x),float(y),s)
    return np.float32(s)


@pytest.mark.parametrize('shape',[(2,3,7,13),(1,2,121,12),(1,3,12,121),(1,1,1,1)])
def test_vector_and_tail_bmm_equal_independent_libm_fma(shape):
    B,M,K,N=shape
    rng=np.random.default_rng(991)
    a=rng.normal(size=(B,M,K)).astype(np.float32)
    b=rng.normal(size=(B,K,N)).astype(np.float32)
    got=extension().ordered_bmm(torch.from_numpy(a),torch.from_numpy(b)).numpy()
    want=np.array([[[scalar_fma(a[t,m],b[t,:,n]) for n in range(N)]
                    for m in range(M)] for t in range(B)],np.float32)
    assert np.array_equal(got,want)


@pytest.mark.parametrize('shape,block', [((3,256,13),128),((2,17,9),8),((1,1,1),1)])
def test_blocked_linear_adds_bias_between_blocks_not_after_all_products(shape,block):
    M,K,N=shape;rng=np.random.default_rng(992)
    a=rng.normal(size=(M,K)).astype(np.float32);w=rng.normal(size=(K,N)).astype(np.float32)
    b=rng.normal(size=N).astype(np.float32)
    want=np.empty((M,N),np.float32)
    for m in range(M):
        for n in range(N):
            total=b[n]
            for k in range(0,K,block):
                total=np.float32(total+scalar_fma(a[m,k:k+block],w[k:k+block,n]))
            want[m,n]=total
    got=extension().blocked_linear(torch.from_numpy(a),torch.from_numpy(w),torch.from_numpy(b),block).numpy()
    assert np.array_equal(got,want)
    if K==256:
        late=np.array([[np.float32(np.float32(scalar_fma(a[m,:128],w[:128,n])+
              scalar_fma(a[m,128:],w[128:,n]))+b[n]) for n in range(N)] for m in range(M)])
        assert np.any(late!=want)  # ensures the fixture detects misplaced bias


@pytest.mark.parametrize('case',['dtype','rank','stride','empty','requires_grad','geometry'])
def test_native_bmm_contract_rejects_malformed_inputs(case):
    a=torch.ones(1,2,3);b=torch.ones(1,3,4)
    if case=='dtype':a=a.double()
    elif case=='rank':a=a[0]
    elif case=='stride':a=torch.ones(1,3,2).transpose(1,2)
    elif case=='empty':a=torch.empty(1,0,3)
    elif case=='requires_grad':a.requires_grad_(True)
    elif case=='geometry':b=torch.ones(2,3,4)
    with pytest.raises(RuntimeError):extension().ordered_bmm(a,b)


@pytest.mark.parametrize('block',[0,-1,4])
def test_native_linear_rejects_invalid_reduction_block(block):
    with pytest.raises(RuntimeError,match='reduction block'):
        extension().blocked_linear(torch.ones(2,3),torch.ones(3,4),torch.ones(4),block)


@pytest.mark.parametrize('spec',[SUDOKU_SHIFT,MAZE11])
def test_scope_preserves_state_and_restores_only_its_instance_override(spec):
    core=freeze(TRM(**core_architecture(spec)))
    before={k:v.clone() for k,v in core.state_dict().items()}
    module=core.blocks[0].ff[2] if spec.dim==64 else core.blocks[0].attn
    with torch.no_grad(),ordered_core(core,spec):
        assert 'forward' in module.__dict__
        x=torch.zeros(32,core.seq_len,dtype=torch.long)
        e=core.token_embed(x)+core.encode_positions(x,spec.height,spec.width)
        y,z=core.recursive_cycle(e,torch.zeros_like(e),torch.zeros_like(e))
        assert torch.isfinite(y).all() and torch.isfinite(z).all()
        with pytest.raises(ValueError,match='already-customized'):
            with ordered_core(core,spec):pass
    assert 'forward' not in module.__dict__
    assert all(torch.equal(v,core.state_dict()[k]) for k,v in before.items())
    with pytest.raises(RuntimeError,match='injected'):
        with ordered_core(core,spec):raise RuntimeError('injected')
    assert 'forward' not in module.__dict__


def test_replay_rejects_gradients_wrong_batch_nonfinite_and_masked_attention():
    core=freeze(TRM(**core_architecture(SUDOKU_SHIFT)))
    with ordered_core(core,SUDOKU_SHIFT):
        with pytest.raises(ValueError,match='no_grad'):
            core.blocks[0].ff[2](torch.zeros(32,16,256))
        with torch.no_grad():
            with pytest.raises(ValueError,match='B=32'):
                core.blocks[0].ff[2](torch.zeros(1,16,256))
            with pytest.raises(ValueError,match='finite'):
                core.blocks[0].ff[2](torch.full((32,16,256),float('nan')))
    core=freeze(TRM(**core_architecture(MAZE11)))
    q=torch.zeros(32,121,48)
    with torch.no_grad(),ordered_core(core,MAZE11):
        for kwargs in ({'need_weights':True},{'need_weights':False,'is_causal':True},
                       {'need_weights':False,'attn_mask':torch.zeros(121,121)}):
            with pytest.raises(ValueError,match='unmasked'):
                core.blocks[0].attn(q,q,q,**kwargs)


def test_unfrozen_core_is_rejected_before_any_override():
    core=TRM(**core_architecture(SUDOKU_SHIFT))
    with pytest.raises(ValueError,match='frozen'):
        with ordered_core(core,SUDOKU_SHIFT):pass
    assert 'forward' not in core.blocks[0].ff[2].__dict__
