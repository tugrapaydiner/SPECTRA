"""Unit contracts; real Inductor execution is tested by the compiler workflow."""
import copy
import pytest
import torch
from model.trm import TRM
from deploy.semantic_exit import native_semantic_exit
from spectra.compiler_runtime import CompilerFPSudoku


def model(dim=16,depth=4):
    torch.manual_seed(2712)
    return TRM(dim=dim,num_tokens=5,seq_len=16,n_layers=1,n=1,T=1,N_sup=depth,
               heads=2,max_grid_size=8).eval().requires_grad_(False)


def same(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    assert torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


@pytest.mark.parametrize('dim',[8,16])
@pytest.mark.parametrize('depth',[1,4])
@pytest.mark.parametrize('kind',['blank','solved','partial'])
@torch.inference_mode()
def test_eager_wrapper_preserves_every_state_and_check(dim,depth,kind):
    m=model(dim,depth);r=CompilerFPSudoku(m,mode='eager')
    x=torch.tensor([[1,2,3,4,3,4,1,2,2,1,4,3,4,3,2,1]])
    if kind=='blank':x.zero_()
    if kind=='partial':x[:,::2]=0
    emb,steps=r.trace(x,depth);ref=m.token_embed(x)+m.encode_positions(x,4,4);same(emb,ref)
    y,z=torch.zeros_like(ref),torch.zeros_like(ref)
    for Y,Z,L in steps:
        y,z=m.recursive_cycle(ref,y,z);same(y,Y);same(z,Z);same(m.out_head(y),L)
    a,w=native_semantic_exit(m,x,depth);b,v=r.solve(x,depth);same(a,b);assert w==v


@torch.inference_mode()
def test_snapshot_and_trace_return_storage_are_independent():
    m=model();r=CompilerFPSudoku(m,mode='eager');x=torch.zeros(1,16,dtype=torch.int64)
    before=r.trace(x)
    for p in m.parameters():p.fill_(float('nan'))
    m.train();m.N_sup=100
    after=r.trace(x)
    for a,b in zip(before[1],after[1]):
        for aa,bb in zip(a,b):same(aa,bb)
    after[0].fill_(123);after[1][0][0].fill_(111)
    same(before[0],r.trace(x)[0]);same(before[1][0][0],r.trace(x)[1][0][0])


@pytest.mark.parametrize('kind',['shape','batch','float','negative','too_large','list','budgetbool','budget0','budget5'])
def test_bad_inputs_fail_closed(kind):
    rt=CompilerFPSudoku(model(),mode='eager');x=torch.zeros(1,16,dtype=torch.int64);k=4
    if kind=='shape':x=x[:,:15]
    elif kind=='batch':x=x.repeat(2,1)
    elif kind=='float':x=x.float()
    elif kind=='negative':x[0,0]=-1
    elif kind=='too_large':x[0,0]=5
    elif kind=='list':x=x.tolist()
    elif kind=='budgetbool':k=True
    elif kind=='budget0':k=0
    elif kind=='budget5':k=5
    with pytest.raises((ValueError,RuntimeError,TypeError)):rt.solve(x,k)


@pytest.mark.parametrize('mode',['invalid',None,True])
def test_unknown_mode(mode):
    with pytest.raises(ValueError):CompilerFPSudoku(model(),mode=mode)


def test_compiler_failure_not_silently_replaced(monkeypatch):
    def fail(*args,**kw):raise RuntimeError('deliberate compilation failure')
    monkeypatch.setattr(torch,'compile',fail)
    with pytest.raises(RuntimeError,match='deliberate'):CompilerFPSudoku(model())


def test_default_fullgraph_options_are_explicit(monkeypatch):
    calls=[]
    def capture(module,**kw):calls.append(kw);return module
    monkeypatch.setattr(torch,'compile',capture)
    r=CompilerFPSudoku(model());assert r.identity()['exact_claim'] is False
    assert calls[0]['backend']=='inductor' and calls[0]['fullgraph'] is True
    assert calls[0]['dynamic'] is False and calls[0]['options']['freezing'] is False
    assert calls[0]['options']['compile_threads']==1


def test_max_autotune_rejects_false_freezing_configuration():
    import torch._inductor.config as ic
    with ic.patch(freezing=False),pytest.raises(RuntimeError,match='before importing'):
        CompilerFPSudoku(model(),mode='max-autotune')


def test_frozen_max_autotune_options(monkeypatch):
    import torch._inductor.config as ic
    monkeypatch.setattr(torch,'compile',lambda m,**kw:m)
    with ic.patch(freezing=True):
        r=CompilerFPSudoku(model(),mode='max-autotune');opts=r.identity()['options']
    assert opts['freezing'] is True and opts['max_autotune'] is True


def test_postconstruction_environment_rejected():
    r=CompilerFPSudoku(model(),mode='eager');x=torch.zeros(1,16,dtype=torch.int64)
    with torch.autocast('cpu',dtype=torch.bfloat16),pytest.raises(ValueError):r.solve(x)


@pytest.mark.parametrize('change',['grad','hook','training','dtype'])
def test_source_contract_is_not_bypassed(change):
    m=model();h=None
    if change=='grad':m.alpha_y.requires_grad_(True)
    elif change=='hook':h=m.register_forward_hook(lambda *args:None)
    elif change=='training':m.train()
    else:m.double()
    try:
        with pytest.raises((ValueError,TypeError)):CompilerFPSudoku(m,mode='eager')
    finally:
        if h is not None:h.remove()


def test_error_suppression_cannot_disguise_fallback():
    import torch._dynamo.config as dc
    with dc.patch(suppress_errors=True),pytest.raises(ValueError,match='suppression'):
        CompilerFPSudoku(model())


def test_public_budget_properties_cannot_be_reassigned():
    r=CompilerFPSudoku(model(),mode='eager')
    for key in ('trained_steps','block_count','mode'):
        with pytest.raises(AttributeError):setattr(r,key,100)
