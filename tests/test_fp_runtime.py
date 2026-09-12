"""Adversarial contract and bit-pattern tests for the prepared FP graph."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import copy
import pytest
import torch
from torch import nn
from deploy.semantic_exit import native_semantic_exit
from model.trm import TRM
from spectra.fp_runtime import PreparedFPSudoku, load_extension


def model(dim=16,layers=1,depth=4):
    torch.manual_seed(16122)
    return TRM(dim=dim,num_tokens=5,seq_len=16,n_layers=layers,n=1,T=1,N_sup=depth,
               heads=2,max_grid_size=8).eval().requires_grad_(False)


def bits(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))


@pytest.mark.parametrize('dim',[8,16,32,64])
@pytest.mark.parametrize('layers',[1,2])
@pytest.mark.parametrize('kind',['blank','solved','partial'])
@torch.inference_mode()
def test_every_recurrent_state_and_logit_bit(dim,layers,kind):
    m=model(dim,layers);p=PreparedFPSudoku(m)
    x=torch.tensor([[1,2,3,4,3,4,1,2,2,1,4,3,4,3,2,1]],dtype=torch.int64)
    if kind=='blank':x.zero_()
    if kind=='partial':x[:,::2]=0
    emb,states=p.trace(x,4);ref=m.token_embed(x)+m.encode_positions(x,4,4);bits(emb,ref)
    y,z=torch.zeros_like(ref),torch.zeros_like(ref)
    for Y,Z,L in states:
        y,z=m.recursive_cycle(ref,y,z);bits(Y,y);bits(Z,z);bits(L,m.out_head(y))
    a,w=native_semantic_exit(m,x,4);b,v=p.solve(x,4);bits(a,b);assert w==v
    if kind=='solved':assert v['executed_steps']==1 and v['final_semantic']


@torch.inference_mode()
def test_owns_snapshot_and_returns_independent_states():
    m=model();p=PreparedFPSudoku(m);x=torch.zeros((1,16),dtype=torch.int64)
    before=p.trace(x);identity=p.identity()
    for parameter in m.parameters():parameter.fill_(float('nan'))
    m.N_sup=200;m.train()
    after=p.trace(x)
    for aa,bb in zip(before[1],after[1]):
        for a,b in zip(aa,bb):bits(a,b)
    after[1][0][0].fill_(123)
    bits(before[1][0][0],p.trace(x)[1][0][0]);assert p.identity()==identity
    assert p.identity()['owned_tensor_bytes']>0
    with pytest.raises(AttributeError):p.trained_steps=123


@pytest.mark.parametrize('change',[
    'train','child_train','grad','dtype','nonfinite','n','T','ternary','act8','seq','vocab',
    'epsilon','norm_affine','gelu','attention_batch','attention_heads','attention_zero',
    'attention_bias_k','hook','prehook','override','extra_module','max_norm','no_bias',
    'ff_length','subclass','global_hook','fastpath','autocast'])
def test_reject_unsupported_graph(change):
    m=model();handle=None;old=torch.backends.mha.get_fastpath_enabled()
    try:
        if change=='train':m.train()
        elif change=='child_train':m.blocks[0].ff.train()
        elif change=='grad':m.alpha_y.requires_grad_(True)
        elif change=='dtype':m.double()
        elif change=='nonfinite':m.alpha_y.fill_(float('nan'))
        elif change=='n':m.n=2
        elif change=='T':m.T=2
        elif change=='ternary':m.ternary=True
        elif change=='act8':m.act8=True
        elif change=='seq':m.seq_len=81
        elif change=='vocab':m.num_tokens=10
        elif change=='epsilon':m.norm_y.eps=1e-3
        elif change=='norm_affine':m.norm_y=nn.RMSNorm(16,elementwise_affine=False).eval()
        elif change=='gelu':m.blocks[0].ff[1].approximate='tanh'
        elif change=='attention_batch':m.blocks[0].attn.batch_first=False
        elif change=='attention_heads':m.blocks[0].attn.num_heads=1
        elif change=='attention_zero':m.blocks[0].attn.add_zero_attn=True
        elif change=='attention_bias_k':m.blocks[0].attn.bias_k=nn.Parameter(torch.zeros(1,1,16),requires_grad=False)
        elif change=='hook':handle=m.blocks[0].register_forward_hook(lambda *args:None)
        elif change=='prehook':handle=m.register_forward_pre_hook(lambda *args:None)
        elif change=='override':m.recursive_cycle=lambda *args:None
        elif change=='extra_module':m.extra=nn.Identity().eval()
        elif change=='max_norm':m.token_embed.max_norm=1
        elif change=='no_bias':m.out_head.bias=None
        elif change=='ff_length':m.blocks[0].ff.append(nn.Identity().eval())
        elif change=='subclass':
            class Custom(TRM):pass
            m.__class__=Custom
        elif change=='global_hook':handle=nn.modules.module.register_module_forward_hook(lambda *args:None)
        elif change=='fastpath':torch.backends.mha.set_fastpath_enabled(False)
        elif change=='autocast':
            with torch.autocast('cpu',dtype=torch.bfloat16),pytest.raises((ValueError,TypeError)):
                PreparedFPSudoku(m)
            return
        with pytest.raises((ValueError,TypeError,RuntimeError)):PreparedFPSudoku(m)
    finally:
        if handle is not None:handle.remove()
        torch.backends.mha.set_fastpath_enabled(old)


@pytest.mark.parametrize('case',['shape','batch','float','negative','large','budget0','budget5','budgetbool','budgetfloat','list'])
def test_input_and_budget_rejection(case):
    p=PreparedFPSudoku(model());x=torch.zeros(1,16,dtype=torch.int64);k=4
    if case=='shape':x=x[:,:15]
    elif case=='batch':x=x.repeat(2,1)
    elif case=='float':x=x.float()
    elif case=='negative':x[0,0]=-1
    elif case=='large':x[0,0]=5
    elif case=='budget0':k=0
    elif case=='budget5':k=5
    elif case=='budgetbool':k=True
    elif case=='budgetfloat':k=2.0
    elif case=='list':x=x.tolist()
    with pytest.raises((ValueError,RuntimeError,TypeError)):p.solve(x,k)


def test_noncontiguous_input_and_postconstruction_environment():
    p=PreparedFPSudoku(model());x=torch.zeros(1,32,dtype=torch.int64)[:,::2]
    bits(p.solve(x)[0],p.solve(x.contiguous())[0])
    old=torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        with pytest.raises(ValueError):p.solve(x)
    finally:torch.backends.mha.set_fastpath_enabled(old)
    with torch.autocast('cpu',dtype=torch.bfloat16),pytest.raises(ValueError):p.solve(x)


def native_parts():
    m=model();d=m.dim;b=m.blocks[0]
    common=[m.token_embed.weight,torch.zeros(1,16,d),m.norm_y.weight,m.norm_z.weight,
            m.alpha_y,m.alpha_z,m.out_head.weight,m.out_head.bias]
    block=[b.norm1.weight,b.attn.in_proj_weight,b.attn.in_proj_bias,b.attn.out_proj.weight,
           b.attn.out_proj.bias,b.norm2.weight,b.ff[0].weight,b.ff[0].bias,b.ff[2].weight,b.ff[2].bias]
    return common,[block],[2]


@pytest.mark.parametrize('case',['common_count','blocks_empty','block_count','heads','token_rank','token_vocab',
                                  'norm_shape','alpha_shape','qkv_shape','ff_shape','nan','dtype','dim_overflow'])
def test_native_constructor_fails_closed(case):
    common,blocks,heads=native_parts()
    if case=='common_count':common.pop()
    elif case=='blocks_empty':blocks=[]
    elif case=='block_count':blocks[0].pop()
    elif case=='heads':heads=[3]
    elif case=='token_rank':common[0]=common[0].flatten()
    elif case=='token_vocab':common[0]=common[0][:4]
    elif case=='norm_shape':common[2]=torch.ones(15)
    elif case=='alpha_shape':common[4]=torch.ones(1)
    elif case=='qkv_shape':blocks[0][1]=blocks[0][1][:47]
    elif case=='ff_shape':blocks[0][6]=blocks[0][6][:,:15]
    elif case=='nan':common[0]=torch.full_like(common[0],float('nan'))
    elif case=='dtype':common[0]=common[0].double()
    elif case=='dim_overflow':common[0]=torch.ones(5,4097)
    with pytest.raises(RuntimeError):load_extension().PreparedFPStep(common,blocks,heads)


def test_native_state_shapes_and_input_range():
    h=load_extension().PreparedFPStep(*native_parts());x=torch.zeros(1,16,dtype=torch.int64)
    with pytest.raises(RuntimeError):h.encode(x.float())
    with pytest.raises(RuntimeError):h.encode(x+5)
    e=h.encode(x);z=torch.zeros_like(e)
    with pytest.raises(RuntimeError):h.step(e[:,:,:15],z,z)
    with pytest.raises(RuntimeError):h.step(e,z.double(),z)


def test_shared_snapshot_has_no_cross_request_state():
    p=PreparedFPSudoku(model());x=torch.zeros(1,16,dtype=torch.int64)
    expected=p.solve(x)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda _:p.solve(x),range(8)))
    for a,w in results:bits(a,expected[0]);assert w==expected[1]


@pytest.mark.parametrize('alpha',[0.0,-0.0,1e-40,-1.25])
@torch.inference_mode()
def test_signed_zero_subnormal_and_negative_update_scales(alpha):
    m=model();m.alpha_y.fill_(alpha);m.alpha_z.fill_(alpha);p=PreparedFPSudoku(m)
    x=torch.zeros(1,16,dtype=torch.int64);e,trace=p.trace(x);y,z=torch.zeros_like(e),torch.zeros_like(e)
    for Y,Z,L in trace:
        y,z=m.recursive_cycle(e,y,z);bits(Y,y);bits(Z,z);bits(L,m.out_head(y))


def test_unsupported_torch_version(monkeypatch):
    m=model();monkeypatch.setattr(torch,'__version__','2.11.0')
    with pytest.raises(RuntimeError,match='2.10'):PreparedFPSudoku(m)


@torch.inference_mode()
def test_matched_preparation_eager_control():
    from scripts.bench_fp_controls import PreparedEagerControl
    m=model();native=PreparedFPSudoku(m);eager=PreparedEagerControl(m)
    for vals in ([0]*16,[1,2,3,4,3,4,1,2,2,1,4,3,4,3,2,1]):
        x=torch.tensor([vals],dtype=torch.int64);a,w=native.solve(x);b,v=eager.solve(x)
        bits(a,b);assert w==v
    for param in m.parameters():param.fill_(float('nan'))
    a,w=native.solve(x);b,v=eager.solve(x);bits(a,b);assert w==v


def test_execution_identity_includes_nonparameter_attention_geometry():
    a=model();b=copy.deepcopy(a);b.blocks[0].attn.num_heads=4;b.blocks[0].attn.head_dim=4
    pa,pb=PreparedFPSudoku(a),PreparedFPSudoku(b)
    assert pa.source_state_sha256==pb.source_state_sha256
    assert pa.identity()['attention_heads']==[2] and pb.identity()['attention_heads']==[4]
    assert pa.identity()!=pb.identity()
