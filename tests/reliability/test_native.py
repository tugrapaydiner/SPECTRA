from concurrent.futures import ThreadPoolExecutor
import random
import numpy as np
import pytest
import torch
from spectra_reliability.native import NativeCPU, build, pack_ternary
from spectra_reliability.sudoku import all_four_by_four, valid, generate_unique
from spectra_reliability.runtime import ValidatedCPURecursiveRuntime, semantic_exit_native
from deploy.m10_artifact import export_cpu_artifact, load_cpu_artifact
from deploy.m10_runtime import CPURecursiveRuntime
from model.trm import TRM
from model.verifier import sudoku_correct
from scripts.m14_attempt5_dual_stream_semantic_exit import dual_stream_semantic_exit_solve


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    # Native compilation failure is not a skip or a false passing result.
    return NativeCPU(build(tmp_path_factory.mktemp("m16_native")))


@pytest.mark.parametrize("hidden,out,vectors", [(1,1,1),(3,7,3),(7,9,1),(16,16,16),
    (48,192,16),(63,65,3),(128,512,1),(257,15,2),(256,64,81)])
def test_cached_scalar_and_avx2_bits(native,hidden,out,vectors):
    rng=np.random.default_rng(hidden+out)
    weights=rng.integers(-1,2,(out,hidden),dtype=np.int8)
    w=pack_ternary(weights); s=rng.uniform(0,2,out).astype(np.float32)
    b=rng.normal(size=out).astype(np.float32); x=rng.normal(size=(vectors,hidden)).astype(np.float32)
    expected=native.checked_linear(x,w,s,b,hidden=hidden)
    with native.weight(w,s,b,hidden) as handle:
        actual=handle.linear(x)
        assert np.array_equal(actual.view(np.uint32),expected.view(np.uint32))
        if native.has_avx2:
            assert np.array_equal(handle.linear(x,avx2=True).view(np.uint32),expected.view(np.uint32))
        dense=x@(weights.astype(np.float32)*s[:,None]).T+b
        assert np.allclose(actual,dense,atol=2e-4,rtol=2e-4)
        assert handle.memory["unpacked_transposed_int8_bytes"]==weights.size


def test_source_mutation_lifetime_and_concurrent_calls(native):
    w=pack_ternary(np.ones((8,8),dtype=np.int8)); s=np.ones(8,dtype=np.float32)
    x=np.ones((1,8),dtype=np.float32); handle=native.weight(w,s,None,8)
    w[:]=255; s[:]=np.nan
    with ThreadPoolExecutor(max_workers=4) as pool:
        result=list(pool.map(lambda _: handle.linear(x),range(40)))
    assert all(np.all(y==8) for y in result)
    handle.close();handle.close()
    with pytest.raises(RuntimeError):handle.linear(x)


@pytest.mark.parametrize("kind",["reserved","padding","nan_scale","negative_scale","bad_bias"])
def test_invalid_payload_fails_at_construction(native,kind):
    w=pack_ternary(np.ones((8,7),dtype=np.int8)); s=np.ones(8,dtype=np.float32); b=np.zeros(8,dtype=np.float32)
    if kind=="reserved":w[0]|=3
    if kind=="padding":w[1]|=64
    if kind=="nan_scale":s[0]=np.nan
    if kind=="negative_scale":s[0]=-1
    if kind=="bad_bias":b[0]=np.inf
    with pytest.raises(ValueError):native.weight(w,s,b,7)


def test_native_checker_matches_independent_and_original(native):
    rng=random.Random(317); rows=[]
    for board in all_four_by_four():
        x=np.array([v if rng.random()>.5 else 0 for v in board],dtype=np.int64)
        y=np.array(board,dtype=np.int64); assert native.sudoku_valid(x,y,2)
        rows.append((x,y))
        for _ in range(16):
            bad=y.copy(); bad[rng.randrange(16)]=rng.randrange(-2,7)
            assert native.sudoku_valid(x,bad,2)==valid(tuple(x),tuple(bad),2)
            rows.append((x,bad))
        bad_x=x.copy();bad_x[0]=-1
        assert not native.sudoku_valid(bad_x,y,2)
    # Compare to the original tensor checker too, not only another new module.
    for start in range(0,len(rows),128):
        rr=rows[start:start+128];xx=torch.from_numpy(np.stack([x for x,y in rr]));yy=torch.from_numpy(np.stack([y for x,y in rr]))
        reference=sudoku_correct(xx,yy,2).bool().tolist()
        assert reference==[native.sudoku_valid(x,y,2) for x,y in rr]
    for _ in range(8):
        x,y=generate_unique(3,32,rng);xx=np.array(x,dtype=np.int64);yy=np.array(y,dtype=np.int64)
        assert native.sudoku_valid(xx,yy,3) and bool(sudoku_correct(torch.from_numpy(xx)[None],torch.from_numpy(yy)[None],3).item())
        yy[0]=yy[1]; assert native.sudoku_valid(xx,yy,3)==valid(xx,yy,3)


def test_no_implicit_dtype_or_stride_changes(native):
    with pytest.raises(ValueError):native.sudoku_valid(np.zeros(16),np.ones(16),2)
    with pytest.raises(ValueError):native.sudoku_valid(np.zeros(15,dtype=np.int64),np.ones(15,dtype=np.int64),2)
    w=pack_ternary(np.ones((8,8),dtype=np.int8));s=np.ones(8,dtype=np.float32)
    with native.weight(w,s,None,8) as h:
        with pytest.raises(ValueError):h.linear(np.zeros((2,8),dtype=np.float64))
        with pytest.raises(ValueError):h.linear(np.zeros((8,2),dtype=np.float32).T)
        with pytest.raises(ValueError):h.linear(np.zeros((1,9),dtype=np.float32))


def test_signed_zero_small_and_large_values(native):
    w=pack_ternary(np.tile(np.array([1,-1,0,1,-1,0,1,1],dtype=np.int8),(16,1)))
    x=np.array([[0.,-0.,1e-38,-1e-38,1e20,-1e20,1.,-1.]],dtype=np.float32);s=np.linspace(0,1,16,dtype=np.float32)
    with native.weight(w,s,None,8) as h:
        base=h.linear(x)
        if native.has_avx2:assert np.array_equal(base.view(np.uint32),h.linear(x,avx2=True).view(np.uint32))


@pytest.mark.parametrize("batch",[1,3])
@torch.inference_mode()
def test_full_runtime_keeps_original_graph_outputs(native,tmp_path,batch):
    torch.manual_seed(103)
    model=TRM(dim=16,num_tokens=5,seq_len=16,n_layers=1,n=1,T=1,N_sup=2,heads=4,max_grid_size=8,ternary=True,act8=True).eval()
    path=tmp_path/'artifact.pt'
    export_cpu_artifact(model,path,height=4,width=4,box=2,source_checkpoint_sha256='test',source_checkpoint_tensor_sha256='test',
        training_seed=103,training_step=0,data_provenance={'test':True},export_git_sha='test')
    loaded=load_cpu_artifact(path);original=CPURecursiveRuntime(loaded)
    x=torch.randint(0,5,(batch,16),dtype=torch.long)
    baseline=original.forward(x)
    for avx in [False,True] if native.has_avx2 else [False]:
        candidate=ValidatedCPURecursiveRuntime(loaded,native,avx2=avx);got=candidate.forward(x)
        assert torch.equal(got.logits,baseline.logits)
        assert torch.equal(got.answer,baseline.answer)
        assert got.work['native_linear_calls']==baseline.work['native_linear_calls']
        assert candidate.backend_report()['packed_only_execution'] is False
        assert candidate.backend_report()['additional_int8_layout_bytes']>0
        candidate.close()
        with pytest.raises(RuntimeError):candidate.forward(x)


@pytest.mark.parametrize("max_steps",[1,2,3,4])
@torch.inference_mode()
def test_semantic_exit_preserves_outputs_and_stopping(native,max_steps):
    torch.manual_seed(201)
    model=TRM(dim=64,num_tokens=5,seq_len=16,n_layers=1,n=1,T=1,N_sup=4,heads=4,max_grid_size=8).eval()
    rng=random.Random(33)
    for _ in range(8):
        puzzle,_=generate_unique(2,7,rng);x=torch.tensor([puzzle],dtype=torch.long)
        old,work=dual_stream_semantic_exit_solve(model,x,max_steps)
        new,nwork=semantic_exit_native(model,x,max_steps,native)
        assert torch.equal(old,new)
        assert work['executed_steps']==nwork['executed_steps']
        assert work['final_semantic']==nwork['final_semantic']
