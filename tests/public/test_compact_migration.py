"""Exhaustive, differential, lifecycle and memory contracts for the sparse core."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from itertools import combinations, product
import json
import os
import random
import struct
import pytest
from experiments.real_traffic.compact_migration import CompactMigrationRuntime, build_compact_migration
from experiments.real_traffic.migration_session import MigrationRuntime, build_migration_session, MigrationVerifier, InvalidMigrationProof

@pytest.fixture(scope='session')
def compact(tmp_path_factory):
    lib=os.environ.get('SPECTRA_COMPACT_LIBRARY') or build_compact_migration(tmp_path_factory.mktemp('compact'))
    return CompactMigrationRuntime(lib)

@pytest.fixture(scope='session')
def legacy(tmp_path_factory):
    lib=os.environ.get('SPECTRA_MIGRATION_LIBRARY') or build_migration_session(tmp_path_factory.mktemp('legacy'))
    return MigrationRuntime(lib)

def feasible(edges,a,b,q):
    return [z for z in product(*[sorted({x,y}) for x,y in zip(a,b)])
            if all(z[u]!=z[v] for u,v in edges) and all(m>>z[v]&1 for v,m in q)]

def compare(compact,edges,a,b,qs,legacy=None,brute=False):
    verifier=MigrationVerifier(edges,a,b)
    with compact.prepare(edges,a,b) as c:
        bfs=c.solve_raw(qs,mode='bfs')
        tiled=c.solve_raw(qs,mode='tiled')
        auto=c.solve_raw(qs)
        for result in (bfs,tiled,auto): verifier.verify(qs,result,minimal=True)
        for result in (tiled,auto):
            assert result.statuses==bfs.statuses and result.labels==bfs.labels
            assert result.proof_vertices_le==bfs.proof_vertices_le
            assert result.proof_offsets==bfs.proof_offsets
        assert len(tiled.engine_ns)==0 # No manufactured per-query latency.
    if legacy:
        for dense in (False,True):
            with legacy.prepare(edges,a,b,compiled=dense) as p: old=p.solve_raw(qs)
            verifier.verify(qs,old,minimal=True)
            assert old.statuses==bfs.statuses and old.labels==bfs.labels
    if brute:
        for i,q in enumerate(qs):
            solutions=feasible(edges,a,b,q)
            assert bool(solutions)==(bfs.statuses[i]==0)
            if solutions:
                z=tuple(bfs.labels[i*len(a):(i+1)*len(a)])
                assert z in solutions
                moved={v for v in range(len(a)) if z[v]!=a[v]}
                assert all(moved <= {v for v in range(len(a)) if s[v]!=a[v]} for s in solutions)
    return bfs,tiled

def test_all_three_vertex_plan_pairs_and_compatible_graphs(compact):
    cases=0
    for a in product(range(3),repeat=3):
        for b in product(range(3),repeat=3):
            possible=tuple((u,v) for u,v in combinations(range(3),2) if a[u]!=a[v] and b[u]!=b[v])
            qs=tuple(tuple((v,m) for v,m in enumerate(pins) if m is not None)
                     for pins in product(*[(None,1<<x,1<<y) for x,y in zip(a,b)]))
            for mask in range(1<<len(possible)):
                edges=tuple(e for i,e in enumerate(possible) if mask>>i&1)
                compare(compact,edges,a,b,qs,brute=True); cases+=1
    assert cases==2169

def test_seeded_small_exhaustive_differential(compact,legacy):
    rng=random.Random(990017)
    for trial in range(150):
        n=rng.randrange(1,8); a=tuple(rng.randrange(5) for _ in range(n)); b=tuple(rng.randrange(5) for _ in range(n))
        edges=tuple((u,v) for u,v in combinations(range(n),2) if a[u]!=a[v] and b[u]!=b[v] and rng.random()<.5)
        qs=tuple(tuple((v,rng.choice((0,1<<a[v],1<<b[v],(1<<a[v])|(1<<b[v]),2**64-1)))
                       for v in range(n) if rng.random()<.5) for _ in range(19))
        compare(compact,edges,a,b,qs,legacy,brute=True)

@pytest.mark.parametrize('count',[0,1,2,31,32,63,64,65,127,128,129,193,1025])
def test_word_boundaries_cycles_sparse_requests(compact,legacy,count):
    n=69;a=tuple(i%3 for i in range(n));b=tuple((i+1)%3 for i in range(n))
    edges=tuple((i,(i+1)%n) for i in range(n))
    bank=((),((0,2),),((0,2),(68,4)),((0,1),(0,2)),((0,0),),((68,1),),((0,7),))
    qs=tuple(bank[i%len(bank)] for i in range(count))
    compare(compact,edges,a,b,qs,legacy)

def test_all_unchanged_and_isolated_changed_vertices(compact,legacy):
    compare(compact,((0,1),(1,2)),(0,1,2),(0,1,2),((),((0,0),),((1,2),)),legacy)
    compare(compact,(),(0,1,63),(1,2,62),((),((0,2),),((2,1<<62),),((2,1<<63),(2,1<<62))),legacy)

def test_redundant_duplicate_edges_do_not_change_relation(compact):
    a=(0,1,2);b=(1,2,0);qs=(((0,2),),((0,2),(2,4)))
    r,_=compare(compact,((0,1),(1,2)),a,b,qs)
    s,_=compare(compact,((0,1),(1,0),(0,1),(1,2),(2,1)),a,b,qs)
    assert r.statuses==s.statuses and r.labels==s.labels

@pytest.mark.parametrize('bad',[(),(True,),(-1,),(64,),('0',)])
def test_invalid_colour_plan(compact,bad):
    with pytest.raises(ValueError): compact.prepare((),bad,bad)

@pytest.mark.parametrize('args',[((),(0,),(0,1)),(((0,0),),(0,),(1,)),(((0,2),),(0,1),(1,0)),(((0,1),),(0,0),(1,0))])
def test_invalid_input_graph(compact,args):
    with pytest.raises(ValueError):compact.prepare(*args)

@pytest.mark.parametrize('pin',[(True,1),(-1,1),(1,1),(0,True),(0,-1),(0,1<<64),(0,1,2)])
def test_invalid_restriction(compact,pin):
    with compact.prepare((),(0,),(1,)) as p:
        with pytest.raises(ValueError):p.solve_raw(((pin,),))

def test_caps_lifecycle_and_owned_byte_outputs(compact):
    with pytest.raises(MemoryError):compact.prepare((),(0,),(1,),max_bytes=1)
    p=compact.prepare((),(0,),(1,))
    r=p.solve_raw((((0,1),),));s=p.solve_raw((((0,2),),))
    assert r.labels==b'\0' and s.labels==b'\1'
    with pytest.raises(MemoryError):p.solve_raw(((),),max_bytes=1)
    for bad in (False,1,'wrong',[],None):
        with pytest.raises(ValueError):p.solve_raw(((),),mode=bad)
    p.close();p.close()
    with pytest.raises(RuntimeError):p.solve_raw(())
    with pytest.raises(RuntimeError):_ = p.info
    with pytest.raises(RuntimeError):p.__enter__()

def test_sparse_index_has_no_quadratic_closure(compact,legacy):
    n=1024; a=tuple(i%3 for i in range(n));b=tuple((i+1)%3 for i in range(n));edges=tuple((i,i+1) for i in range(n-1))
    with compact.prepare(edges,a,b) as p:
        info=p.info; result=p.solve_raw((((0,2),),),mode='tiled')
        assert info['components']==n
        assert info['payload_bytes']<40*n
        assert result.scratch_payload_bytes<64*n
    with legacy.prepare(edges,a,b,compiled=True) as p:
        assert info['payload_bytes']<p.info['payload_bytes']/4

def test_forged_answers_paths_and_nonminimality_are_rejected(compact):
    a=(0,1,0);b=(1,0,2);edges=((0,1),);qs=(((0,2),(1,2)),(),((0,0),))
    with compact.prepare(edges,a,b) as p:r=p.solve_raw(qs)
    verifier=MigrationVerifier(edges,a,b)
    for bad in (replace(r,statuses=b'\0\0\1'),replace(r,proof_kinds=b'\0\0\2'),
                replace(r,proof_vertices_le=struct.pack('<III',0,2,0))):
        with pytest.raises(InvalidMigrationProof):verifier.verify(qs,bad,minimal=True)
    labels=bytearray(r.labels);labels[5]=2
    with pytest.raises(InvalidMigrationProof,match='minimal'):verifier.verify(qs,replace(r,labels=bytes(labels)),minimal=True)

def test_parallel_calls_do_not_share_mutable_worker_state(compact):
    with compact.prepare(((0,1),),(0,1),(1,0)) as p:
        qs=(((0,2),),((0,1),(1,1)))*65
        with ThreadPoolExecutor(max_workers=6) as pool: results=list(pool.map(lambda _:p.solve_raw(qs),range(30)))
    assert len({(r.statuses,r.labels,r.proof_vertices_le) for r in results})==1

def test_proof_graph_sparsification_preserves_dense_cycle_relation(compact,legacy):
    n=128;a=tuple(i%2 for i in range(n));b=tuple(1-x for x in a)
    edges=tuple((u,v) for u in range(n) for v in range(u+1,n) if a[u]!=a[v])
    qs=((),((0,2),),((0,2),(1,2)),((127,1),))*33
    compare(compact,edges,a,b,qs,legacy)
    with compact.prepare(edges,a,b,sparsify=True) as sparse, compact.prepare(edges,a,b,sparsify=False) as full:
        assert sparse.info['source_dependency_arcs']==8192
        assert sparse.info['proof_dependency_arcs']<=2*(n-1)
        assert sparse.info['payload_bytes']<full.info['payload_bytes']/10
        assert sparse.solve_raw(qs).labels==full.solve_raw(qs).labels
    with pytest.raises(ValueError):compact.prepare(edges,a,b,sparsify=1)
