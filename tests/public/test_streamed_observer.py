"""Independent original-input observer, including altered proofs and minima."""
from dataclasses import replace
from itertools import combinations,product
import os,random,struct
from concurrent.futures import ThreadPoolExecutor
import pytest
from experiments.real_traffic.streamed_observer import StreamedObserverRuntime,build_streamed_observer
from experiments.real_traffic.compact_migration import CompactMigrationRuntime,build_compact_migration
from experiments.real_traffic.migration_session import MigrationVerifier,MigrationBatch

@pytest.fixture(scope='session', params=['bfs','tiled'])
def observer(tmp_path_factory, request):
    p=os.environ.get('SPECTRA_STREAMED_OBSERVER_LIBRARY') or build_streamed_observer(tmp_path_factory.mktemp('original-observer'))
    runtime=StreamedObserverRuntime(p)
    original=runtime.prepare
    def prepare(*args,**kwargs):
        from functools import partial
        obj=original(*args,**kwargs)
        obj.verify=partial(obj.verify, mode=request.param)
        return obj
    runtime.prepare=prepare
    return runtime

@pytest.fixture(scope='session')
def runtime(tmp_path_factory):
    p=os.environ.get('SPECTRA_COMPACT_LIBRARY') or build_compact_migration(tmp_path_factory.mktemp('observed-runtime'))
    return CompactMigrationRuntime(p)

def test_exhaustive_three_vertex_models_and_queries(observer,runtime):
    count=0
    for a in product(range(3),repeat=3):
        for b in product(range(3),repeat=3):
            possible=tuple((u,v) for u,v in combinations(range(3),2) if a[u]!=a[v] and b[u]!=b[v])
            queries=tuple(tuple((v,pin) for v,pin in enumerate(pins) if pin is not None)
                          for pins in product(*[(None,1<<x,1<<y) for x,y in zip(a,b)]))
            for mask in range(1<<len(possible)):
                edges=tuple(e for i,e in enumerate(possible) if mask>>i&1)
                with runtime.prepare(edges,a,b) as p,observer.prepare(edges,a,b) as check:
                    result=p.solve_raw(queries,mode='tiled');receipt=check.verify(queries,result)
                    expected=MigrationVerifier(edges,a,b).verify(queries,result,minimal=True)
                    assert receipt.minimum_checked and receipt.sat==expected['sat'] and receipt.unsat==expected['unsat']
                    assert receipt.migrations==expected['migrations']
                count+=1
    assert count==2169

def fixture(runtime):
    a=(0,1,0);b=(1,0,2);e=((0,1),);q=(((0,2),(1,2)),(),((0,0),))
    with runtime.prepare(e,a,b) as p:result=p.solve_raw(q)
    return e,a,b,q,result

def test_tampering_is_rejected(observer,runtime):
    e,a,b,q,r=fixture(runtime)
    mutated=[replace(r,statuses=b'\0\0\1'),replace(r,statuses=b'\2\0\1'),
             replace(r,proof_kinds=b'\0\0\2'),replace(r,proof_offsets=(0,1,0,3)),
             replace(r,proof_vertices_le=struct.pack('<III',0,2,0)),replace(r,proof_vertices_le=b'bad'),
             replace(r,labels=b''),replace(r,proof_offsets=(0,2,2,4)),replace(r,proof_offsets=(False,2,2,3))]
    with observer.prepare(e,a,b) as check:
        assert check.verify(q,r).minimum_checked
        for z in mutated:
            with pytest.raises(ValueError):check.verify(q,z)
        # A valid but unnecessary migration is not a minimum-change answer.
        labels=bytearray(r.labels);labels[5]=2
        changed=replace(r,labels=bytes(labels))
        with pytest.raises(ValueError,match='minimum'):check.verify(q,changed)
        assert check.verify(q,changed,minimal=False).minimum_checked is False
        # Invalid colour must be rejected before a mask shift is attempted.
        labels=bytearray(r.labels);labels[4]=255
        with pytest.raises(ValueError):check.verify(q,replace(r,labels=bytes(labels)))
        labels=bytearray(r.labels);labels[0]=7
        with pytest.raises(ValueError):check.verify(q,replace(r,labels=bytes(labels)))
        with pytest.raises(MemoryError):check.verify(q,r,max_bytes=1)
        with pytest.raises(ValueError):check.verify(q,r,minimal=1)

@pytest.mark.parametrize('count',[0,1,63,64,65,129])
def test_duplicates_and_empty_masks(observer,runtime,count):
    a=(0,1,63);b=(1,0,62);e=((0,1),);bank=((),((0,2),(0,1)),((2,0),),((2,1<<62),(2,1<<62)),((0,3),))
    q=tuple(bank[i%len(bank)] for i in range(count))
    with runtime.prepare(e,a,b) as p,observer.prepare(e,a,b) as check:
        r=p.solve_raw(q);x=check.verify(q,r);y=MigrationVerifier(e,a,b).verify(q,r,minimal=True)
        assert x.sat==y['sat'] and x.unsat==y['unsat']

def test_checker_does_not_require_solver_or_quotient(observer):
    e=((0,1),);a=(0,1);b=(1,0)
    # Handmade SAT witness and a separate handmade original conflict path.
    r=MigrationBatch(b'\0\1',b'\1\0\0\0',b'\0\1',(0,0,2),struct.pack('<II',0,1),(),(),0)
    q=(((0,2),),((0,2),(1,2)))
    with observer.prepare(e,a,b) as check:
        assert check.verify(q,r).sat==1
        assert check.info['payload_bytes']<100

def test_original_edges_are_checked_without_candidate_preprocessing(observer):
    # Outside-minimum checking still directly rejects an original edge violation.
    e=((0,1),);a=(0,1);b=(1,0);q=((),)
    r=MigrationBatch(b'\0',b'\0\0',b'\0',(0,0),b'',(),(),0)
    with observer.prepare(e,a,b) as check:
        with pytest.raises(ValueError,match='original conflict'):check.verify(q,r,minimal=False)

def test_lifecycle_parallel_and_linear_storage(observer,runtime):
    e,a,b,q,r=fixture(runtime);p=observer.prepare(e,a,b)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda _:p.verify(q,r),range(30)))
    assert all(z.sat==1 and z.unsat==2 for z in results)
    p.close();p.close()
    with pytest.raises(RuntimeError):p.verify(q,r)
    with pytest.raises(RuntimeError):_=p.info
    with pytest.raises(RuntimeError):p.__enter__()
    n=10000;aa=tuple(i%3 for i in range(n));bb=tuple((i+1)%3 for i in range(n));ee=tuple((v,v+1) for v in range(n-1))
    with observer.prepare(ee,aa,bb,max_bytes=4*1024**2) as p:assert p.info['payload_bytes']<20*n

def test_checked_service_never_returns_an_unchecked_answer(observer,runtime):
    from experiments.real_traffic.compact_service import CompactService
    service=object.__new__(CompactService);service.solver=runtime;service.observer=observer
    e,a,b,q,_=fixture(runtime)
    with service.prepare(e,a,b) as p:
        result=p.solve(q)
        assert result.audit.minimum_checked and result.audit.sat==1 and result.audit.unsat==2
        assert p.info['combined_index_payload_bytes']>0
    with pytest.raises(RuntimeError):p.solve(q)
    with pytest.raises(RuntimeError):p.__enter__()
    with pytest.raises(MemoryError):service.prepare(e,a,b,max_bytes=1)

@pytest.mark.parametrize('changed_lane',[0,31,63,64,128])
def test_dense_bit_checker_rejects_each_lane_of_a_false_colouring(observer,changed_lane):
    n=16;count=129;a=tuple(i%2 for i in range(n));b=tuple(1-x for x in a)
    e=tuple((u,v) for u in range(n) for v in range(u+1,n) if a[u]!=a[v]);q=((),)*count
    labels=bytes(a)*count
    good=MigrationBatch(bytes(count),labels,bytes(count),(0,)*(count+1),b'',(),(),0)
    with observer.prepare(e,a,b) as check:
        assert check.verify(q,good).sat==count
        changed=bytearray(labels);changed[changed_lane*n]=b[0]
        with pytest.raises(ValueError,match='original conflict'):
            check.verify(q,replace(good,labels=bytes(changed)),minimal=False)


def test_canonical_bank_is_reused_but_mutable_rows_are_snapshotted(observer):
    q=(((0,1),(2,8)),(),((1,2),))
    assert observer.freeze_queries(q) is q
    mutable=[[[0,1],[2,8]],[],[[1,2]]]
    frozen=observer.freeze_queries(mutable)
    assert frozen==q
    mutable[0][0][1]=7
    assert frozen==q
    assert observer.freeze_queries((iter(row) for row in q))==q
    class TupleChild(tuple):pass
    frozen=observer.freeze_queries(TupleChild((TupleChild((TupleChild((0,1)),)),)))
    assert frozen==(((0,1),),)
    assert type(frozen) is type(frozen[0]) is type(frozen[0][0]) is tuple


def test_freezing_is_bounded_and_propagates_iterator_errors(observer):
    with pytest.raises(MemoryError):observer.freeze_queries((() for _ in range(5)),max_queries=4)
    with pytest.raises(MemoryError):observer.freeze_queries((((0,1),(0,1)),),max_pins=1)
    with pytest.raises(MemoryError):observer.freeze_queries([[[0,1,2]]])
    for q in (((),), [[[True,1]]], [[[0,False]]], [[[0,1.5]]]):
        if q==((),):
            assert observer.freeze_queries(q)==q
        else:
            with pytest.raises(ValueError):observer.freeze_queries(q)
    def broken():
        yield ()
        raise RuntimeError('input iterator failed')
    with pytest.raises(RuntimeError,match='input iterator failed'):observer.freeze_queries(broken())
    with pytest.raises(TypeError):observer.freeze_queries([None])
    assert observer.freeze_queries((),max_queries=0,max_pins=0)==()


@pytest.mark.parametrize('count',[1,63,64,65,128,129])
def test_unseeded_cycles_are_not_minimum_solutions(observer,count):
    a=(0,1);b=(1,0);e=((0,1),);q=((),)*count
    r=MigrationBatch(bytes(count),bytes(b)*count,bytes(count),(0,)*(count+1),b'',(),(),0)
    with observer.prepare(e,a,b) as check:
        assert check.verify(q,r,minimal=False).sat==count
        with pytest.raises(ValueError,match='minimum'):check.verify(q,r,minimal=True)


def test_all_original_edge_states_not_just_solver_answers(observer):
    import random
    rng=random.Random(72091)
    states=0
    for _ in range(100):
        n=rng.randrange(2,8)
        a=tuple(rng.randrange(5) for _ in range(n));b=tuple(rng.randrange(5) for _ in range(n))
        e=tuple((u,v) for u,v in combinations(range(n),2) if a[u]!=a[v] and b[u]!=b[v] and rng.randrange(2))
        # Enumerate proper and deliberately improper submitted assignments.
        with observer.prepare(e,a,b) as ck:
            for bits in product((0,1),repeat=n):
                labels=bytes(b[v] if bits[v] else a[v] for v in range(n))
                r=MigrationBatch(b'\0',labels,b'\0',(0,0),b'',(),(),0)
                valid=all(labels[u]!=labels[v] for u,v in e)
                if valid:assert ck.verify(((),),r,minimal=False).sat==1
                else:
                    with pytest.raises(ValueError,match='original conflict'):ck.verify(((),),r,minimal=False)
                states+=1
    assert states>3000


def test_vacuous_edges_need_no_retained_edge_bank(observer):
    # A and B use disjoint colour ranges; every A/B combination is valid.
    n=16;a=tuple(range(n));b=tuple(v+16 for v in range(n))
    e=tuple(combinations(range(n),2))
    with observer.prepare(e,a,b) as ck:
        assert ck.info['original_edges']==120
        assert ck.info['original_dependency_arcs']==0
        assert ck.info['payload_bytes']==6*n+4
        r=MigrationBatch(b'\0',bytes(b),b'\0',(0,0),b'',(),(),0)
        assert ck.verify(((),),r,minimal=False).sat==1
        with pytest.raises(ValueError,match='minimum'):ck.verify(((),),r)
    # Improper original plans are still rejected even if an edge would be pruned.
    with pytest.raises(ValueError):observer.prepare(((0,1),),(0,0),(1,2))


def test_new_checked_service_freezes_once_and_checks_fresh_outputs(observer,runtime):
    from experiments.real_traffic.streamed_service import StreamedService
    service=object.__new__(StreamedService);service.solver=runtime;service.observer=observer
    e,a,b,q,_=fixture(runtime)
    with service.prepare(e,a,b) as p:
        r=p.solve([list(map(list,row)) for row in q],mode='tiled')
        assert r.audit.sat==1 and r.audit.unsat==2 and r.audit.minimum_checked
        assert p.info['combined_index_payload_bytes']<1000
    with pytest.raises(RuntimeError):p.solve(q)


def test_frozen_entry_points_do_not_skip_native_validation(observer,runtime):
    with runtime.prepare(((0,1),),(0,1),(1,0)) as p,observer.prepare(((0,1),),(0,1),(1,0)) as ck:
        r=p.solve_frozen(((),))
        with pytest.raises(ValueError):p.solve_frozen([[]])
        with pytest.raises(ValueError):p.solve_frozen(([(0,1)],))
        with pytest.raises(ValueError):p.solve_frozen((((False,1),),))
        with pytest.raises(ValueError):p.solve_frozen((((0,1<<64),),))
        with pytest.raises(ValueError):ck.verify_frozen([[]],r)
        with pytest.raises(ValueError):ck.verify_frozen(([(0,1)],),r)
        with pytest.raises(ValueError):ck.verify_frozen((((0,True),),),r)
