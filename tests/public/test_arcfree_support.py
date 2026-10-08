"""Whole-relation contracts for the exact arc-free support table."""
from itertools import combinations, product
import os, random
from pathlib import Path
import pytest
from spectra.cnf.quotient_query import QuotientRuntime, build_quotient_runtime
from spectra.cnf.arcfree_support import ArcFreeSupportTable, ResidualImplications

@pytest.fixture(scope='session')
def runtime(tmp_path_factory):
    path=os.environ.get('SPECTRA_QUOTIENT_LIBRARY')
    if path is None:path=build_quotient_runtime(tmp_path_factory.mktemp('arc-free-native'))
    return QuotientRuntime(path)

def truth(n,k,edges,masks,query):
    domains=[[c for c in range(k) if (masks[v]>>c)&1 and all(v!=qv or (qd>>c)&1 for qv,qd in query)] for v in range(n)]
    return [bytes(x) for x in product(*domains) if all(x[a]!=x[b] for a,b in edges)]

def test_all_small_arc_free_relations(runtime):
    # Disjoint edges/equivalence gadgets have no residual SCC-DAG arcs.
    rng=random.Random(19037)
    for n in range(1,7):
        bank=tuple(combinations(range(n),2))
        for _ in range(80):
            edges=tuple(e for e in bank if rng.randrange(5)==0)
            # A common two-colour palette makes each connected bipartite component
            # one exact binary choice. Non-bipartite cases are rejected at setup.
            masks=tuple(3 for _ in range(n))
            possible=truth(n,2,edges,masks,())
            if not possible:
                with pytest.raises(ValueError):ArcFreeSupportTable(runtime,n,2,edges,masks=masks)
                continue
            try:table=ArcFreeSupportTable(runtime,n,2,edges,masks=masks)
            except ResidualImplications:
                # Some directed implication structures are satisfiable but not
                # independent components; explicit refusal is the contract.
                continue
            for width in range(min(3,n)+1):
                for vertices in combinations(range(n),width):
                    for choices in product((1,2),repeat=width):
                        query=tuple(zip(vertices,choices));models=truth(n,2,edges,masks,query)
                        result=table.solve(query)
                        assert (result.status=='SAT_VERIFIED')==bool(models)
                        if models:assert result.labels in models
                        else:assert result.reason=='restriction_conflict' and result.labels==b''
            table.close()

def test_random_compiler_agreement(runtime):
    rng=random.Random(77121)
    accepted=0
    for _ in range(500):
        n=rng.randrange(1,12);edges=tuple(e for e in combinations(range(n),2) if rng.randrange(7)==0)
        masks=tuple(rng.choice((1,2,3)) for _ in range(n))
        if not truth(n,2,edges,masks,()):continue
        try:
            table=ArcFreeSupportTable(runtime,n,2,edges,masks=masks)
            ordinary=runtime.prepare(n,2,edges,masks=masks,mode='scc')
        except (ResidualImplications,ValueError):continue
        accepted+=1
        for __ in range(20):
            query=tuple((v,rng.choice((1,2,3))) for v in range(n) if rng.randrange(3)==0)
            a=table.solve(query);b=ordinary.solve(query)
            assert a.status==b.status
            if a.status=='SAT_VERIFIED':
                assert ordinary.check(a.labels,query)
        ordinary.close();table.close()
    assert accepted>=40

def test_arcs_are_refused(runtime):
    # Palettes {0,1}, {1,2} on an edge produce a one-way residual implication.
    with pytest.raises(ResidualImplications):ArcFreeSupportTable(runtime,2,3,((0,1),),masks=(3,6))

def test_duplicate_restrictions_intersect(runtime):
    with ArcFreeSupportTable(runtime,2,2,((0,1),),masks=(3,3)) as table:
        assert table.solve(((0,1),(0,1))).status=='SAT_VERIFIED'
        r=table.solve(((0,1),(0,2)))
        assert r.status=='UNKNOWN' and r.reason=='restriction_conflict'

def test_strict_inputs_close_and_record(runtime):
    table=ArcFreeSupportTable(runtime,1,2,(),masks=(3,))
    for bad in ([ (0,1) ],((True,1),),((0,True),),((1,1),),((0,4),),(0,1)):
        with pytest.raises(ValueError):table.solve(bad)
    record=table.solve(((0,1),)).record();record['labels'][0]=9
    assert table.solve(((0,1),)).labels==b'\x00'
    assert table.info['arc_free'] is True
    table.close();table.close()
    with pytest.raises(RuntimeError):table.solve()
    with pytest.raises(RuntimeError):table.__enter__()

def test_checker_rejection_is_fatal(runtime,monkeypatch):
    table=ArcFreeSupportTable(runtime,1,1,(),masks=(1,))
    monkeypatch.setattr(table._prepared,'check',lambda *_:False)
    with pytest.raises(AssertionError):table.solve()
    table.close()
