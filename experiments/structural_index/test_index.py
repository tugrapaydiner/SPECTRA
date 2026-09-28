"""Semantic and algorithmic contracts for the optional export-local index."""
import copy,gc,math,os,random,weakref
import pytest
from m2cgen import ast,interpreters
from m2cgen.interpreters.structural_cache import NativeIndex,PythonIndex,StructuralCache


def leaf(n=1):return ast.NumVal(n)
def chain(n,reuse=False):
    x=ast.FeatureRef(0)
    for i in range(n):x=ast.BinNumExpr(x,leaf(i+1),ast.BinNumOpType.ADD)
    x.to_reuse=reuse;return x


def roots():
    a,b=leaf(),ast.FeatureRef(1)
    out=[leaf(0.),leaf(-0.),leaf(1.),ast.NumVal(1,dtype=int),b,ast.FeatureRef(0),leaf(math.inf),leaf(-math.inf)]
    for name in ('IdExpr','AbsExpr','AtanExpr','ExpExpr','LogExpr','Log1pExpr','SigmoidExpr','SqrtExpr','TanhExpr'):
        out.extend([getattr(ast,name)(a),getattr(ast,name)(b)])
    for op in ast.BinNumOpType:
        out += [ast.BinNumExpr(a,b,op),ast.BinNumExpr(b,a,op)]
        x,y=ast.VectorVal([a,b]),ast.VectorVal([b,a])
        out += [ast.BinVectorExpr(x,y,op),ast.BinVectorNumExpr(x,a,op)]
    out += [ast.PowExpr(a,b),ast.PowExpr(b,a),ast.VectorVal([a,b]),ast.SoftmaxExpr([a,b]),ast.VectorVal([])]
    for op in ast.CompOpType:out.append(ast.CompExpr(a,b,op))
    out += [ast.IfExpr(ast.CompExpr(a,b,ast.CompOpType.LT),a,b)]
    return out


@pytest.mark.parametrize('backend',['native','python'])
def test_equivalence_matches_original_pairwise(backend):
    index=NativeIndex() if backend=='native' else PythonIndex()
    objects=roots();objects+=copy.deepcopy(objects)
    keys=[index.key(x) for x in objects]
    for i,x in enumerate(objects):
        for j,y in enumerate(objects):assert bool(x==y)==(keys[i]==keys[j]),(i,j,type(x),type(y))


@pytest.mark.parametrize('memo',[False,True])
def test_same_types_and_ordered_children(memo):
    index=NativeIndex(memo=memo);a=chain(50);b=chain(51)
    assert index.key(a)!=index.key(b)
    assert index.key(a)==index.key(copy.deepcopy(a))
    assert index.key(ast.VectorVal([a,b]))!=index.key(ast.VectorVal([b,a]))


def test_collision_never_becomes_equality(monkeypatch):
    monkeypatch.setenv('SPECTRA_INDEX_COLLISIONS','1')
    index=NativeIndex();objects=roots();keys=[index.key(x) for x in objects]
    for i,x in enumerate(objects):
        for j,y in enumerate(objects):assert (keys[i]==keys[j])==bool(x==y)


def test_no_ast_hash_or_eq_is_used(monkeypatch):
    objects=roots();index=NativeIndex()
    def forbidden(*args):raise AssertionError('recursive AST operation invoked')
    for cls in {type(x) for x in objects}:
        monkeypatch.setattr(cls,'__hash__',forbidden);monkeypatch.setattr(cls,'__eq__',forbidden)
    for obj in objects:assert index.key(obj) is not None


@pytest.mark.parametrize('n',[16,64,256,1024,10000])
def test_same_type_population_visits_each_identity_once(n):
    index=NativeIndex();index.key(chain(1));root=chain(n);index.key(root)
    before=index.stats();x=root
    for _ in range(n):index.key(x);x=x.left
    after=index.stats()
    assert before['visits']==after['visits']==2*n+4
    assert after['memo_hits']>=n


def test_node_budget_is_enforced_and_can_reuse_after_clear(monkeypatch):
    monkeypatch.setenv('SPECTRA_INDEX_NODES','5')
    index=NativeIndex()
    with pytest.raises(index.module.LimitError):index.key(chain(10))
    assert index.key(leaf()) is not None


def test_native_allocation_budget_tracks_peak(monkeypatch):
    monkeypatch.setenv('SPECTRA_INDEX_BYTES','8192');index=NativeIndex()
    with pytest.raises(index.module.LimitError):index.key(chain(1000))
    assert index.stats()['peak_native_bytes']<=8192


def test_cycle_rejected_and_cleaned():
    root=chain(2);root.left=root;index=NativeIndex()
    with pytest.raises(ValueError,match='cycle'):index.key(root)
    assert index.key(leaf(4)) is not None
    root.left=leaf(0)


def test_resources_hold_objects_until_index_release():
    x=chain(4);refs=[weakref.ref(x),weakref.ref(x.left)];index=NativeIndex();index.key(x)
    del x;gc.collect();assert all(r() is not None for r in refs)
    del index;gc.collect();assert all(r() is None for r in refs)


def test_none_and_mapping_semantics():
    a,b=chain(2),chain(2);cache=StructuralCache();cache[a]=None
    assert cache.get(b,'missing') is None and b in cache
    assert list(cache)==[a] and list(cache.values())==[None]
    cache[b]=3;assert len(cache)==1 and cache.pop(a)==3 and len(cache)==0
    with pytest.raises(KeyError):cache[b]


def test_to_reuse_not_in_equality():
    cache=StructuralCache();a,b=chain(2),chain(2,True);cache[a]='x';assert cache[b]=='x'


def test_custom_stored_cross_type_equality_falls_back():
    class Cross:
        def __hash__(self):return hash(1.)
        def __eq__(self,x):return type(x) is ast.NumVal and x.value==1.
    cache=StructuralCache();key=Cross();cache[key]='x'
    assert cache[leaf(1)]=='x' and cache.fallback_reason is not None


def test_custom_child_falls_back_without_losing_prior_entries():
    class Number(ast.NumVal):pass
    cache=StructuralCache();a=ast.ExpExpr(leaf(1));cache[a]='first'
    b=ast.ExpExpr(Number(2));cache[b]='second'
    assert cache[a]=='first' and cache[b]=='second'
    assert cache.fallback_reason=='unsupported'


@pytest.mark.parametrize('value',[float('nan'),2**62])
def test_nonreflexive_or_wide_scalar_falls_back(value):
    a=ast.NumVal(value,dtype=lambda x:x);cache=StructuralCache();cache[a]='x'
    assert cache[a]=='x' and cache.fallback_reason=='unsupported'


def test_custom_enum_falls_back():
    from enum import Enum
    class Other(Enum): ADD='+'
    a=ast.BinNumExpr(leaf(1),leaf(2),ast.BinNumOpType.ADD);a.op=Other.ADD
    index=NativeIndex()
    with pytest.raises(index.module.Unsupported):index.key(a)


def test_clear_rebuilds_after_mutation_between_epochs():
    a=chain(2);cache=StructuralCache();cache[a]='old';cache.clear();a.right.value=9.
    cache[a]='new';assert cache[copy.deepcopy(a)]=='new'
    assert cache.get(chain(2)) is None


def test_interpreter_reexport_uses_fresh_snapshot():
    interpreter=interpreters.CInterpreter();a=ast.ExpExpr(chain(2),to_reuse=True)
    first=interpreter.interpret(ast.VectorVal([a,a]));a.expr.right.value=99.
    second=interpreter.interpret(ast.VectorVal([a,a]))
    assert first!=second and second==interpreters.CInterpreter().interpret(ast.VectorVal([a,a]))


@pytest.mark.parametrize('seed',range(8))
def test_random_mapping_agrees(seed):
    rng=random.Random(seed);keys=[leaf(i) for i in range(6)]+[chain(i) for i in range(1,7)]
    reference={};cache=StructuralCache()
    for _ in range(200):
        key=copy.deepcopy(rng.choice(keys));op=rng.randrange(5)
        if op==0:reference[key]=rng.randrange(20);cache[key]=reference[key]
        elif op==1:assert reference.get(key)==cache.get(key)
        elif op==2:assert reference.pop(key,None)==cache.pop(key,None)
        elif op==3:assert reference.setdefault(key,3)==cache.setdefault(key,3)
        else:reference.clear();cache.clear()
        assert list(reference.items())==list(cache.items())


def test_failed_epoch_cannot_reuse_old_tokens(monkeypatch):
    monkeypatch.setenv('SPECTRA_INDEX_NODES','10');cache=StructuralCache();cache[leaf(1)]='old'
    with pytest.raises(MemoryError):cache[chain(20)]='bad'
    with pytest.raises(RuntimeError,match='failed index'):cache.get(leaf(2))
    cache.clear();cache[leaf(2)]='new';assert cache[leaf(2)]=='new'


@pytest.mark.parametrize('depth',[8,16,24])
def test_demand_mode_shared_diamond_is_not_expanded(depth):
    root=ast.FeatureRef(0)
    for _ in range(depth):root=ast.BinNumExpr(root,root,ast.BinNumOpType.ADD)
    root=ast.ExpExpr(root,to_reuse=True);index=NativeIndex(selective=True)
    token=index.key(root);stats=index.stats()
    assert stats['visits']==depth+2 and stats['identity_entries']<=depth+2
    assert index.key(root)==token


def test_demand_mode_keeps_only_needed_or_shared_nodes():
    root=ast.ExpExpr(chain(300),to_reuse=True);index=NativeIndex(selective=True)
    key=index.key(root)
    assert index.stats()['identity_entries']<10
    assert index.key(root)==key


def test_full_and_demand_agree_on_shared_and_distinct_graphs():
    full=NativeIndex();demand=NativeIndex(selective=True)
    a=chain(40);objects=[ast.ExpExpr(a),ast.ExpExpr(copy.deepcopy(a)),ast.VectorVal([a,a])]
    for x in objects:
        for y in objects:assert (full.key(x)==full.key(y))==(demand.key(x)==demand.key(y))==bool(x==y)
