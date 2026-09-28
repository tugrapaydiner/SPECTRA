"""Adversarial key semantics and measured hashing work, not timing assertions."""
import copy
import random

import pytest
from m2cgen import ast, interpreters
from m2cgen.interpreters.type_guarded_cache import TypeGuardedCache, CACHE_MISS


def expr(n=1, reuse=True):
    return ast.ExpExpr(ast.BinNumExpr(ast.FeatureRef(0), ast.NumVal(n),
                                    ast.BinNumOpType.ADD), to_reuse=reuse)


def test_equal_distinct_reuse():
    a, b = expr(), expr()
    cache = TypeGuardedCache()
    assert a is not b and a == b
    cache[a] = 'saved'
    assert cache.get(b, CACHE_MISS) == 'saved'
    cache[b] = None
    assert cache.get(a, CACHE_MISS) is None and len(cache) == 1


def test_collision_and_miss_sentinel():
    cache = TypeGuardedCache()
    a, b = ast.NumVal(1), ast.FeatureRef(1)
    assert hash(a) == hash(b) and a != b
    cache[a] = None
    assert cache.get(b, CACHE_MISS) is CACHE_MISS
    cache[b] = 0
    assert cache.get(a, CACHE_MISS) is None and cache.get(b) == 0


@pytest.mark.parametrize('operation', ['assignment', 'update', 'setdefault', 'ior_values'])
def test_custom_cross_type_key_disables_negative_filter(operation):
    class Cross:
        def __hash__(self): return hash(1.)
        def __eq__(self, other): return type(other) is ast.NumVal and other.value == 1.
    key, cache = Cross(), TypeGuardedCache()
    if operation == 'assignment': cache[key] = 'hit'
    elif operation == 'update': cache.update([(key, 'hit')])
    elif operation == 'setdefault': cache.setdefault(key, 'hit')
    else:
        # MutableMapping intentionally has no dict.__ior__ bypass: use public update.
        cache.update({key: 'hit'})
    assert cache.get(ast.NumVal(1), CACHE_MISS) == {key: 'hit'}.get(ast.NumVal(1))


def test_unknown_unhashable_lookup_keeps_error():
    class Unhashable: __hash__ = None
    cache = TypeGuardedCache()
    with pytest.raises(TypeError): cache.get(Unhashable(), CACHE_MISS)
    with pytest.raises(TypeError): Unhashable() in cache


def test_get_hashes_positive_query_once():
    class Count:
        calls = 0
        def __hash__(self): Count.calls += 1; return 7
        def __eq__(self, other): return type(other) is Count
    cache = TypeGuardedCache(); cache[Count()] = 'x'
    Count.calls = 0
    assert cache.get(Count(), CACHE_MISS) == 'x'
    assert Count.calls == 1


def test_failed_insert_does_not_poison_cache():
    cache = TypeGuardedCache()
    class Bad:
        def __hash__(self): raise RuntimeError('test')
    with pytest.raises(RuntimeError): cache[Bad()] = 0
    assert len(cache) == 0 and not cache._custom


def test_mutation_between_exports_and_cache_reset():
    interpreter = interpreters.CInterpreter()
    x = expr(1)
    source1 = interpreter.interpret(ast.VectorVal([x, x]))
    x.expr.right.value = 2.
    source2 = interpreter.interpret(ast.VectorVal([x, x]))
    assert source1 != source2
    assert source2 == interpreters.CInterpreter().interpret(ast.VectorVal([x, x]))


@pytest.mark.parametrize('seed', range(10))
def test_random_mapping_operations_match_dictionary(seed):
    rng = random.Random(seed)
    keys = [ast.NumVal(n) for n in range(8)] + [ast.FeatureRef(n) for n in range(8)] + [expr(n) for n in range(8)]
    control, cache = {}, TypeGuardedCache()
    for _ in range(500):
        key = copy.deepcopy(rng.choice(keys)); action = rng.randrange(6)
        if action == 0:
            value = rng.choice([None, 0, 17, 'v']); control[key] = value; cache[key] = value
        elif action == 1:
            assert cache.get(key, CACHE_MISS) == control.get(key, CACHE_MISS)
        elif action == 2:
            assert (key in cache) == (key in control)
        elif action == 3:
            assert cache.pop(key, CACHE_MISS) == control.pop(key, CACHE_MISS)
        elif action == 4:
            assert cache.setdefault(key, 3) == control.setdefault(key, 3)
        else:
            cache.clear(); control.clear()
        assert list(cache.items()) == list(control.items())


@pytest.mark.parametrize('kind', ['ExpExpr','BinNumExpr','VectorVal'])
def test_known_type_miss_does_not_hash_children(monkeypatch, kind):
    cls = getattr(ast, kind)
    cache = TypeGuardedCache();cache[ast.NumVal(19)] = 'leaf'
    x = expr() if kind=='ExpExpr' else expr().expr if kind=='BinNumExpr' else ast.VectorVal([expr(),expr(2)])
    def forbidden(self): raise AssertionError('impossible lookup hashed')
    monkeypatch.setattr(cls, '__hash__', forbidden)
    assert cache.get(x, CACHE_MISS) is CACHE_MISS


def test_same_type_miss_still_uses_structural_lookup(monkeypatch):
    cache = TypeGuardedCache();cache[expr(1)] = 'x'
    original = ast.ExpExpr.__hash__; counts = []
    def observed(self): counts.append(1); return original(self)
    monkeypatch.setattr(ast.ExpExpr, '__hash__', observed)
    assert cache.get(expr(2), CACHE_MISS) is CACHE_MISS and counts == [1]
