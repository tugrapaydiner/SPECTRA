"""Negative type filter plus single structural lookup for pinned m2cgen ASTs.

This is an export-local experiment, not a general-purpose dictionary replacement.
Well-formed stock AST equality rejects different exact types. Custom stored types
turn off the negative filter; potentially matching keys still use normal hashing
and equality. Mutable hashes, malformed nodes and arbitrary side effects are not
made safe. No permanent hash memoization or identity-only substitution is used.
"""
from collections.abc import MutableMapping

CACHE_MISS = object()


def strict_ast_types():
    from m2cgen import ast
    names = (
        'IdExpr', 'FeatureRef', 'NumVal', 'AbsExpr', 'AtanExpr', 'ExpExpr',
        'LogExpr', 'Log1pExpr', 'SigmoidExpr', 'SqrtExpr', 'TanhExpr', 'PowExpr',
        'BinNumExpr', 'VectorVal', 'SoftmaxExpr', 'BinVectorExpr',
        'BinVectorNumExpr', 'CompExpr', 'IfExpr',
    )
    return frozenset(getattr(ast, name) for name in names)


class TypeGuardedCache(MutableMapping):
    """Retain ordinary mapping semantics for valid expression keys.

    Historical types are conservative after deletion. Reset is explicit. A get
    performs at most one underlying structural dictionary lookup, unlike an
    `if key in cache: cache[key]` hit. None is a valid stored value, not a sentinel.
    """
    def __init__(self):
        self._values = {}
        self._seen = set()
        self._custom = False
        self._strict = strict_ast_types()

    def __contains__(self, key):
        cls = type(key)
        if cls in self._strict and not self._custom and cls not in self._seen:
            return False
        return key in self._values

    def get(self, key, default=None):
        cls = type(key)
        if cls in self._strict and not self._custom and cls not in self._seen:
            return default
        return self._values.get(key, default)

    def __getitem__(self, key):
        return self._values[key]

    def __setitem__(self, key, value):
        self._values[key] = value
        self._seen.add(type(key))
        self._custom = self._custom or type(key) not in self._strict

    def __delitem__(self, key):
        del self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def clear(self):
        self._values.clear()
        self._seen.clear()
        self._custom = False
