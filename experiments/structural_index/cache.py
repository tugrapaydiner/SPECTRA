"""Export-local structural caches for pinned, quiescent m2cgen expression graphs.

Hash-consing is established prior art. The native candidate memoizes fingerprints
only as bucket selectors and verifies structural equality on collisions. A separate
Python canonical-ID implementation is retained as a simple control. Neither this
cache nor the original dict makes concurrent AST mutation safe.
"""
from __future__ import annotations
from collections.abc import MutableMapping
import importlib.util
import math
import os
from pathlib import Path

CACHE_MISS = object()


def schema():
    from m2cgen import ast
    import numpy as np
    entries = [('IdExpr',1,('expr',)),('FeatureRef',0,('index',)),('NumVal',0,('value',))]
    entries += [(n,1,('expr',)) for n in ('AbsExpr','AtanExpr','ExpExpr','LogExpr','Log1pExpr','SigmoidExpr','SqrtExpr','TanhExpr')]
    entries += [('PowExpr',3,('base_expr','exp_expr'))]
    entries += [(n,2,('left','right','op')) for n in ('BinNumExpr','BinVectorExpr','BinVectorNumExpr','CompExpr')]
    entries += [(n,5,('exprs','output_size')) for n in ('VectorVal','SoftmaxExpr')]
    entries += [('IfExpr',4,('test','body','orelse'))]
    return tuple((getattr(ast,n),k,a) for n,k,a in entries), (np.float64,np.float32,np.int64,np.int32,np.uint64,np.uint32), tuple(ast.BinNumOpType)+tuple(ast.CompOpType)


class Unsupported(ValueError):
    pass


class PythonIndex:
    """Straightforward bottom-up tuple interning, included as a strong simple control."""
    def __init__(self, memo=True):
        specs, numbers, ops = schema()
        self.specs={t:(i,k,a) for i,(t,k,a) in enumerate(specs,1)}
        self.numbers=numbers;self.ops=ops;self.memo={};self.nodes={};self.use_memo=memo
        self.visits=0;self.memo_hits=0

    def scalar(self,x):
        if type(x) not in (float,int,bool)+self.numbers: raise Unsupported('custom scalar')
        if hasattr(x,'__index__') and not isinstance(x,float) and abs(int(x))>2**53:raise Unsupported('wide integer')
        v=float(x)
        if math.isnan(v):raise Unsupported('nonreflexive scalar')
        return v

    def describe(self,obj):
        spec=self.specs.get(type(obj))
        if spec is None:raise Unsupported('custom AST')
        tag,kind,attrs=spec
        if kind==0:return (tag,self.scalar(getattr(obj,attrs[0]))),()
        if kind==5:
            children=getattr(obj,attrs[0])
            if type(children) not in (tuple,list):raise Unsupported('custom sequence')
            return (tag,self.scalar(getattr(obj,attrs[1]))),children
        if kind==2:
            op=getattr(obj,attrs[2])
            if not any(op is o for o in self.ops):raise Unsupported('custom operation')
            return (tag,op), (getattr(obj,attrs[0]),getattr(obj,attrs[1]))
        return (tag,), tuple(getattr(obj,a) for a in attrs)

    def key(self,root):
        memo=self.memo; ident=id(root)
        if self.use_memo and ident in memo:self.memo_hits+=1;return memo[ident][1]
        active=set(); stack=[(root,False)]
        while stack:
            obj,done=stack.pop(); oid=id(obj)
            if done:
                payload,children=self.describe(obj)
                signature=(payload,tuple(memo[id(c)][1] for c in children))
                token=self.nodes.setdefault(signature,len(self.nodes)+1)
                memo[oid]=(obj,token);active.remove(oid)
            elif self.use_memo and oid in memo:self.memo_hits+=1
            else:
                if oid in active:raise ValueError('cycle in expression graph')
                active.add(oid);self.visits+=1
                _,children=self.describe(obj);stack.append((obj,True))
                stack.extend((c,False) for c in reversed(children))
        return memo[ident][1]

    def stats(self):
        return dict(visits=self.visits,memo_hits=self.memo_hits,identity_entries=len(self.memo),canonical_entries=len(self.nodes))


_MODULE=None

def native_module():
    global _MODULE
    if _MODULE is None:
        location=Path(os.environ['SPECTRA_STRUCTURAL_LIBRARY']).resolve(strict=True)
        spec=importlib.util.spec_from_file_location('_spectra_structural',location)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);_MODULE=module
    return _MODULE


class VerifiedFingerprint:
    __slots__=('obj','digest','index')
    def __init__(self,obj,digest,index):self.obj=obj;self.digest=digest;self.index=index
    def __hash__(self):return self.digest
    def __eq__(self,other):
        if type(other) is not VerifiedFingerprint or self.index is not other.index:return False
        if self.digest!=other.digest:return False
        return self.obj is other.obj or self.index.module.equal(self.index.handle,self.obj,other.obj)


class NativeIndex:
    def __init__(self,memo=True,selective=False):
        specs,numbers,ops=schema();self.module=native_module()
        self.handle=self.module.create(specs,numbers,ops,int(os.environ.get('SPECTRA_INDEX_BYTES',536870912)),
            int(os.environ.get('SPECTRA_INDEX_NODES',10000000)),memo,os.environ.get('SPECTRA_INDEX_COLLISIONS')=='1')
        if selective:self.module.select(self.handle)
    def key(self,obj):
        value=self.module.key(self.handle,obj)
        return VerifiedFingerprint(obj,value,self) if hasattr(self.module,'equal') else value
    def stats(self):return self.module.stats(self.handle)


class StructuralCache(MutableMapping):
    def __init__(self):
        self._strict={s[0] for s in schema()[0]}
        self._seen=set();self._index=None;self._values={};self._fallback=None
        self.fallback_reason=None;self._failed=False

    def _make_index(self):
        mode=os.environ.get('SPECTRA_INDEX_MODE','native')
        if mode=='python':return PythonIndex()
        if mode=='native':return NativeIndex()
        if mode=='demand':return NativeIndex(selective=True)
        if mode=='no_memo':return NativeIndex(memo=False)
        raise ValueError('unknown structural index mode')

    def _abandon_index(self,reason):
        self._fallback={key:value for key,value in self._values.values()}
        self._values.clear();self._index=None;self.fallback_reason=reason

    def _key(self,obj):
        if self._index is None:self._index=self._make_index()
        try:return self._index.key(obj)
        except Unsupported:
            self._abandon_index('unsupported');return None
        except Exception as error:
            if isinstance(self._index,NativeIndex) and isinstance(error,self._index.module.Unsupported):
                self._abandon_index('unsupported');return None
            # Native failure resets its index. Never reuse old cached integer IDs.
            self._failed=True
            raise

    def get(self,key,default=None):
        if self._failed:raise RuntimeError('failed index epoch; clear before reuse')
        if self._fallback is not None:return self._fallback.get(key,default)
        cls=type(key)
        if cls not in self._strict:
            self._abandon_index('custom key');return self._fallback.get(key,default)
        if cls not in self._seen:return default
        token=self._key(key)
        if self._fallback is not None:return self._fallback.get(key,default)
        item=self._values.get(token)
        return default if item is None else item[1]

    def __contains__(self,key):return self.get(key,CACHE_MISS) is not CACHE_MISS
    def __getitem__(self,key):
        value=self.get(key,CACHE_MISS)
        if value is CACHE_MISS:raise KeyError(key)
        return value
    def __setitem__(self,key,value):
        if self._failed:raise RuntimeError('failed index epoch; clear before reuse')
        if self._fallback is not None:self._fallback[key]=value;return
        if type(key) not in self._strict:
            self._abandon_index('custom stored key');self._fallback[key]=value;return
        token=self._key(key)
        if self._fallback is not None:self._fallback[key]=value;return
        previous=self._values.get(token)
        self._values[token]=(key if previous is None else previous[0],value)
        self._seen.add(type(key))
    def __delitem__(self,key):
        if self._failed:raise RuntimeError('failed index epoch; clear before reuse')
        if self._fallback is not None:del self._fallback[key];return
        token=self._key(key)
        if self._fallback is not None:del self._fallback[key];return
        del self._values[token]
    def __iter__(self):return iter(self._fallback) if self._fallback is not None else (p[0] for p in self._values.values())
    def __len__(self):return len(self._fallback) if self._fallback is not None else len(self._values)
    def values(self):return self._fallback.values() if self._fallback is not None else (p[1] for p in self._values.values())
    def clear(self):self.__init__()
    def stats(self):return {'fallback':self.fallback_reason,**(self._index.stats() if self._index is not None else {})}
