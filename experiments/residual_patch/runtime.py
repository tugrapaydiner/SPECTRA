"""Checked owner for a standalone native SAT experiment, not a production solver.

The C ABI accepts trusted pointers only through this module. Construction owns a
copy of every input, validates dimensions, and compiles outside the source tree.
All active configurations have bounded move counts. Returned buffers are copies.
"""
from __future__ import annotations
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import numpy as np
from data.cnf import CNF

F, H, W, CAP = 16, 16, 289, 24
WORK = ('moves', 'flips', 'literal_updates', 'feature_literal_visits', 'proposals',
        'model_calls', 'model_macs', 'restarts', 'checker_literal_visits',
        'native_ns', 'unsatisfied', 'native_status')
MODES = {'probsat': 0, 'walksat': 1, 'random_patch': 2, 'greedy_patch': 3, 'learned_patch': 4}
_LIB = None
_BUILD = None
D = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
I = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
L = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
B = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')

def library():
    global _LIB, _BUILD
    if _LIB is not None:
        return _LIB
    source = Path(__file__).with_name('kernel.cpp')
    flags = ['-O3', '-std=c++17', '-shared', '-fPIC', '-ffp-contract=off', '-fno-fast-math']
    compiler = os.environ.get('CXX', 'g++')
    identity = subprocess.check_output([compiler, '--version'], text=True)
    digest = hashlib.sha256(source.read_bytes()+identity.encode()+repr(flags).encode()).hexdigest()
    build = Path(tempfile.gettempdir())/'spectra-residual-patch'/digest
    build.mkdir(parents=True, exist_ok=True)
    target = build/'kernel.so'
    start = time.perf_counter_ns()
    if not target.exists():
        temporary = build/f'{os.getpid()}.so'
        try:
            subprocess.run([compiler, *flags, str(source), '-o', str(temporary)], check=True,
                           capture_output=True, text=True, timeout=120)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    _BUILD = {'compiler': identity, 'flags': flags, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'binary_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
              'build_or_cache_ns': time.perf_counter_ns()-start}
    lib = C.CDLL(str(target))
    lib.rp_error.restype = C.c_char_p
    lib.rp_create.argtypes = [C.c_int,C.c_int,I,I,C.c_int,B];lib.rp_create.restype=C.c_void_p
    lib.rp_destroy.argtypes=[C.c_void_p];lib.rp_destroy.restype=None
    lib.rp_inspect.argtypes=[C.c_void_p,B,I,L]
    lib.rp_patch.argtypes=[C.c_void_p,C.c_int,C.c_int,I]
    lib.rp_features.argtypes=[C.c_void_p,C.c_int,C.c_int,D]
    lib.rp_pool.argtypes=[C.c_void_p,C.c_uint64,I,D]
    lib.rp_scores.argtypes=[D,C.c_int,D,D]
    lib.rp_run.argtypes=[C.c_void_p,C.c_uint64,C.c_int,C.c_int64,C.c_int64,C.c_int,C.c_int,D,C.c_double,L]
    _LIB=lib
    return lib

def build_info():
    library()
    return dict(_BUILD)

def uint_seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError('unsigned 64-bit seed required')
    return seed

def parameters(weights):
    a = np.asarray(weights, dtype=np.float64)
    if a.shape != (W,) or not np.isfinite(a).all():
        raise ValueError('exact finite 16x16x1 scalar network parameters required')
    return np.array(a, dtype=np.float64, order='C', copy=True)

class State:
    def __init__(self, formula: CNF, witness: tuple[bool, ...]):
        if not isinstance(formula, CNF):
            raise TypeError('strict immutable CNF required')
        formula.validate_witness(witness)
        if formula.nvars > 4096 or len(formula.clauses)>65536 or sum(map(len,formula.clauses))>1000000:
            raise ValueError('native development geometry exceeded')
        self.formula=formula
        self.lib=library()
        offsets=np.array([0, *np.cumsum([len(c) for c in formula.clauses])],dtype=np.int32)
        lits=np.array([x for c in formula.clauses for x in c],dtype=np.int32)
        assignment=np.array(witness,dtype=np.uint8)
        self._ptr=self.lib.rp_create(formula.nvars,len(formula.clauses),offsets,lits,len(lits),assignment)
        if not self._ptr:
            raise ValueError(self.lib.rp_error().decode())
    def _alive(self):
        if not self._ptr:
            raise RuntimeError('native state is closed')
    def _ok(self,status):
        if status < 0:
            raise ValueError(self.lib.rp_error().decode())
        return status
    def close(self):
        if getattr(self,'_ptr',None):
            self.lib.rp_destroy(self._ptr);self._ptr=None
    def __enter__(self):
        self._alive();return self
    def __exit__(self,*args):self.close()
    def __del__(self):self.close()
    def inspect(self):
        self._alive()
        a=np.empty(self.formula.nvars,dtype=np.uint8);counts=np.empty(len(self.formula.clauses),dtype=np.int32);work=np.zeros(9,dtype=np.int64)
        unsat=self._ok(self.lib.rp_inspect(self._ptr,a,counts,work))
        return {'witness':tuple(bool(x) for x in a),'counts':counts.tolist(), 'unsatisfied':unsat,
                'work':dict(zip(WORK[:9],map(int,work)))}
    def _patch(self,variables):
        if type(variables) is not tuple or not 1<=len(variables)<=2 or any(type(v) is not int or not 0<=v<self.formula.nvars for v in variables) or len(set(variables)) != len(variables):
            raise ValueError('one or two distinct zero-based integer variables required')
        self._alive();return variables[0],variables[1] if len(variables)==2 else -1
    def patch(self,variables):
        u,v=self._patch(variables);mb=np.zeros(2,dtype=np.int32)
        self._ok(self.lib.rp_patch(self._ptr,u,v,mb));return tuple(map(int,mb))
    def features(self,variables):
        u,v=self._patch(variables);f=np.empty(F,dtype=np.float64)
        self._ok(self.lib.rp_features(self._ptr,u,v,f));return f
    def pool(self,seed):
        self._alive();uint_seed(seed)
        p=np.empty((CAP,2),dtype=np.int32);f=np.empty((CAP,F),dtype=np.float64)
        n=self._ok(self.lib.rp_pool(self._ptr,seed,p,f))
        return p[:n].copy(),f[:n].copy()
    def run(self,seed,*,mode='probsat',moves=1000,time_ns=0,interval=32,restart=0,weights=None,cb=2.3):
        self._alive();uint_seed(seed)
        if mode not in MODES or type(moves) is not int or not 0<=moves<=10000000 or type(time_ns) is not int or not 0<=time_ns<2**63 or type(interval) is not int or not 1<=interval<2**31 or type(restart) is not int or not 0<=restart<2**31:
            raise ValueError('invalid bounded search configuration')
        if type(cb) not in (float,int) or not np.isfinite(cb) or not 0<cb<=20:
            raise ValueError('invalid break exponent')
        if mode=='learned_patch' and weights is None:raise ValueError('learned mode requires a model')
        w=parameters(weights) if weights is not None else np.zeros(W,dtype=np.float64)
        out=np.zeros(12,dtype=np.int64)
        self._ok(self.lib.rp_run(self._ptr,seed,MODES[mode],moves,time_ns,interval,restart,w,cb,out))
        result=dict(zip(WORK,map(int,out)))
        result['witness']=self.inspect()['witness']
        return result

def native_scores(features,weights):
    f=np.ascontiguousarray(features,dtype=np.float64)
    if f.ndim!=2 or f.shape[1]!=F or len(f)>100000 or not np.isfinite(f).all():
        raise ValueError('invalid feature matrix')
    w=parameters(weights);out=np.empty(len(f),dtype=np.float64);lib=library()
    if lib.rp_scores(f,len(f),w,out)<0:raise ValueError(lib.rp_error().decode())
    return out
