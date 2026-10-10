"""Opt-in exact sparse covering/exclusion search; no external solver fallback.

Build explicitly on Linux. The original choice groups are checked after every SAT
return. Models, defaults and the old cover runtime are not replaced.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sysconfig
import tempfile
import threading
import time

SOURCE=Path(__file__).resolve().parents[1]/'_native'/'sparse_cover.cpp'


def build_sparse_runtime(directory: str|Path, *, compiler: str='g++', sanitize: bool=False) -> Path:
    if platform.system()!='Linux':raise NotImplementedError('Linux builder only; other platforms unvalidated')
    if type(sanitize) is not bool:raise TypeError('sanitize must be bool')
    cc=shutil.which(compiler)
    if not cc:raise FileNotFoundError(compiler)
    dest=Path(directory).resolve();dest.mkdir(parents=True,exist_ok=True)
    library=dest/('_spectra_sparse'+sysconfig.get_config_var('EXT_SUFFIX'))
    receipt=dest/'build.json'
    if library.exists() or receipt.exists():raise FileExistsError('build outputs already exist')
    flags=['-std=c++17','-shared','-fPIC','-Wall','-Wextra','-Werror','-fno-fast-math','-ffp-contract=off']
    flags+=['-O1','-g','-fsanitize=undefined','-fno-sanitize-recover=all'] if sanitize else ['-O3','-DNDEBUG']
    start=time.perf_counter_ns()
    with tempfile.TemporaryDirectory(dir=dest,prefix='.sparse-') as tmp:
        target=Path(tmp)/library.name
        include=sysconfig.get_path('include')
        command=[cc,*flags,'-I'+include,str(SOURCE),'-o',str(target)]
        result=subprocess.run(command,capture_output=True,text=True,timeout=120)
        if result.returncode:raise RuntimeError(result.stderr)
        record={'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                'library_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'command':command,
                'compiler':subprocess.run([cc,'--version'],capture_output=True,text=True,check=True).stdout,
                'elapsed_ns':time.perf_counter_ns()-start,'stdout':result.stdout,'stderr':result.stderr,
                'python_include':include,'sanitize':sanitize}
        os.link(target,library)
        with receipt.open('x') as f:json.dump(record,f,indent=2,sort_keys=True);f.write('\n')
    return library


def valid_witness(nvars: int,covers: tuple,exclusive: tuple,witness: tuple,assumptions: tuple=()) -> bool:
    """Original-group observer: independent of native incidence and search state."""
    if type(witness) is not tuple or len(witness)!=nvars or any(type(x) is not bool for x in witness):return False
    for group in covers:
        if not any(witness[v-1] for v in group):return False
    for group in exclusive:
        seen=False
        for v in group:
            if witness[v-1]:
                if seen:return False
                seen=True
    return all(witness[abs(a)-1]==(a>0) for a in assumptions)


@dataclass(frozen=True)
class SparseResult:
    status: str
    witness: tuple[bool,...]
    reason: str
    work: int
    nodes: int
    assignments: int
    backtracks: int
    index_bytes: int
    state_payload_bytes: int
    trace: str
    elapsed_ns: int
    binary_calls: int = 0
    binary_edges: int = 0
    binary_solved: int = 0
    jumps: int = 0
    skipped_levels: int = 0
    analysis_visits: int = 0
    conflict_payload_bytes: int = 0

    def record(self):
        return {**self.__dict__,'witness':list(self.witness),'schema':'spectra.sparse.v1','learned':False}


class SparseRuntime:
    """Explicit trusted native module load. No compiling or loading external solvers."""
    def __init__(self,library: str|Path):
        spec=importlib.util.spec_from_file_location('_spectra_sparse',Path(library).resolve())
        if spec is None or spec.loader is None:raise ValueError('invalid extension location')
        self._module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self._module)
        if not hasattr(self._module,"abi") or self._module.abi()!=9:raise ValueError("unsupported sparse native ABI; explicitly rebuild the runtime")

    def prepare(self,nvars: int,covers: tuple,exclusive: tuple, *,max_build_bytes: int=64*1024*1024):
        return PreparedSparse(self,nvars,covers,exclusive,max_build_bytes=max_build_bytes)

    def solve(self,nvars: int,covers: tuple,exclusive: tuple, *,max_work: int=1000000,
              max_build_bytes: int=64*1024*1024,max_state_bytes: int=64*1024*1024,
              heap: bool=False,active: bool=False,xor_units: bool=True,degree: bool=False,lcv: bool=False,binary: bool=False,compact_updates: bool=False,native_check: bool=False,wdeg: bool=False,backjump: bool=False,full_conflicts: bool=False,assumptions: tuple=()) -> SparseResult:
        start=time.perf_counter_ns()
        with self.prepare(nvars,covers,exclusive,max_build_bytes=max_build_bytes) as p:
            r=p.solve(max_work=max_work,max_state_bytes=max_state_bytes,heap=heap,active=active,xor_units=xor_units,degree=degree,lcv=lcv,binary=binary,compact_updates=compact_updates,native_check=native_check,wdeg=wdeg,backjump=backjump,full_conflicts=full_conflicts,assumptions=assumptions)
        return SparseResult(r.status,r.witness,r.reason,r.work,r.nodes,r.assignments,r.backtracks,
                            r.index_bytes,r.state_payload_bytes,r.trace,time.perf_counter_ns()-start,r.binary_calls,r.binary_edges,r.binary_solved,r.jumps,r.skipped_levels,r.analysis_visits,r.conflict_payload_bytes)


class PreparedSparse:
    def __init__(self,runtime: SparseRuntime,nvars: int,covers: tuple,exclusive: tuple, *,max_build_bytes: int):
        self._lock=threading.RLock();self._module=runtime._module
        # Native admission validates exact tuple/int types, distinctness and sizes,
        # and copies all data. Input callbacks and arbitrary iterators are unsupported.
        self._handle=self._module.create(nvars,covers,exclusive,max_build_bytes)
        self._original=(nvars,covers,exclusive)

    def solve(self, *,max_work: int=1000000,max_state_bytes: int=64*1024*1024,
              heap: bool=False,active: bool=False,xor_units: bool=True,degree: bool=False,lcv: bool=False,binary: bool=False,compact_updates: bool=False,native_check: bool=False,wdeg: bool=False,backjump: bool=False,full_conflicts: bool=False,assumptions: tuple=()) -> SparseResult:
        start=time.perf_counter_ns()
        with self._lock:
            if self._handle is None:raise RuntimeError('sparse index is closed')
            if any(type(flag) is not bool for flag in (heap,active,xor_units,degree,lcv,binary,compact_updates,native_check,wdeg,backjump,full_conflicts)):raise ValueError('search flags must be bool')
            if heap and active:raise ValueError('heap and active strategies cannot be combined')
            flags=int(heap)+2*int(active)+4*int(xor_units)+8*int(degree)+16*int(lcv)+32*int(binary)+64*int(compact_updates)+128*int(wdeg)+256*int(backjump)+512*int(full_conflicts)
            values=self._module.solve(self._handle,max_work,max_state_bytes,flags,assumptions)
            witness,reason,work,nodes,assigned,backtracks,index_bytes,state_bytes,trace,bcalls,bedges,bsolved,jumps,skipped,analysis,conflict_bytes=values
            original_ok=(self._module.check(*self._original,witness,assumptions) if native_check
                         else valid_witness(*self._original,witness,assumptions))
            if reason==0 and not original_ok:raise AssertionError('native SAT fails original constraints')
            if work>max_work:raise AssertionError('work cap exceeded')
        return SparseResult('SAT_VERIFIED' if original_ok else 'UNKNOWN',witness,
                            ('satisfied','budget','exhausted')[reason],work,nodes,assigned,
                            backtracks,index_bytes,state_bytes,f'{trace:016x}',time.perf_counter_ns()-start,bcalls,bedges,bsolved,jumps,skipped,analysis,conflict_bytes)

    def close(self):
        with self._lock:self._handle=None

    def __enter__(self):
        with self._lock:
            if self._handle is None:raise RuntimeError('sparse index is closed')
        return self

    def __exit__(self,*_):self.close()
