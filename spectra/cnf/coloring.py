"""Opt-in finite-domain graph/list colouring with original-input checking.

The untrained backend supports 0..64 colours and bounded exact search. Every
published SAT result is checked from the original edges and allowed-colour masks.
No UNSAT claim, default promotion, general CNF fallback, or import-time compilation.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
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

SOURCE=Path(__file__).resolve().parents[1]/'_native'/'coloring.cpp'
DEFAULT_BYTES=64*1024*1024

def build_coloring_runtime(directory: str|Path, *, compiler: str='g++', sanitize: bool=False) -> Path:
    if platform.system()!='Linux':raise NotImplementedError('Linux builder only; other platforms unvalidated')
    if type(sanitize) is not bool:raise TypeError('sanitize must be bool')
    cc=shutil.which(compiler)
    if not cc:raise FileNotFoundError(compiler)
    dest=Path(directory).resolve();dest.mkdir(parents=True,exist_ok=True)
    library=dest/('_spectra_coloring'+sysconfig.get_config_var('EXT_SUFFIX'))
    receipt=dest/'build.json'
    if library.exists() or receipt.exists():raise FileExistsError('build outputs already exist')
    flags=['-std=c++17','-shared','-fPIC','-Wall','-Wextra','-Werror','-fno-fast-math','-ffp-contract=off']
    flags+=['-O1','-g','-fsanitize=undefined','-fno-sanitize-recover=all'] if sanitize else ['-O3','-DNDEBUG']
    start=time.perf_counter_ns()
    with tempfile.TemporaryDirectory(dir=dest,prefix='.coloring-') as tmp:
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


def check_coloring(n: int, k: int, edges: tuple, labels: tuple, masks: tuple=()) -> bool:
    """Independent standard-library observer used by audits and differential tests."""
    if type(n) is not int or type(k) is not int or not 0 <= n <= 100000 or not 0 <= k <= 64:
        return False
    if any(type(value) is not tuple for value in (edges, masks, labels)):
        return False
    if len(labels) != n or len(masks) not in (0,n) or len(edges)>2000000:
        return False
    if any(type(c) is not int or not 0 <= c < k for c in labels):
        return False
    for i, mask in enumerate(masks):
        if type(mask) is not int or not 0 <= mask < (1 << k) or not (mask >> labels[i]) & 1:
            return False
    for edge in edges:
        if type(edge) is not tuple or len(edge) != 2:
            return False
        a,b=edge
        if any(type(v) is not int or not 0 <= v < n for v in edge) or labels[a] == labels[b]:
            return False
    return True


@dataclass(frozen=True)
class ColoringResult:
    status: str
    labels: tuple[int, ...]
    reason: str
    work: int
    branches: int
    backtracks: int
    domain_reductions: int
    binary_calls: int
    binary_edges: int
    binary_solutions: int
    index_payload_bytes: int
    state_payload_bytes: int
    trail_peak: int
    trace: str
    elapsed_ns: int

    def record(self) -> dict:
        return {**self.__dict__, 'labels':list(self.labels),
                'schema':'spectra.coloring.v1', 'learned':False}


class ColoringRuntime:
    """Load one explicitly built trusted extension; no external solver dependency."""
    def __init__(self, library: str | Path):
        spec=importlib.util.spec_from_file_location('_spectra_coloring',Path(library).resolve())
        if spec is None or spec.loader is None:
            raise ValueError('invalid native extension location')
        self._module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._module)
        if self._module.abi()!=2:
            raise ValueError('unsupported colouring ABI; explicitly rebuild')

    def check(self, n: int, k: int, edges: tuple, labels: tuple, masks: tuple=()) -> bool:
        return self._module.check(n,k,edges,masks,labels)

    def prepare(self, n: int, k: int, edges: tuple, *, masks: tuple=(),
                max_build_bytes: int=DEFAULT_BYTES) -> PreparedColoring:
        return PreparedColoring(self,n,k,edges,masks,max_build_bytes)

    def solve(self, n: int, k: int, edges: tuple, *, masks: tuple=(),
              max_work: int=1000000, max_build_bytes: int=DEFAULT_BYTES,
              max_state_bytes: int=DEFAULT_BYTES, symmetry: bool=True,
              binary: bool=True, buckets: bool=True) -> ColoringResult:
        """Original graph -> fresh index/search -> checked labels -> disposal.

        Module loading, building, file parsing and process startup are excluded.
        The two payload caps exclude Python inputs/output and allocator overhead;
        they are not process RSS or execution-time limits.
        """
        start=time.perf_counter_ns()
        with self.prepare(n,k,edges,masks=masks,max_build_bytes=max_build_bytes) as p:
            result=p.solve(max_work=max_work,max_state_bytes=max_state_bytes,
                           symmetry=symmetry,binary=binary,buckets=buckets)
        return replace(result,elapsed_ns=time.perf_counter_ns()-start)


class PreparedColoring:
    def __init__(self, runtime, n, k, edges, masks, max_build_bytes):
        self._lock=threading.RLock()
        self._module=runtime._module
        self._handle=self._module.create(n,k,edges,masks,max_build_bytes)
        self._original=(n,k,edges,masks)

    def solve(self, *,max_work: int=1000000,max_state_bytes: int=DEFAULT_BYTES,
              symmetry: bool=True,binary: bool=True, buckets: bool=True) -> ColoringResult:
        start=time.perf_counter_ns()
        with self._lock:
            if self._handle is None:
                raise RuntimeError('colouring index is closed')
            values=self._module.solve(self._handle,max_work,max_state_bytes,symmetry,binary,buckets)
            labels,reason,*counts=values
            verified=self._module.check(*self._original,labels)
            if reason==0 and not verified:
                raise AssertionError('native solution fails original graph/list constraints')
            work,branches,backs,reductions,bcalls,bedges,bsols,index_bytes,state_bytes,peak,trace=counts
            if work>max_work or state_bytes>max_state_bytes:
                raise AssertionError('native work/state cap exceeded')
        return ColoringResult('SAT_VERIFIED' if verified else 'UNKNOWN',labels,
                              ('satisfied','budget','exhausted')[reason],
                              work,branches,backs,reductions,bcalls,bedges,bsols,
                              index_bytes,state_bytes,peak,f'{trace:016x}',
                              time.perf_counter_ns()-start)

    def close(self) -> None:
        with self._lock:
            self._handle=None

    def __enter__(self):
        with self._lock:
            if self._handle is None:
                raise RuntimeError('colouring index is closed')
        return self

    def __exit__(self,*_):
        self.close()
