"""Exact persistent list-colouring quotient; opt-in, classical, dependency-free.

Modes: none (no contraction), parity (same-list parity only), scc (full binary
implication equivalences), hybrid (parity followed by SCC). All queries are
intersected with original allowed-colour masks. Complete byte witnesses are lifted
and checked from original constraints before being reported SAT_VERIFIED.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, importlib.util, json, os, platform, shutil, subprocess
import sysconfig, tempfile, threading, time
from pathlib import Path
SOURCE=Path(__file__).resolve().parents[1]/'_native'/'quotient_query.cpp'
DEFAULT_BYTES=64*1024*1024
MODES={'none':0,'parity':1,'scc':2,'hybrid':3}
_RAW_CERTIFICATE_FIELDS={
    'schema','n','k','impossible','palettes','initial','node',
    'colour0','colour1','wide','offsets','arcs',
}


def _canonical_certificate(raw: dict) -> dict:
    """Convert the ABI-v3 native CSR export into the public certificate schema.

    Native ABI v3 exports each residual row as ``(source, target, forbidden)``.
    The independently verified public form removes the redundant source column,
    supplies atom starts explicitly, and lets ``offsets`` delimit each source.
    Refuse malformed native evidence rather than repairing or guessing it.
    """
    if type(raw) is not dict or set(raw) != _RAW_CERTIFICATE_FIELDS:
        raise AssertionError('native certificate field inventory differs')
    palettes=raw['palettes']
    if (type(palettes) is not list
            or any(type(value) is not int or not 0 <= value < 1 << 64
                   for value in palettes)):
        raise AssertionError('native certificate palette bank differs')
    start=[0]
    for palette in palettes:
        start.append(start[-1]+palette.bit_count())
    atoms=start[-1]
    offsets=raw['offsets']
    triples=raw['arcs']
    if (type(offsets) is not list or len(offsets) != atoms+1
            or any(type(value) is not int for value in offsets)
            or not offsets or offsets[0] != 0
            or any(left > right for left,right in zip(offsets,offsets[1:]))):
        raise AssertionError('native certificate CSR offsets differ')
    if type(triples) is not list or offsets[-1] != len(triples):
        raise AssertionError('native certificate arc inventory differs')
    arcs=[]
    for source in range(atoms):
        for index in range(offsets[source],offsets[source+1]):
            item=triples[index]
            if type(item) is not tuple or len(item) != 3:
                raise AssertionError('native certificate arc row differs')
            observed_source,target,forbidden=item
            if (type(observed_source) is not int or observed_source != source
                    or type(target) is not int or not 0 <= target < len(palettes)
                    or type(forbidden) is not int
                    or not 0 <= forbidden < 1 << 64):
                raise AssertionError('native certificate arc geometry differs')
            arcs.append([target,forbidden])
    result=dict(raw)
    result['start']=start
    result['arcs']=arcs
    return result


def build_quotient_runtime(directory: str|Path, *, compiler: str='g++', sanitize: bool=False) -> Path:
    if platform.system()!='Linux':raise NotImplementedError('Linux builder only; other platforms unvalidated')
    if type(sanitize) is not bool:raise TypeError('sanitize must be bool')
    cc=shutil.which(compiler)
    if not cc:raise FileNotFoundError(compiler)
    dest=Path(directory).resolve();dest.mkdir(parents=True,exist_ok=True)
    library=dest/('_spectra_quotient_query'+sysconfig.get_config_var('EXT_SUFFIX'))
    receipt=dest/'build.json'
    if library.exists() or receipt.exists():raise FileExistsError('build outputs already exist')
    flags=['-std=c++17','-shared','-fPIC','-Wall','-Wextra','-Werror','-Wno-free-nonheap-object','-fno-fast-math','-ffp-contract=off']
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


@dataclass(frozen=True)
class QueryResult:
    status: str
    labels: bytes
    reason: str
    work: int
    branches: int
    backtracks: int
    reductions: int
    trail_peak: int
    state_payload_bytes: int
    trace: int
    quotient_vertices: int
    index_payload_bytes: int
    quotient_arcs: int
    elapsed_ns: int

    def record(self):
        return {**self.__dict__, 'labels':list(self.labels),
                'schema':'spectra.quotient_query.v1', 'learned':False}

class QuotientRuntime:
    def __init__(self, library: str|Path):
        spec=importlib.util.spec_from_file_location('_spectra_quotient_query',Path(library).resolve())
        if spec is None or spec.loader is None:
            raise ValueError('invalid trusted native extension path')
        self._module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._module)
        if self._module.abi()!=3:
            raise ValueError('unsupported quotient ABI; rebuild explicitly')

    def checker(self,n,k,edges,masks=(),max_bytes=DEFAULT_BYTES):
        return OriginalChecker(self._module,n,k,edges,masks,max_bytes)

    def prepare(self,n:int,k:int,edges:tuple,*,masks:tuple=(),mode:str='hybrid',
                max_build_bytes:int=DEFAULT_BYTES):
        if mode not in MODES:
            raise ValueError('unknown quotient mode')
        return PreparedQuotient(self,n,k,edges,masks,mode,max_build_bytes)

    def check(self,n,k,edges,labels,masks=(),restrictions=()):
        return self._module.check(n,k,edges,masks,restrictions,labels)

class PreparedQuotient:
    def __init__(self,runtime,n,k,edges,masks,mode,max_build_bytes):
        self._module=runtime._module
        self._lock=threading.RLock()
        self._checker=runtime.checker(n,k,edges,masks,max_build_bytes)
        self._handle=self._module.create(n,k,edges,masks,MODES[mode],max_build_bytes-self._checker.payload_bytes)
        self._original=(n,k,edges,masks)
        self.mode=mode

    @property
    def info(self):
        with self._lock:
            if self._handle is None: raise RuntimeError('quotient is closed')
            values=self._module.info(self._handle)
        info=dict(zip(('vertices','quotient_vertices','atoms','index_payload_bytes',
                       'build_payload_bound','quotient_arcs'),values))
        info['original_checker_payload_bytes']=self._checker.payload_bytes
        info['total_owned_index_payload_bytes']=values[3]+self._checker.payload_bytes
        info['total_build_payload_bound']=values[4]+self._checker.payload_bytes
        return info

    def check(self, labels: bytes, restrictions: tuple=()) -> bool:
        """Run the separately owned original-input checker."""
        with self._lock:
            if self._handle is None:
                raise RuntimeError('quotient is closed')
            return self._checker.check(labels, restrictions)

    def certificate(self) -> dict:
        """Export canonical integer-only evidence for independent audit.

        Export/verification is optional, not silently excluded from a claimed
        certified-setup cost. The ordinary solve contract still checks all edges.
        """
        with self._lock:
            if self._handle is None:raise RuntimeError('quotient is closed')
            raw=self._module.certificate(self._handle)
        return _canonical_certificate(raw)

    def solve_support(self, restrictions: tuple = ()) -> tuple[bytes, int, int, int]:
        """Solve an arc-free quotient natively and independently check the witness."""
        with self._lock:
            if self._handle is None:
                raise RuntimeError('quotient is closed')
            labels, reason, touched, changed = self._module.support_solve(
                self._handle, restrictions)
            if reason == 0 and not self.check(labels, restrictions):
                raise AssertionError('support-table answer fails original constraints')
            return labels, reason, touched, changed

    def solve(self,restrictions:tuple=(),*,max_work:int=1000000,
              max_state_bytes:int=DEFAULT_BYTES,core_first:bool=True) -> QueryResult:
        start=time.perf_counter_ns()
        with self._lock:
            if self._handle is None: raise RuntimeError('quotient is closed')
            labels,reason,*counts=self._module.solve(self._handle,restrictions,max_work,max_state_bytes,core_first)
            if reason==0 and not self._checker.check(labels,restrictions):
                raise AssertionError('quotient answer fails original graph/list/query constraints')
            if counts[0]>max_work or counts[5]>max_state_bytes:
                raise AssertionError('resource bound exceeded')
        return QueryResult('SAT_VERIFIED' if reason==0 else 'UNKNOWN',labels,
            ('satisfied','budget','exhausted','base_contradiction','restriction_conflict')[reason],
            *counts,time.perf_counter_ns()-start)

    def close(self):
        with self._lock:
            self._handle=None
            self._checker.close()

    def __enter__(self):
        with self._lock:
            if self._handle is None: raise RuntimeError('quotient is closed')
        return self

    def __exit__(self,*_): self.close()

class OriginalChecker:
    """Independently owned original edges/masks; full verification on every call.

    Input types and ranges are validated once for an immutable static graph. All
    vertices, edges and current restrictions are then checked for every full byte
    witness. No quotient/search index or previous validity result is reused.
    """
    def __init__(self,module,n,k,edges,masks,max_bytes):
        self._module=module
        self._handle=module.observer_create(n,k,edges,masks,max_bytes)
        self.payload_bytes=module.observer_size(self._handle)
    def check(self,labels,restrictions=()):
        if self._handle is None:raise RuntimeError('original checker is closed')
        return self._module.observer_check(self._handle,restrictions,labels)
    def cache_match(self,models,restrictions=()):
        """Eligibility only. A hit still requires full original-input checking."""
        if self._handle is None:raise RuntimeError('original checker is closed')
        return self._module.observer_cache_probe(self._handle,restrictions,tuple(models))
    def close(self):self._handle=None
