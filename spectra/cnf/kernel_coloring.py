"""Experimental two-choice implication kernel with bounded multi-choice search.

Only exact raw graph/list colouring is supported. The binary part is compiled once
per prepared index, then assumptions are propagated reversibly through its SCC DAG.
All SAT answers are independently checked from original tuples. UNKNOWN is not UNSAT.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import hashlib,importlib.util,json,os,platform,shutil,subprocess,sysconfig,tempfile,threading,time
from pathlib import Path
SOURCE=Path(__file__).resolve().parents[1]/'_native'/'kernel_coloring.cpp'
DEFAULT_BYTES=64*1024*1024

def build_kernel_coloring_runtime(directory: str|Path, *, compiler: str='g++', sanitize: bool=False) -> Path:
    if platform.system()!='Linux':raise NotImplementedError('Linux builder only; other platforms unvalidated')
    if type(sanitize) is not bool:raise TypeError('sanitize must be bool')
    cc=shutil.which(compiler)
    if not cc:raise FileNotFoundError(compiler)
    dest=Path(directory).resolve();dest.mkdir(parents=True,exist_ok=True)
    library=dest/('_spectra_kernel_coloring'+sysconfig.get_config_var('EXT_SUFFIX'))
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


@dataclass(frozen=True)
class KernelColoringResult:
    status: str
    labels: tuple[int,...]
    reason: str
    search_work: int
    core_branches: int
    backtracks: int
    domain_changes: int
    component_assignments: int
    events: int
    core_vertices: int
    binary_vertices: int
    components: int
    implication_arcs: int
    index_payload_bytes: int
    build_payload_bound: int
    state_payload_bytes: int
    trail_peak: int
    trace: str
    condensation_arcs: int
    elapsed_ns: int

    def record(self):
        return {**self.__dict__,'labels':list(self.labels),'schema':'spectra.kernel_coloring.v1','learned':False}


class KernelColoringRuntime:
    def __init__(self,library: str|Path):
        spec=importlib.util.spec_from_file_location('_spectra_kernel_coloring',Path(library).resolve())
        if spec is None or spec.loader is None:raise ValueError('invalid native module location')
        self._module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self._module)
        if self._module.abi()!=1:raise ValueError('unsupported kernel-colouring ABI')

    def check(self,n,k,edges,labels,masks=()):
        return self._module.check(n,k,edges,masks,labels)

    def prepare(self,n:int,k:int,edges:tuple,*,masks:tuple=(),max_build_bytes:int=DEFAULT_BYTES):
        return PreparedKernelColoring(self,n,k,edges,masks,max_build_bytes)

    def solve(self,n:int,k:int,edges:tuple,*,masks:tuple=(),max_search_work:int=1000000,
              max_build_bytes:int=DEFAULT_BYTES,max_state_bytes:int=DEFAULT_BYTES,symmetry:bool=True):
        """Fresh preparation, search, original-input checking and disposal.

        max_search_work limits search events, not SCC construction or whole-job time.
        Native array payload bounds are not total process RSS. Use an outer deadline.
        """
        start=time.perf_counter_ns()
        with self.prepare(n,k,edges,masks=masks,max_build_bytes=max_build_bytes) as index:
            result=index.solve(max_search_work=max_search_work,max_state_bytes=max_state_bytes,symmetry=symmetry)
        return replace(result,elapsed_ns=time.perf_counter_ns()-start)


class PreparedKernelColoring:
    def __init__(self,runtime,n,k,edges,masks,max_build_bytes):
        self._lock=threading.RLock();self._module=runtime._module
        self._handle=self._module.create(n,k,edges,masks,max_build_bytes)
        self._original=(n,k,edges,masks)

    def solve(self,*,max_search_work:int=1000000,max_state_bytes:int=DEFAULT_BYTES,symmetry:bool=True):
        start=time.perf_counter_ns()
        with self._lock:
            if self._handle is None:raise RuntimeError('kernel index is closed')
            labels,reason,*values=self._module.solve(self._handle,max_search_work,max_state_bytes,symmetry)
            verified=self._module.check(*self._original,labels)
            if reason==0 and not verified:raise AssertionError('kernel answer fails original constraints')
            if len(values)!=16:raise AssertionError('native counter shape differs')
            if values[0]>max_search_work or values[12]>max_state_bytes:raise AssertionError('native resource cap exceeded')
            values[14]=f'{values[14]:016x}'
        return KernelColoringResult('SAT_VERIFIED' if verified else 'UNKNOWN',labels,
            ('satisfied','search_budget','core_exhausted','kernel_contradiction','self_loop','empty_domain')[reason],
            *values,time.perf_counter_ns()-start)

    def close(self):
        with self._lock:self._handle=None

    def __enter__(self):
        with self._lock:
            if self._handle is None:raise RuntimeError('kernel index is closed')
        return self

    def __exit__(self,*_):self.close()
