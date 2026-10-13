"""Sparse two-plan migration; no quadratic reachability table.

Modes share one exact SCC index, output format and original-path proof contract.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, importlib.util, json, os, platform, shutil, subprocess, sysconfig, tempfile, threading, time
from pathlib import Path
from experiments.real_traffic.migration_session import MigrationBatch
SOURCE=Path(__file__).with_suffix('.cpp')
DEFAULT_BYTES=512*1024**2
MODES={'bfs':0,'tiled':1,'auto':2}

def build_compact_migration(directory,*,compiler='g++',sanitize=False):
    if platform.system()!='Linux':raise NotImplementedError('Linux builder only')
    if type(sanitize) is not bool:raise TypeError('sanitize must be bool')
    cc=shutil.which(compiler)
    if cc is None:raise FileNotFoundError(compiler)
    dest=Path(directory).resolve();dest.mkdir(parents=True,exist_ok=True)
    library=dest/('_spectra_compact_migration'+sysconfig.get_config_var('EXT_SUFFIX'))
    receipt=dest/'build.json'
    if library.exists() or receipt.exists():raise FileExistsError('build outputs already exist')
    flags=['-std=c++17','-shared','-fPIC','-Wall','-Wextra','-Werror','-fno-fast-math','-ffp-contract=off']
    flags+=['-O1','-g','-fsanitize=undefined','-fno-sanitize-recover=all'] if sanitize else ['-O3','-DNDEBUG']
    started=time.perf_counter_ns()
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        target=Path(tmp)/library.name
        command=[cc,*flags,'-I'+sysconfig.get_path('include'),str(SOURCE),'-o',str(target)]
        p=subprocess.run(command,capture_output=True,text=True,timeout=120)
        if p.returncode:raise RuntimeError(p.stderr)
        record={'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'library_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'command':command,'elapsed_ns':time.perf_counter_ns()-started,'sanitize':sanitize,'compiler':subprocess.check_output([cc,'--version'],text=True),'stderr':p.stderr}
        os.link(target,library)
        with receipt.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    return library

@dataclass(frozen=True)
class CompactBatch(MigrationBatch):
    mode: str
    dag_edge_visits: int
    proof_edge_visits: int
    tiles: int
    scratch_payload_bytes: int
    batch_admission_bound: int

class CompactMigrationRuntime:
    def __init__(self, library):
        spec=importlib.util.spec_from_file_location('_spectra_compact_migration',Path(library).resolve())
        if spec is None or spec.loader is None: raise ValueError('invalid trusted native library')
        self.module=importlib.util.module_from_spec(spec); spec.loader.exec_module(self.module)
    def prepare(self, edges, plan_a, plan_b, *, max_bytes=DEFAULT_BYTES, sparsify=True):
        return PreparedCompactMigration(self.module,tuple(map(tuple,edges)),tuple(plan_a),tuple(plan_b),max_bytes,sparsify)

class PreparedCompactMigration:
    def __init__(self,module,edges,a,b,cap,sparse):
        self._module=module; self._lock=threading.RLock()
        self._handle=module.create(edges,a,b,cap,sparse)
    @property
    def info(self):
        with self._lock:
            if self._handle is None: raise RuntimeError('migration session is closed')
            return dict(zip(('vertices','changed_vertices','components','proof_dependency_arcs','component_arcs','payload_bytes','build_payload_bound','source_dependency_arcs'),self._module.info(self._handle)))
    def solve_raw(self, queries, *, mode='auto', max_bytes=DEFAULT_BYTES):
        if type(mode) is not str or mode not in MODES: raise ValueError('unknown execution mode')
        begin=time.perf_counter_ns()
        qs=tuple(tuple(tuple(pin) for pin in q) for q in queries)
        with self._lock:
            if self._handle is None: raise RuntimeError('migration session is closed')
            s,l,k,offsets,path,chosen,edges,proof_edges,tiles,scratch,bound=self._module.solve(self._handle,qs,MODES[mode],max_bytes)
        # A tile has no meaningful individual query solve time. Do not invent one.
        return CompactBatch(s,l,k,offsets,path,(),(),time.perf_counter_ns()-begin,
                            ('bfs','tiled')[chosen],edges,proof_edges,tiles,scratch,bound)
    def solve_frozen(self, queries, *, mode='auto', max_bytes=DEFAULT_BYTES):
        if type(mode) is not str or mode not in MODES: raise ValueError('unknown execution mode')
        begin=time.perf_counter_ns()
        qs=queries
        if type(qs) is not tuple:raise ValueError('frozen queries must be an exact tuple')
        with self._lock:
            if self._handle is None: raise RuntimeError('migration session is closed')
            s,l,k,offsets,path,chosen,edges,proof_edges,tiles,scratch,bound=self._module.solve(self._handle,qs,MODES[mode],max_bytes)
        # A tile has no meaningful individual query solve time. Do not invent one.
        return CompactBatch(s,l,k,offsets,path,(),(),time.perf_counter_ns()-begin,
                            ('bfs','tiled')[chosen],edges,proof_edges,tiles,scratch,bound)
    def close(self):
        with self._lock:self._handle=None
    def __enter__(self):
        with self._lock:
            if self._handle is None: raise RuntimeError('migration session is closed')
        return self
    def __exit__(self,*args): self.close()
