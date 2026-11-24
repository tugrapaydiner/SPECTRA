"""Two-plan support with original dependency proofs and minimal migration sets.

Inputs are only the original conflict edges, two proper plans and current queries.
No retained statuses, precomputed contradiction proofs or gold witnesses are used.
These are classical reachability algorithms, not an algorithmic novelty claim.
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
import struct
import subprocess
import sysconfig
import tempfile
import threading
import time
SOURCE=Path(__file__).with_suffix('.cpp')
DEFAULT_BYTES=512*1024**2

def build_migration_session(directory,*,compiler='g++',sanitize=False):
    if platform.system()!='Linux':raise NotImplementedError('Linux builder only')
    if type(sanitize) is not bool:raise TypeError('sanitize must be bool')
    cc=shutil.which(compiler)
    if cc is None:raise FileNotFoundError(compiler)
    dest=Path(directory).resolve();dest.mkdir(parents=True,exist_ok=True)
    library=dest/('_spectra_migration_session'+sysconfig.get_config_var('EXT_SUFFIX'))
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
class MigrationBatch:
    statuses: bytes
    labels: bytes
    proof_kinds: bytes
    proof_offsets: tuple[int,...]
    proof_vertices_le: bytes
    engine_ns: tuple[int,...]
    operations: tuple[int,...]
    elapsed_ns: int

class MigrationRuntime:
    def __init__(self,library):
        spec=importlib.util.spec_from_file_location('_spectra_migration_session',Path(library).resolve())
        if spec is None or spec.loader is None:raise ValueError('invalid native extension path')
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
    def prepare(self,edges,plan_a,plan_b,*,compiled=False,max_bytes=DEFAULT_BYTES):
        return PreparedMigration(self.module,tuple(map(tuple,edges)),tuple(plan_a),tuple(plan_b),compiled,max_bytes)

class PreparedMigration:
    def __init__(self,module,edges,a,b,compiled,cap):
        self._module=module;self._lock=threading.RLock();self._n=len(a)
        self._handle=module.create(edges,a,b,compiled,cap)
    @property
    def info(self):
        with self._lock:
            if self._handle is None:raise RuntimeError('migration session is closed')
            return dict(zip(('vertices','dependency_arcs','closure_bytes','payload_bytes'),self._module.info(self._handle)))
    def solve_raw(self,queries,*,max_bytes=DEFAULT_BYTES):
        started=time.perf_counter_ns()
        qs=tuple(tuple(tuple(item) for item in q) for q in queries)
        with self._lock:
            if self._handle is None:raise RuntimeError('migration session is closed')
            values=self._module.solve(self._handle,qs,max_bytes)
        return MigrationBatch(*values,time.perf_counter_ns()-started)
    def solve_frozen(self,queries,*,max_bytes=DEFAULT_BYTES):
        started=time.perf_counter_ns()
        qs=queries
        if type(qs) is not tuple:raise ValueError('frozen queries must be an exact tuple')
        with self._lock:
            if self._handle is None:raise RuntimeError('migration session is closed')
            values=self._module.solve(self._handle,qs,max_bytes)
        return MigrationBatch(*values,time.perf_counter_ns()-started)
    def close(self):
        with self._lock:self._handle=None
    def __enter__(self):
        if self._handle is None:raise RuntimeError('migration session is closed')
        return self
    def __exit__(self,*_):self.close()

class InvalidMigrationProof(ValueError):pass

class MigrationVerifier:
    """Independent standard-library observer of original edges and request pins.

    It does not read a native index. SAT checking checks every original constraint.
    Optional minimality checking independently reconstructs the required closure.
    UNSAT checking needs only a chain of real conflicts linking opposite pins.
    """
    def __init__(self,edges,plan_a,plan_b):
        self.a=tuple(plan_a);self.b=tuple(plan_b);self.n=len(self.a)
        if not self.n or len(self.b)!=self.n:raise ValueError('plan geometry differs')
        if any(type(c) is not int or not 0<=c<64 for c in self.a+self.b):raise ValueError('invalid colour')
        self.edges=tuple(tuple(e) for e in edges);self.edge_set=set()
        self.dependents=[set() for _ in self.a]
        for e in self.edges:
            if len(e)!=2 or any(type(v) is not int or not 0<=v<self.n for v in e):raise ValueError('bad edge')
            u,v=e
            if self.a[u]==self.a[v] or self.b[u]==self.b[v]:raise ValueError('improper plan')
            self.edge_set.add((min(u,v),max(u,v)))
            if self.b[u]==self.a[v]:self.dependents[u].add(v)
            if self.b[v]==self.a[u]:self.dependents[v].add(u)
    def pins(self,query):
        moving=set();staying=set();empty=set()
        for item in query:
            if type(item) is not tuple or len(item)!=2:raise InvalidMigrationProof('bad query')
            v,m=item
            if type(v) is not int or not 0<=v<self.n or type(m) is not int or not 0<=m<1<<64:raise InvalidMigrationProof('bad pin')
            aa=bool(m>>self.a[v]&1);bb=bool(m>>self.b[v]&1)
            if not aa and not bb:empty.add(v)
            elif self.a[v]!=self.b[v]:
                if not aa:moving.add(v)
                if not bb:staying.add(v)
        return moving,staying,empty
    def verify(self,queries,batch:MigrationBatch,*,minimal=False):
        qcount=len(queries)
        if (type(batch.statuses) is not bytes or type(batch.labels) is not bytes or type(batch.proof_kinds) is not bytes or type(batch.proof_vertices_le) is not bytes
            or len(batch.statuses)!=qcount or len(batch.labels)!=qcount*self.n
            or len(batch.proof_kinds)!=qcount or len(batch.proof_offsets)!=qcount+1
            or len(batch.proof_vertices_le)%4):raise InvalidMigrationProof('output geometry differs')
        offsets=batch.proof_offsets
        if (any(type(x) is not int or x<0 for x in offsets) or offsets[0]!=0 or offsets[-1]*4!=len(batch.proof_vertices_le)
            or any(a>b for a,b in zip(offsets,offsets[1:]))):raise InvalidMigrationProof('bad proof offsets')
        sat=unsat=path_edges=moved_total=0
        for i,q in enumerate(queries):
            moving,staying,empty=self.pins(q)
            status=batch.statuses[i];kind=batch.proof_kinds[i]
            if status not in (0,1):raise InvalidMigrationProof('unknown status')
            raw=batch.proof_vertices_le[4*offsets[i]:4*offsets[i+1]]
            path=tuple(x[0] for x in struct.iter_unpack('<I',raw))
            if len(path)>self.n or any(v>=self.n for v in path):raise InvalidMigrationProof('bad proof vertex')
            if status==0:
                labels=batch.labels[i*self.n:(i+1)*self.n]
                if kind or path or empty:raise InvalidMigrationProof('SAT carries contradiction')
                if any(c not in (a,b) for c,a,b in zip(labels,self.a,self.b)):raise InvalidMigrationProof('label outside plans')
                if any(labels[u]==labels[v] for u,v in self.edges):raise InvalidMigrationProof('original conflict violated')
                if any(not(mask>>labels[v]&1) for v,mask in q):raise InvalidMigrationProof('request violated')
                actual={v for v in range(self.n) if labels[v]!=self.a[v]}
                if minimal:
                    required=set(moving);queue=list(moving)
                    for v in queue:
                        for w in self.dependents[v]:
                            if w not in required:required.add(w);queue.append(w)
                    if actual!=required:raise InvalidMigrationProof('migration set is not inclusion-minimal')
                moved_total+=len(actual);sat+=1
            elif kind==2:
                if len(path)!=1 or path[0] not in empty:raise InvalidMigrationProof('false empty-domain contradiction')
                unsat+=1
            elif kind==1:
                if not path or path[0] not in moving or path[-1] not in staying or len(set(path))!=len(path):raise InvalidMigrationProof('bad contradiction endpoints')
                for u,v in zip(path,path[1:]):
                    if (min(u,v),max(u,v)) not in self.edge_set or self.b[u]!=self.a[v]:raise InvalidMigrationProof('path edge not justified by original conflict')
                path_edges+=len(path)-1;unsat+=1
            else:raise InvalidMigrationProof('missing contradiction proof')
        return {'sat':sat,'unsat':unsat,'proof_path_edges':path_edges,'migrations':moved_total,'minimality_checked':minimal,'precomputed_outcomes_used':False}
