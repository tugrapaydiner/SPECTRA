"""Explicit builds for stronger frozen-model native baseline implementations."""
from pathlib import Path
import ctypes as C,hashlib,json,subprocess,time
from .controls import ControlSession

def build(out,kind,blas=None):
    if kind not in ('blas','finite'):raise ValueError('unknown comparator')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    src=Path(__file__).with_name(kind+'_control.cpp');root=Path(__file__).resolve().parents[2]
    lib=out/'control.so';cmd=['g++','-std=c++17','-O3','-mavx2','-fno-fast-math','-ffp-contract=off','-fPIC','-shared',f'-DBP_BASE_RUNTIME="{root}/spectra/_native/ovo/runtime.cpp"']
    sources=[src]
    if kind=='blas':
        if blas is None:raise ValueError('explicit pinned LP64 scipy OpenBLAS library required')
        blas=Path(blas).resolve(strict=True);cmd +=[str(src),str(blas),'-Wl,-rpath,'+str(blas.parent)]
        sources+=[src.with_name('controls.cpp')]
    else:
        cmd +=[f'-DAK_BASE_RUNTIME="{root}/spectra/_native/ovo/runtime.cpp"',f'-DFK_ADAPTIVE="{root}/experiments/adaptive_kernel/runtime.cpp"',f'-DBP_FINITE_RUNTIME="{root}/experiments/finite_kernel/runtime.cpp"',str(src)]
        sources+=[root/'experiments/adaptive_kernel/runtime.cpp',root/'experiments/finite_kernel/runtime.cpp']
    sources += sorted(p for p in (root/'spectra/_native/ovo').rglob('*') if p.suffix in ('.cpp','.hpp'))
    cmd += ['-o',str(lib)];start=time.perf_counter();p=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
    record={'command':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'seconds':time.perf_counter()-start,'sources':{x.relative_to(root).as_posix():hashlib.sha256(x.read_bytes()).hexdigest() for x in sources}}
    if blas:record['external_blas']={'path':str(blas),'sha256':hashlib.sha256(blas.read_bytes()).hexdigest()}
    if p.returncode==0:record['library_sha256']=hashlib.sha256(lib.read_bytes()).hexdigest()
    (out/'build.json').write_text(json.dumps(record,indent=2))
    if p.returncode:raise RuntimeError(p.stderr)
    return lib

class BlasSession(ControlSession):
    def __init__(self,model,library):
        super().__init__(model,library)
        if self.svm:self.close();raise ValueError('BLAS adapter needs a dense network')
        fn=self._lib.ct_blas_config;fn.restype=C.c_char_p;self.blas_config=fn().decode()
        self._run=self._lib.ct_blas_run
        self._run.argtypes=[C.c_void_p,C.POINTER(C.c_uint8),C.c_int,C.c_int,C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_double),C.c_uint64]
        self._run.restype=C.c_int
