"""Unmodified CatBoost C API and exported-C++ comparators, same raw input boundary."""
from __future__ import annotations
import ctypes as C,hashlib,json,subprocess,time
from pathlib import Path
from contextlib import ExitStack
from spectra.svm_lifetime import _NativeOwner
from .session import input_buffer,TreeSession,options

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build(out,*,upstream=None,export=None,tree_library=None,relocatable=False,target="avx2"):
    if target not in ('portable','avx2'):raise ValueError('unsupported comparator target')
    if (upstream is None)==(export is None):raise ValueError('one comparator source required')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);source=Path(__file__).with_name('catboost_control.cpp');library=out/'control.so'
    cmd=['g++','-std=c++17','-O3','-fno-fast-math','-ffp-contract=off','-shared','-fPIC']
    if target=='avx2':cmd+=['-mavx2']
    sourceids={'wrapper':sha(source)}
    if export is not None:cmd +=[f'-DCB_EXPORT_SOURCE="{Path(export).resolve()}"',str(source)];sourceids['export']=sha(export)
    else:
        upstream=Path(upstream).resolve();cmd +=['-I'+str(upstream),str(source)]
        cmd += ['-L'+str(upstream),'-l:libcatboostmodel.so','-Wl,-rpath,$ORIGIN'] if relocatable else [str(upstream/'libcatboostmodel.so'),'-Wl,-rpath,'+str(upstream)]
        sourceids.update(header=sha(upstream/'c_api.h'),official_library=sha(upstream/'libcatboostmodel.so'))
    if tree_library is not None:
        if export is not None:raise ValueError('refinement uses the official library, not C++ export')
        tree_library=Path(tree_library).resolve(strict=True)
        cmd +=['-DCB_WITH_REFINEMENT']
        cmd += ['-L'+str(tree_library.parent),'-l:'+tree_library.name,'-Wl,-rpath,$ORIGIN'] if relocatable else [str(tree_library),'-Wl,-rpath,'+str(tree_library.parent)]
        sourceids['tree_library']=sha(tree_library)
    cmd +=['-o',str(library)];start=time.perf_counter();r={'command':cmd,'sources':sourceids,'returncode':None}
    try:
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=180);r.update(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    except (OSError,subprocess.TimeoutExpired) as e:r['error']=str(e)
    r['wall_seconds']=time.perf_counter()-start
    if r['returncode']==0:r['library_sha256']=sha(library)
    (out/'build.json').write_text(json.dumps(r,indent=2))
    if r['returncode']!=0:raise RuntimeError('comparator failed; see build.json')
    return library

class CatBoostSession(_NativeOwner):
    def __init__(self,model,library,maximum,*,expected_sha256,source_json_sha256=None):
        self._init_lifetime('native CatBoost comparator')
        self.source_json_sha256=source_json_sha256
        with Path(model).open('rb') as f:raw=f.read(64*1024**2+1)
        self.sha256=hashlib.sha256(raw).hexdigest()
        if self.sha256!=expected_sha256:raise ValueError('CatBoost model changed')
        if type(maximum) is not int or not 1<=maximum<=255:raise ValueError('invalid code maximum')
        self._lib=lib=C.CDLL(str(Path(library).resolve(strict=True)));lib.cb_abi.restype=C.c_int
        if lib.cb_abi()!=1:raise ValueError('comparator ABI')
        lib.cb_error.restype=C.c_char_p;lib.cb_create.argtypes=[C.c_void_p,C.c_uint64,C.c_int];lib.cb_create.restype=C.c_void_p
        lib.cb_destroy.argtypes=[C.c_void_p];lib.cb_destroy.restype=None
        lib.cb_info.argtypes=[C.c_void_p,C.POINTER(C.c_uint32)];lib.cb_info.restype=C.c_int
        lib.cb_run.argtypes=[C.c_void_p,C.POINTER(C.c_uint8),C.c_int,C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_double),C.c_uint64];lib.cb_run.restype=C.c_int
        self._refine=None
        if hasattr(lib,'cb_refine'):
            self._refine=lib.cb_refine
            self._refine.argtypes=[C.c_void_p,C.c_void_p,C.POINTER(C.c_uint8),C.c_int,C.c_int,C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_uint32),C.POINTER(C.c_uint8)];self._refine.restype=C.c_int
            self._tree_run_address=C.cast(lib.st_run,C.c_void_p).value
        b=C.create_string_buffer(raw,len(raw));h=lib.cb_create(b,len(raw),maximum)
        if not h:raise ValueError(lib.cb_error().decode(errors='replace'))
        self._adopt(h,lib,'cb_destroy')
        with self._operation() as handle:
            out=(C.c_uint32*3)();self._check(lib.cb_info(handle,out));self.features,self.classes,self.maximum=tuple(out)
    def _check(self,status):
        if status:raise ValueError(self._lib.cb_error().decode(errors='replace'))
    def _invoke(self,values,scores):
        with input_buffer(values,self.features) as (data,n),self._operation() as h:
            out=(C.c_int*n)();cells=n*self.classes if scores else 0
            if cells>8000000:raise ValueError('score cap')
            result=(C.c_double*cells)() if scores else None
            self._check(self._lib.cb_run(h,data,n,self.features,out,result,cells));return bytes(result) if scores else list(out)
    def refine(self,compact,values,*,checkpoint=0,inspect=False):
        options(checkpoint,True,False)
        if type(compact) is not TreeSession or self.source_json_sha256 is None or compact.meta['source_sha256']!=self.source_json_sha256:
            raise ValueError('refinement requires a trusted source JSON/CBM pairing')
        if compact.meta['maximum']!=self.maximum:raise ValueError('different declared domains')
        fn=self._refine
        if fn is None or compact._run_address!=self._tree_run_address:raise ValueError('refinement must use its exact linked tree runtime')
        with input_buffer(values,self.features) as (data,n),ExitStack() as stack:
            handles={id(w):stack.enter_context(w._operation()) for w in sorted((self,compact),key=id)}
            out=(C.c_int*n)();steps=(C.c_uint32*n)();fallback=(C.c_uint8*n)()
            self._check(fn(handles[id(self)],handles[id(compact)],data,n,self.features,checkpoint,out,steps,fallback))
            return {'indices':list(out),'trees_evaluated':list(steps),'fallback':list(fallback)} if inspect else list(out)
    def predict_buffer(self,values):return self._invoke(values,False)
    def scores(self,values):return self._invoke(values,True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--upstream',type=Path);p.add_argument('--export',type=Path);p.add_argument('--tree-library',type=Path);p.add_argument('--relocatable',action='store_true');p.add_argument('--target',choices=['portable','avx2'],default='avx2');a=p.parse_args();print(build(a.out,upstream=a.upstream,export=a.export,tree_library=a.tree_library,relocatable=a.relocatable,target=a.target))
