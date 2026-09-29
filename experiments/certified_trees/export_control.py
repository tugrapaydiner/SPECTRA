"""Original unmodified CatBoost-generated C++ with a raw-code batch wrapper.

This baseline is not CatBoost's optimized shared model library; both are measured.
"""
from pathlib import Path
import ctypes as C,hashlib,json,subprocess,time
from spectra.svm_lifetime import _NativeOwner
TEMPLATE=r'''
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cfenv>
#include "model.cpp"
extern "C" int export_run(const uint8_t*q,int rows,int d,int maximum,int32_t*out,double*scores,uint64_t nscore){
 try{
  if(rows<0||rows>65536||d!=@D@||maximum!=@MAX@||uint64_t(rows)*d>8000000||(!q&&rows)||(!out&&rows)||(scores&&nscore!=uint64_t(rows)*@C@)||std::fegetround()!=FE_TONEAREST)return 1;
  for(uint64_t i=0;i<uint64_t(rows)*d;++i)if(q[i]>maximum)return 2;
  std::vector<float> input(d);std::vector<int32_t> result(rows);std::vector<double> all(scores?nscore:0);
  for(int r=0;r<rows;++r){for(int f=0;f<d;++f)input[f]=float(q[uint64_t(r)*d+f]);auto values=ApplyCatboostModelMulti(input);
   if(values.size()!=@C@)return 3;for(double v:values)if(!std::isfinite(v))return 4;
   result[r]=int(std::max_element(values.begin(),values.end())-values.begin());if(scores)std::copy(values.begin(),values.end(),all.begin()+uint64_t(r)*@C@);
  }
  std::copy(result.begin(),result.end(),out);if(scores)std::copy(all.begin(),all.end(),scores);return 0;
 }catch(...){return 5;}
}
'''

def build(model,out):
    model=Path(model);out=Path(out);out.mkdir(parents=True,exist_ok=False);fit=json.loads((model/'FIT.json').read_text())
    raw=(model/'model.cpp').read_bytes();(out/'model.cpp').write_bytes(raw)
    wrapper=TEMPLATE.replace('@D@',str(fit['features'])).replace('@MAX@',str(fit['maximum'])).replace('@C@',str(len(fit['classes'])))
    (out/'wrapper.cpp').write_text(wrapper);lib=out/'export.so'
    cmd=['g++','-std=c++17','-O3','-mavx2','-fno-fast-math','-ffp-contract=off','-fPIC','-shared',str(out/'wrapper.cpp'),'-o',str(lib)]
    t=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
    rec={'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'seconds':time.perf_counter()-t,
         'source_sha256':hashlib.sha256(raw).hexdigest(),'wrapper_sha256':hashlib.sha256(wrapper.encode()).hexdigest(),**fit}
    if r.returncode==0:rec['library_sha256']=hashlib.sha256(lib.read_bytes()).hexdigest()
    (out/'build.json').write_text(json.dumps(rec,indent=2))
    if r.returncode:raise RuntimeError('original C++ export build failed')
    return lib

class ExportSession:
    def __init__(self,folder):
        folder=Path(folder);m=json.loads((folder/'build.json').read_text());self.features=m['features'];self.maximum=m['maximum'];self.classes=len(m['classes'])
        self.lib=C.CDLL(str((folder/'export.so').resolve(strict=True)));self.run=self.lib.export_run
        self.run.argtypes=[C.POINTER(C.c_uint8),C.c_int,C.c_int,C.c_int,C.POINTER(C.c_int32),C.POINTER(C.c_double),C.c_uint64];self.run.restype=C.c_int
    def _call(self,buf,scores):
        try:v=memoryview(buf)
        except TypeError as e:raise ValueError('uint8 buffer required') from e
        data=None
        try:
            if v.format!='B' or v.readonly or not v.c_contiguous or v.ndim not in (1,2) or (v.ndim==2 and v.shape[1]!=self.features):raise ValueError('contiguous writable uint8 features required')
            n=v.nbytes//self.features
            if v.nbytes%self.features or v.nbytes>8000000 or n>65536:raise ValueError('export input cap')
            data=(C.c_uint8*v.nbytes).from_buffer(v) if v.nbytes else None
            out=(C.c_int32*n)();count=n*self.classes if scores else 0;result=(C.c_double*count)() if scores else None
            rc=self.run(data,n,self.features,self.maximum,out,result,count)
            if rc:raise ValueError('original export rejected input: '+str(rc))
            return bytes(result) if scores else list(out)
        finally:del data;v.release()
    def predict_buffer(self,q):return self._call(q,False)
    def scores(self,q):return self._call(q,True)
