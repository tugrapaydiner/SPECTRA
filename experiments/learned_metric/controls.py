"""Same-input native linear/MLP controls; exported frozen weights, no fitting.

Each SIMD lane is an output unit, retaining scalar input-term order. All layers,
normalization, output decoding and copies are charged by the public adapter.
"""
from __future__ import annotations
import argparse,ctypes as C,hashlib,json,subprocess,time
from pathlib import Path

TEMPLATE=r'''
#include <cstdint>
#include <cmath>
#include <vector>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
@DATA@
void linear(const double* x,const double* weights,const double* bias,int inputs,int outputs,double* y,bool relu) {
 int o=0;
#if defined(__AVX2__)
 for(;o+4<=outputs;o+=4){
  __m256d acc=_mm256_setzero_pd();
  for(int i=0;i<inputs;++i)acc=_mm256_add_pd(acc,_mm256_mul_pd(_mm256_set1_pd(x[i]),_mm256_loadu_pd(weights+i*outputs+o)));
  acc=_mm256_add_pd(acc,_mm256_loadu_pd(bias+o));
  if(relu)acc=_mm256_max_pd(acc,_mm256_setzero_pd());
  _mm256_storeu_pd(y+o,acc);
 }
#endif
 for(;o<outputs;++o){double sum=0;for(int i=0;i<inputs;++i)sum+=x[i]*weights[i*outputs+o];sum+=bias[o];y[o]=relu&&sum<0?0:sum;}
}
extern "C" int ctl_run(const uint8_t* q,int rows,int d,int* out){
 if(rows<0||rows>65536||d!=D||int64_t(rows)*D>8000000|| (rows&&(!q||!out)))return 1;
 for(int64_t k=0;k<int64_t(rows)*D;++k)if(q[k]>MAXIMUM)return 2;
 double input[D],h0[H],h1[H],scores[CLASSES];
 for(int r=0;r<rows;++r){
  for(int j=0;j<D;++j)input[j]=double(q[r*D+j])/double(MAXIMUM);
  @RUN@
  int winner=0;for(int c=1;c<CLASSES;++c)if(scores[c]>scores[winner])winner=c;
  out[r]=winner;
 }
 return 0;
}
'''

def build(folder,out,compiler='g++',target='avx2'):
 import numpy as np
 folder=Path(folder);out=Path(out);out.mkdir(parents=True,exist_ok=False)
 with np.load(folder/'weights.npz') as f:arrays={k:f[k] for k in f.files}
 mlp='w0' in arrays
 matrices=[arrays[f'w{i}'] for i in range(3)] if mlp else [arrays['w'].T]
 biases=[arrays[f'b{i}'] for i in range(3)] if mlp else [arrays['b']]
 d=matrices[0].shape[0];nc=len(arrays['classes']);D=int(arrays['D']);H=matrices[0].shape[1] if mlp else 1
 data=f'constexpr int D={d}, CLASSES={nc}, H={H}, MAXIMUM={D};\n'
 for k,(w,b) in enumerate(zip(matrices,biases)):
  data+='static const double W'+str(k)+'[]={'+','.join(float(v).hex() for v in w.ravel())+'};\n'
  data+='static const double B'+str(k)+'[]={'+','.join(float(v).hex() for v in b.ravel())+'};\n'
 run='linear(input,W0,B0,D,H,h0,true);linear(h0,W1,B1,H,H,h1,true);linear(h1,W2,B2,H,CLASSES,scores,false);' if mlp else 'linear(input,W0,B0,D,CLASSES,scores,false);'
 raw=TEMPLATE.replace('@DATA@',data).replace('@RUN@',run).encode();source=out/'model.cpp';source.write_bytes(raw)
 lib=out/'control.so';cmd=[compiler,'-std=c++17','-O3','-fno-fast-math','-ffp-contract=off','-shared','-fPIC']
 if target=='avx2':cmd+=['-mavx2']
 cmd +=[str(source),'-o',str(lib)]
 t=time.perf_counter();p=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
 receipt={'command':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'seconds':time.perf_counter()-t,
   'source_sha256':hashlib.sha256(raw).hexdigest(),'weights_sha256':hashlib.sha256((folder/'weights.npz').read_bytes()).hexdigest(),
   'features':d,'maximum':D,'classes':arrays['classes'].tolist(),'parameters':sum(w.size+b.size for w,b in zip(matrices,biases))}
 if p.returncode==0:receipt['library_sha256']=hashlib.sha256(lib.read_bytes()).hexdigest()
 (out/'build.json').write_text(json.dumps(receipt,indent=2))
 if p.returncode:raise RuntimeError(p.stderr)
 return lib

class ControlSession:
 def __init__(self,folder):
  folder=Path(folder);meta=json.loads((folder/'build.json').read_text());self.features=meta['features'];self.labels=tuple(meta['classes'])
  self._lib=C.CDLL(str((folder/'control.so').resolve()))
  self._run=self._lib.ctl_run;self._run.argtypes=[C.POINTER(C.c_uint8),C.c_int,C.c_int,C.POINTER(C.c_int)];self._run.restype=C.c_int
 def predict_buffer(self,values):
  try:v=memoryview(values)
  except TypeError as e:raise ValueError('uint8 buffer required') from e
  data=None
  try:
   if v.format!='B' or v.readonly or not v.c_contiguous or v.ndim not in (1,2):raise ValueError('writable contiguous uint8 codes required')
   if v.ndim==2 and v.shape[1]!=self.features:raise ValueError('wrong features')
   if v.nbytes%self.features or v.nbytes>8000000:raise ValueError('wrong input size')
   n=v.nbytes//self.features
   if n>65536:raise ValueError('too many rows')
   data=(C.c_uint8*v.nbytes).from_buffer(v) if v.nbytes else None
   out=(C.c_int*n)()
   if self._run(data,n,self.features,out):raise ValueError('invalid native control input')
   return [self.labels[i] for i in out]
  finally:del data;v.release()

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--target',choices=['avx2','portable'],default='avx2');a=p.parse_args()
 for task in ('letter','pendigits'):
  for arm in ('linear','mlp'):print(build(a.models/(task+'-'+arm),a.out/(task+'-'+arm),target=a.target),flush=True)
