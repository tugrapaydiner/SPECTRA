"""Export the fixed QDA control to native code; input conversion stays in timer."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path

def build(folder,out):
 import numpy as np
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 with np.load(Path(folder)/'weights.npz') as f:values={k:f[k] for k in f.files}
 means=values['means'];nc,d=means.shape
 transformed=values['rotations']*np.power(values['scalings'][:,None,:],-.5)
 bias=np.log(values['priors'])-.5*np.log(values['scalings']).sum(axis=1)
 source='#include <cstdint>\n#include <cmath>\n'
 for name,array in [('MEANS',means),('A',transformed),('B',bias)]:
  source+='static const double '+name+'[]={'+','.join(float(v).hex() for v in array.ravel())+'};\n'
 source+=f'''extern "C" int ctl_run(const uint8_t*q,int rows,int features,int*out){{
 if(rows<0||rows>65536||features!={d}||(rows&&(!q||!out)))return 1;
 for(int r=0;r<rows;++r){{double x[{d}],score[{nc}];for(int j=0;j<{d};++j)x[j]=double(q[r*{d}+j])/255.;
  for(int c=0;c<{nc};++c){{double norm=0;
   for(int j=0;j<{d};++j){{double sum=0;for(int i=0;i<{d};++i)sum+=(x[i]-MEANS[c*{d}+i])*A[c*{d*d}+i*{d}+j];norm+=sum*sum;}}
   score[c]=B[c]-.5*norm;
  }}int winner=0;for(int c=1;c<{nc};++c)if(score[c]>score[winner])winner=c;out[r]=winner;
 }}return 0;
}}'''
 p=out/'model.cpp';p.write_text(source);lib=out/'control.so';cmd=['g++','-std=c++17','-O3','-mavx2','-fno-fast-math','-ffp-contract=off','-shared','-fPIC',str(p),'-o',str(lib)]
 t=time.perf_counter();run=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
 receipt={'features':d,'classes':values['classes'].tolist(),'maximum':255,'parameters':int(means.size+transformed.size+bias.size),
  'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'command':cmd,'seconds':time.perf_counter()-t,'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr,
  'weights_sha256':hashlib.sha256((Path(folder)/'weights.npz').read_bytes()).hexdigest()}
 if run.returncode==0:receipt['library_sha256']=hashlib.sha256(lib.read_bytes()).hexdigest()
 (out/'build.json').write_text(json.dumps(receipt,indent=2))
 if run.returncode:raise RuntimeError(run.stderr)
 return lib
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(build(a.model,a.out))
