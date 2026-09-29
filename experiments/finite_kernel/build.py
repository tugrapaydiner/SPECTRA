"""Explicit experiment builds. No download, fitting, implicit installation or release."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build(folder:Path,target='avx2',sanitize=False,gather=False,simple=False):
 folder=folder.resolve();folder.mkdir(parents=True,exist_ok=False)
 flags=['-std=c++17','-O3','-fno-fast-math','-ffp-contract=off','-fPIC','-shared']
 if target=='avx2':flags+=['-mavx2','-mpopcnt']
 if gather:flags+=['-DAK_GATHER']
 if simple:flags+=['-DAK_SIMPLE_BLOCKS']
 if sanitize:flags+=['-O1','-g','-fsanitize=undefined','-fno-sanitize-recover=all']
 source=Path(__file__).with_name('runtime.cpp');base=ROOT/'spectra/_native/ovo/runtime.cpp';prior=ROOT/'experiments/adaptive_kernel/runtime.cpp';lib=folder/'finite.so'
 cmd=['g++',*flags,f'-DAK_BASE_RUNTIME="{base}"',f'-DFK_ADAPTIVE="{prior}"',str(source),'-o',str(lib)]
 start=time.perf_counter();p=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
 sources=[source,prior]+[x for x in base.parent.rglob('*') if x.suffix in ('.hpp','.cpp')]
 rec={'command':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'wall_seconds':time.perf_counter()-start,
      'source_sha256':{str(x.relative_to(ROOT)):sha(x) for x in sources}}
 if p.returncode==0:rec['library_sha256']=sha(lib)
 (folder/'build.json').write_text(json.dumps(rec,indent=2))
 if p.returncode:raise RuntimeError(p.stderr)
 return lib
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--target',choices=['portable','avx2'],default='avx2');p.add_argument('--ubsan',action='store_true');p.add_argument('--gather',action='store_true');p.add_argument('--simple',action='store_true');a=p.parse_args();print(build(a.out,a.target,a.ubsan,a.gather,a.simple))
