"""Explicit, source-bound experimental build; no compile on import."""
from pathlib import Path
import argparse,hashlib,json,subprocess,time
ROOT=Path(__file__).resolve().parents[2]
def build(out,target='avx2',sanitize=False):
 if target not in ('portable','avx2'):raise ValueError('unknown target')
 out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
 base=ROOT/'spectra/_native/ovo/runtime.cpp';source=Path(__file__).with_name('runtime.cpp');library=out/'metric.so'
 flags=['-std=c++17','-O3','-fno-fast-math','-ffp-contract=off','-fPIC','-shared']
 if target=='avx2':flags+=['-mavx2']
 if sanitize:flags+=['-O1','-g','-fsanitize=undefined','-fno-sanitize-recover=all']
 command=['g++',*flags,f'-DLM_BASE_RUNTIME="{base}"',str(source),'-o',str(library)]
 t=time.perf_counter();p=subprocess.run(command,capture_output=True,text=True,timeout=120)
 sources=[source]+sorted(f for f in base.parent.rglob('*') if f.suffix in ('.cpp','.hpp'))
 report={'command':command,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'wall_seconds':time.perf_counter()-t,
  'source_sha256':{f.relative_to(ROOT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}}
 if p.returncode==0:report['library_sha256']=hashlib.sha256(library.read_bytes()).hexdigest()
 (out/'build.json').write_text(json.dumps(report,indent=2))
 if p.returncode:raise RuntimeError(p.stderr)
 return library
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--target',choices=['portable','avx2'],default='avx2');p.add_argument('--ubsan',action='store_true');a=p.parse_args();print(build(a.out,a.target,a.ubsan))
