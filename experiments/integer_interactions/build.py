"""Explicit strict-arithmetic build with source-bound success/failure receipts."""
from pathlib import Path
import argparse,hashlib,json,subprocess,time
ROOT=Path(__file__).resolve().parents[2]
def build(out,target='portable',sanitize=False):
    if target not in ('portable','avx2'):raise ValueError('target not supported')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);base=ROOT/'spectra/_native/ovo/runtime.cpp';src=Path(__file__).with_name('runtime.cpp');lib=out/'interaction.so'
    cmd=['g++','-std=c++17','-O3','-fno-fast-math','-ffp-contract=off','-fPIC','-shared']
    if target=='avx2':cmd+=['-mavx2']
    if sanitize:cmd+=['-O1','-g','-fsanitize=undefined','-fno-sanitize-recover=all']
    cmd+=[f'-DII_BASE_RUNTIME="{base}"',str(src),'-o',str(lib)]
    start=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
    sources=[src]+[f for f in base.parent.rglob('*') if f.suffix in ('.cpp','.hpp')]
    record={'command':cmd,'exitcode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'wall_seconds':time.perf_counter()-start,
            'sources':{f.relative_to(ROOT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}}
    if not r.returncode:record['library_sha256']=hashlib.sha256(lib.read_bytes()).hexdigest()
    (out/'build.json').write_text(json.dumps(record,indent=2))
    if r.returncode:raise RuntimeError(r.stderr)
    return lib
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--target',default='portable');p.add_argument('--ubsan',action='store_true');a=p.parse_args();print(build(a.out,a.target,a.ubsan))
