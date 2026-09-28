"""Explicit compiler invocation; no training/import-time build."""
from __future__ import annotations
import argparse,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def build(out,target='avx2',ubsan=False):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    flags=['-std=c++17','-O3','-fno-fast-math','-ffp-contract=off','-fPIC','-shared']
    if target=='avx2':flags+=['-mavx2','-mpopcnt']
    elif target!='portable':raise ValueError('unsupported target')
    if ubsan:flags+=['-O1','-g','-fsanitize=undefined','-fno-sanitize-recover=all']
    source=Path(__file__).with_name('runtime.cpp');base=ROOT/'spectra/_native/ovo/runtime.cpp';library=out/'learned_kernel.so'
    command=['g++',*flags,f'-DMK_BASE_RUNTIME="{base}"',str(source),'-o',str(library)]
    start=time.perf_counter();p=subprocess.run(command,capture_output=True,text=True,timeout=120)
    files=[source]+[f for f in base.parent.rglob('*') if f.suffix in ('.cpp','.hpp')]
    receipt={'command':command,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'wall_seconds':time.perf_counter()-start,
             'source_sha256':{f.relative_to(ROOT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    if p.returncode==0:receipt['library_sha256']=hashlib.sha256(library.read_bytes()).hexdigest()
    (out/'build.json').write_text(json.dumps(receipt,indent=2))
    if p.returncode:raise RuntimeError('build failed: '+str(out/'build.json'))
    return library
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--target',default='avx2');p.add_argument('--ubsan',action='store_true');a=p.parse_args();print(build(a.out,a.target,a.ubsan))
