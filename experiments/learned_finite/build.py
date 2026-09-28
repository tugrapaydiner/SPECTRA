"""Explicit experiment-only build; production default and ABI unchanged."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def build(out,target='popcnt',sanitize=False):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    base=ROOT/'spectra/_native/ovo/runtime.cpp';source=Path(__file__).with_name('runtime.cpp');lib=out/'learned_finite.so'
    flags=['-std=c++17','-O3','-fno-fast-math','-ffp-contract=off','-fPIC','-shared']
    if target=='popcnt':flags+=['-mpopcnt']
    elif target!='portable':raise ValueError('unknown build target')
    if sanitize:flags+=['-O1','-g','-fsanitize=undefined','-fno-sanitize-recover=all']
    command=['g++',*flags,f'-DLF_BASE="{base}"',str(source),'-o',str(lib)]
    start=time.perf_counter();p=subprocess.run(command,capture_output=True,text=True,timeout=120)
    paths=[source]+[x for x in base.parent.rglob('*') if x.suffix in ('.hpp','.cpp')]
    receipt={'command':command,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'wall_seconds':time.perf_counter()-start,
             'sources':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}}
    if p.returncode==0:receipt['library_sha256']=hashlib.sha256(lib.read_bytes()).hexdigest()
    (out/'build.json').write_text(json.dumps(receipt,indent=2))
    if p.returncode:raise RuntimeError(p.stderr)
    return lib
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--target',default='popcnt');p.add_argument('--ubsan',action='store_true');a=p.parse_args();print(build(a.out,a.target,a.ubsan))
