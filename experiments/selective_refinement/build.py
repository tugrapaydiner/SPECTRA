"""Explicit experimental native build; defaults to portable CPU instructions."""
from pathlib import Path
import hashlib,json,subprocess,time

def build(out,target='portable',sanitize=False):
    if target not in ('portable','avx2'):raise ValueError('invalid target')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2];src=Path(__file__).with_suffix('.cpp')
    src=src.with_name('runtime.cpp');lib=out/'refinement.so'
    files={'SR_PROTOTYPE':root/'experiments/budgeted_prototypes/runtime.cpp',
           'SR_FINITE':root/'experiments/finite_kernel/runtime.cpp',
           'FK_ADAPTIVE':root/'experiments/adaptive_kernel/runtime.cpp',
           'AK_BASE_RUNTIME':root/'spectra/_native/ovo/runtime.cpp'}
    cmd=['g++','-std=c++17','-O3','-fno-fast-math','-ffp-contract=off','-fPIC','-shared','-DBP_PACKET','-DBP_REGISTERS']
    if target=='avx2':cmd+=['-mavx2','-mpopcnt']
    if sanitize:cmd+=['-O1','-g','-fsanitize=undefined','-fno-sanitize-recover=all']
    cmd += [f'-D{k}="{v}"' for k,v in files.items()]+[str(src),'-o',str(lib)]
    start=time.perf_counter();p=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
    paths=[src,*files.values(),*(root/'spectra/_native/ovo').rglob('*.hpp')]
    rec={'command':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'seconds':time.perf_counter()-start,
         'sources':{str(x.relative_to(root)):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}}
    if not p.returncode:rec['library_sha256']=hashlib.sha256(lib.read_bytes()).hexdigest()
    (out/'build.json').write_text(json.dumps(rec,indent=2))
    if p.returncode:raise RuntimeError(p.stderr)
    return lib
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--target',default='portable');p.add_argument('--ubsan',action='store_true');a=p.parse_args();print(build(a.out,a.target,a.ubsan))
