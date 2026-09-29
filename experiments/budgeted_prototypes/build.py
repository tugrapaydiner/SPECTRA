"""Explicit Linux x86-64 strict-arithmetic experimental build."""
from pathlib import Path
import hashlib,json,subprocess,time,platform,sys

def build(out,target='portable',sanitize=False):
    if target not in ('portable','avx2'):raise ValueError('bad target')
    if sys.platform!='linux' or platform.machine().lower() not in ('x86_64','amd64'):raise ValueError('tested scope Linux x86-64')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);src=Path(__file__).with_name('runtime.cpp');lib=out/'prototypes.so'
    cmd=['g++','-std=c++17','-O3','-fno-fast-math','-ffp-contract=off','-shared','-fPIC']
    if target=='avx2':cmd+=['-mavx2']
    if sanitize:cmd+=['-O1','-g','-fsanitize=undefined','-fno-sanitize-recover=all']
    cmd += [str(src),'-o',str(lib)];start=time.perf_counter();receipt={'command':cmd,'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest()}
    try:
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=120);receipt.update(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    except (OSError,subprocess.TimeoutExpired) as e:receipt.update(returncode=None,error=str(e))
    receipt['seconds']=time.perf_counter()-start
    if receipt['returncode']==0:receipt['library_sha256']=hashlib.sha256(lib.read_bytes()).hexdigest()
    (out/'build.json').write_text(json.dumps(receipt,indent=2))
    if receipt['returncode']!=0:raise RuntimeError('native build failed; see build.json')
    return lib
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--target',default='portable');p.add_argument('--ubsan',action='store_true');a=p.parse_args();print(build(a.out,a.target,a.ubsan))
