"""Explicit source-bound experimental build; imports never invoke a compiler."""
from pathlib import Path
import argparse,hashlib,json,subprocess,time,platform,sys

def build(out,target='portable',sanitize=False,tiled=True):
    if target not in ('portable','avx2'):raise ValueError('unknown target')
    if sys.platform!='linux' or platform.machine().lower() not in ('x86_64','amd64'):raise ValueError('accepted target is Linux x86-64')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).with_name('runtime.cpp');library=out/'trees.so'
    cmd=['g++','-std=c++17','-O3','-fno-fast-math','-ffp-contract=off','-shared','-fPIC']
    if target=='avx2':cmd+=['-mavx2']
    if tiled:cmd+=['-DST_TILES']
    if sanitize:cmd+=['-O1','-g','-fsanitize=undefined','-fno-sanitize-recover=all']
    cmd +=[str(source),'-o',str(library)];start=time.perf_counter()
    receipt={'command':cmd,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'returncode':None}
    try:
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=120);receipt.update(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
    except (OSError,subprocess.TimeoutExpired) as e:receipt['error']=str(e)
    receipt['seconds']=time.perf_counter()-start
    if receipt['returncode']==0:receipt['library_sha256']=hashlib.sha256(library.read_bytes()).hexdigest()
    (out/'build.json').write_text(json.dumps(receipt,indent=2))
    if receipt['returncode']!=0:raise RuntimeError('build failed; inspect build.json')
    return library
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--target',default='portable');p.add_argument('--ubsan',action='store_true');p.add_argument('--row-major',action='store_true');a=p.parse_args();print(build(a.out,a.target,a.ubsan,not a.row_major))
