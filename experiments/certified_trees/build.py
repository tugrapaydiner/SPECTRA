"""Explicit, strict-arithmetic Linux build. No compilation on import."""
from pathlib import Path
import argparse,hashlib,json,platform,subprocess,time

def build(out,target='portable',ubsan=False,layout='original'):
    if layout not in ('original','vector','register'):raise ValueError('unsupported layout')
    if target not in ('portable','avx2') or platform.system()!='Linux':raise ValueError('unsupported target')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).with_name('runtime.cpp');library=out/'trees.so'
    command=['g++','-std=c++17','-O3','-fno-fast-math','-ffp-contract=off','-fPIC','-shared']
    if target=='avx2':command+=['-mavx2']
    if layout in ('vector','register'):command+=['-DTC_VECTOR_ROUTING']
    if layout=='register':command+=['-DTC_FAST_END']
    if ubsan:command+=['-O1','-g','-fsanitize=undefined','-fno-sanitize-recover=all']
    command+=[str(source),'-o',str(library),'-ldl']
    start=time.perf_counter();rec={'command':command,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    try:
        p=subprocess.run(command,capture_output=True,text=True,timeout=120)
        rec.update(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    except (OSError,subprocess.TimeoutExpired) as e:rec.update(returncode=None,error=str(e))
    rec['wall_seconds']=time.perf_counter()-start
    if rec['returncode']==0:rec['library_sha256']=hashlib.sha256(library.read_bytes()).hexdigest()
    (out/'build.json').write_text(json.dumps(rec,indent=2))
    if rec['returncode']!=0:raise RuntimeError('build failed; see build.json')
    return library
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--target',default='portable');p.add_argument('--ubsan',action='store_true');p.add_argument('--layout',default='original');a=p.parse_args();print(build(a.out,a.target,a.ubsan,a.layout))
