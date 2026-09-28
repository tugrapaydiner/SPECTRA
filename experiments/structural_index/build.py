"""Explicit CPython extension build; no installation/import side effects."""
import hashlib,json,subprocess,sys,sysconfig,time
from pathlib import Path

def build(folder, flags=(), source_name="fingerprint.cpp"):
    if sys.implementation.name!='cpython' or sys.platform!='linux' or sysconfig.get_config_var('Py_GIL_DISABLED'):
        raise ValueError('accepted build scope is GIL-enabled CPython on Linux')
    folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).with_name(source_name).resolve()
    output=folder/('_spectra_structural'+sysconfig.get_config_var('EXT_SUFFIX'))
    command=['g++','-std=c++17','-O3','-fno-fast-math','-ffp-contract=off','-shared','-fPIC',
        '-I'+sysconfig.get_path('include'),*flags,str(source),'-o',str(output)]
    start=time.perf_counter();p=subprocess.run(command,capture_output=True,text=True,timeout=90)
    result=dict(command=command,seconds=time.perf_counter()-start,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    if p.returncode==0:result['library_sha256']=hashlib.sha256(output.read_bytes()).hexdigest();result['bytes']=output.stat().st_size
    (folder/'build.json').write_text(json.dumps(result,indent=2))
    if p.returncode:raise RuntimeError(p.stderr)
    return output

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--source',choices=['native.cpp','fingerprint.cpp'],default='fingerprint.cpp');a=p.parse_args();print(build(a.folder,source_name=a.source))
