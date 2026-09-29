"""Compile frozen source models before empirical certificate/latency observations."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
from . import certificate_oracle as co
from .packing import compile_binary,pack_original,verify_binary,metadata

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(models,out):
    models=Path(models);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    lock=json.loads((models/'FINAL_LOCK.json').read_text())
    for name,h in lock['files'].items():
        if sha(models/name)!=h:raise ValueError('frozen source model changed '+name)
    results={}
    for task in ('letter','pendigits','satellite','optdigits'):
        folder=out/task;folder.mkdir();raw=(models/task/'source.json').read_bytes();info=json.loads((models/task/'fit.json').read_text())
        d=info['features'];D=info['maximum'];t0=time.perf_counter()
        full=pack_original(co.parse_source(raw,D,d));(folder/'full.sct').write_bytes(full)
        for bits in (8,16):
            start=time.perf_counter();blob,record=compile_binary(raw,D,features=d,bits=bits,pairwise=False)
            (folder/f'q{bits}.sct').write_bytes(blob);(folder/f'q{bits}.json').write_bytes(co.canonical(record))
            verified=verify_binary(raw,blob)
            results[f'{task}/q{bits}']={**metadata(blob),**verified,'compiler_and_verification_seconds':time.perf_counter()-start}
            print(task,bits,len(blob),round(results[f'{task}/q{bits}']['compiler_and_verification_seconds'],3),flush=True)
        results[task+'/full']={**metadata(full),**verify_binary(raw,full)}
        print(task,'total',round(time.perf_counter()-t0,3),flush=True)
    receipt={'format':'spectra.tree.compilation-panel.v1','source_model_lock_sha256':sha(models/'FINAL_LOCK.json'),
             'compiler_sources':{n:sha(Path(__file__).parent/n) for n in ('certificate_oracle.py','packing.py','compile_panel.py')},
             'files':{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()},'models':results,
             'scope':'model-only compilation, all bounds reconstructed from source; no empirical inputs or labels used'}
    (out/'COMPILED_LOCK.json').write_text(json.dumps(receipt,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.models,a.out)
