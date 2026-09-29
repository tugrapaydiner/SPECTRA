"""One fresh process and one verified row; compare actual loaded model inventories.

Own Linux VmHWM, no numerical Python imports. Filesystem cache may be warm; this
is not worst-case batch memory, cold hardware, or a deployment guarantee.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
from .deployment import load_deployment
from experiments.budgeted_prototypes.session import PrototypeSession
from experiments.budgeted_prototypes.controls import ControlSession

def run(a):
    package=json.loads((a.kit/'MANIFEST.json').read_text());entry=package['tasks'][a.task]
    lib=a.kit/'native'/a.target/'refinement.so'
    if hashlib.sha256(lib.read_bytes()).hexdigest()!=package['libraries'][a.target]['sha256']:raise ValueError('library bytes differ')
    folder=a.kit/'models'/a.task
    with (a.kit/'inputs'/(a.task+'.u8')).open('rb') as f:row=bytearray(f.read(entry['features']))
    expected=json.loads((a.kit/'expected'/(a.task+'.json')).read_text())[a.arm][0]
    begin=time.perf_counter_ns()
    if a.arm=='primary':session=load_deployment(folder,lib,expected_policy_sha256=entry['policy_sha256'])
    elif a.arm=='fast':session=PrototypeSession(folder/'fast.spp',lib)
    else:session=ControlSession(folder/'strong.srt',a.finite,maximum=entry['maximum'])
    with session as m:
        setup=time.perf_counter_ns()-begin
        if m.predict_buffer(row)!=[expected]:raise ValueError('resource probe label mismatch')
        status=Path('/proc/self/status').read_text().splitlines()
        memory={key:int(next(l.split()[1] for l in status if l.startswith(key+':'))) for key in ('VmHWM','VmRSS')}
        if {'numpy','scipy','sklearn','torch','pandas'}&sys.modules.keys():raise ValueError('numerical framework imported')
        return {'task':a.task,'arm':a.arm,'target':a.target,'setup_ns':setup,'memory_kib':memory,'runtime_info':m.info,
                'policy_sha256':entry['policy_sha256'],'prototype_library_sha256':package['libraries'][a.target]['sha256'],
                'finite_library_sha256':hashlib.sha256(a.finite.read_bytes()).hexdigest() if a.arm=='strong' else None,
                'matched':True,'scope':'fresh process, one verified input, warm filesystem possible; native model inventories differ'}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kit',type=Path,required=True);p.add_argument('--task',required=True)
    p.add_argument('--arm',choices=['primary','fast','strong'],required=True);p.add_argument('--target',choices=['portable','avx2'],default='avx2')
    p.add_argument('--finite',type=Path)
    print(json.dumps(run(p.parse_args()),sort_keys=True))
