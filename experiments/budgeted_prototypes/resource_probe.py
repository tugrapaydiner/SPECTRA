"""One fresh process, frozen model, own VmHWM; no numerical Python imports."""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def run(a):
    from experiments.budgeted_prototypes.session import PrototypeSession
    from experiments.budgeted_prototypes.controls import ControlSession
    from experiments.budgeted_prototypes.strong_controls import BlasSession
    fit=json.loads((a.folder/'fit.json').read_text());d=fit['features'];arm=fit['arm']
    with a.input.open('rb') as f:row=bytearray(f.read(d))
    start=time.perf_counter_ns()
    if arm in ('fixed','centers','local'):session=PrototypeSession(a.folder/'model.spp',a.library)
    elif arm=='svc':session=ControlSession(a.folder/'model.srt',a.library,maximum=fit['maximum'])
    elif arm=='mlp':session=BlasSession(a.folder/'model.snn',a.library)
    else:session=ControlSession(a.folder/'model.snn',a.library)
    with session as w:
        setup=time.perf_counter_ns()-start
        expected=json.loads(a.expected.read_text())
        if w.predict_buffer(row)!=expected[:1]:raise ValueError('memory probe label mismatch')
        lines=Path('/proc/self/status').read_text().splitlines()
        memory={k:int(next(s.split()[1] for s in lines if s.startswith(k+':'))) for k in ('VmHWM','VmRSS')}
        forbidden={'numpy','scipy','torch','sklearn','pandas'}&sys.modules.keys()
        if forbidden:raise ValueError('numerical Python framework imported')
        return {'task':fit['task'],'arm':arm,'setup_ns':setup,'memory_kib':memory,'runtime_info':w.info,
                'model_sha256':w.sha256,'library_sha256':hashlib.sha256(a.library.read_bytes()).hexdigest(),
                'matched':True,'python':sys.version,'scope':'actual fresh-process footprint and one-row warm prediction; filesystem cache may be warm; not a memory or latency guarantee'}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('folder','input','library','expected'):p.add_argument('--'+k,type=Path,required=True)
    print(json.dumps(run(p.parse_args()),sort_keys=True))
