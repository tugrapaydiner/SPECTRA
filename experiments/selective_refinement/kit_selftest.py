"""Replay every frozen policy without numerical Python frameworks. No training."""
from pathlib import Path
import argparse,hashlib,json,platform,sys
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from experiments.selective_refinement.deployment import load_deployment

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(target):
    if sys.platform!='linux' or platform.machine().lower() not in ('x86_64','amd64'):raise ValueError('tested library scope is Linux x86-64')
    if target=='avx2' and 'avx2' not in Path('/proc/cpuinfo').read_text():raise ValueError('AVX2 CPU required')
    m=json.loads((ROOT/'MANIFEST.json').read_text());lib=ROOT/'native'/target/'refinement.so'
    if sha(lib)!=m['libraries'][target]['sha256']:raise ValueError('native identity differs')
    for name,h in m['source_files'].items():
        if sha(ROOT/name)!=h:raise ValueError('source identity differs: '+name)
    counts=0;details={}
    for task,entry in m['tasks'].items():
        inp=ROOT/'inputs'/f'{task}.u8';labels=ROOT/'expected'/f'{task}.json'
        if sha(inp)!=entry['input_sha256'] or sha(labels)!=entry['expected_sha256']:raise ValueError('corpus bytes differ')
        q=bytearray(inp.read_bytes());expect=json.loads(labels.read_text());observations={}
        if len(q)!=entry['rows']*entry['features']:raise ValueError('corpus geometry')
        for policy in ('primary','strict','loose'):
            with load_deployment(ROOT/'models'/task,lib,policy=policy,expected_policy_sha256=entry['policy_sha256']) as w:
                pred,work=w.inspect_buffer(q)
                if pred!=expect[policy]:raise ValueError('prediction mismatch')
                if work!=entry['observed']['work'][policy]:raise ValueError('routing count differs')
                counts+=len(pred);observations[policy]=work
                if w.predict_buffer(bytearray())!=[]:raise ValueError('empty request')
                if policy=='primary':
                    for mode in ('fast','strong','blind'):
                        pred=w.predict_buffer(q,mode=mode)
                        if pred!=expect[mode]:raise ValueError('control prediction mismatch')
                        counts+=len(pred)
                    pred=[];step=31*entry['features']
                    for start in range(0,len(q),step):pred.extend(w.predict_buffer(q[start:start+step]))
                    if pred!=expect['primary']:raise ValueError('chunked result differs')
            try:w.predict_buffer(q)
            except ValueError:pass
            else:raise AssertionError('closed model accepted input')
        details[task]=observations
    forbidden={'numpy','scipy','sklearn','torch','pandas'}&sys.modules.keys()
    if forbidden:raise ValueError('unexpected numerical dependencies: '+str(forbidden))
    return {'status':'PASS','target':target,'prediction_checks':counts,'underlying_rows':sum(v['rows'] for v in m['tasks'].values()),
        'additional_primary_chunk_replays':sum(v['rows'] for v in m['tasks'].values()),'details':details,
        'scope':'frozen same-environment replay; not new accuracy or a risk-under-shift guarantee'}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--target',choices=['portable','avx2'],default='portable');p.add_argument('--out',type=Path);a=p.parse_args();r=run(a.target)
    if a.out:
        with a.out.open('x') as f:json.dump(r,f,indent=2)
    print(json.dumps(r,indent=2))
