"""Complete fixed-model comparison, no fitting or selected timing-cell retries.

Run initialize, each task exactly once, then summarize. Both elapsed wall time
and process CPU time are retained; validation/identical output checks are timed.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import struct
import sys
import time
from .compiler import VerifiedTotal
from .session import TotalSession
from ..tree_residual.compile import VerifiedResidual
from ..tree_residual.session import ResidualSession
from ..certified_trees.session import TreeSession, VerifiedCompact
from ..certified_trees.export_control import ExportSession

TASKS=('letter','pendigits','satellite','optdigits')
ARMS=('official128','official1210','exported_cpp','full16','residual_adaptive',
      'total_flat','total_interned','exact_flat','exact_interned')
CHUNKS=(1,32,256)
REPEATS=7
CYCLES=10
SEED=2026092967
ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def digest(x):return hashlib.sha256(struct.pack('<'+'i'*len(x),*x)).hexdigest()
def schedule():
    rng=random.Random(SEED);out=[]
    for task in TASKS:
        for rep in range(REPEATS):
            jobs=[(chunk,arm) for chunk in CHUNKS for arm in ARMS];rng.shuffle(jobs)
            out.extend((task,rep,chunk,arm) for chunk,arm in jobs)
    return out

def initialize(a):
    a.out.mkdir(parents=True,exist_ok=False)
    files={}
    for task in TASKS:
        for name in ('model.json','model-16.sct','model.cbm','input.u8','indices.i32'):
            path=a.parent_sdk/'models'/task/name;files[str(path)]=sha(path)
        for name in ('flat.sctt','interned.sctt'):
            path=a.models/task/name;files[str(path)]=sha(path)
        path=a.residual_sdk/'models'/task/'model.scr';files[str(path)]=sha(path)
        path=a.parent_evidence/'replay'/task/'cpp/export.so';files[str(path)]=sha(path)
    for path in (a.library,a.residual_library,a.parent_evidence/'native-register/trees.so',
                 a.parent_evidence/'official/libcatboostmodel-linux-x86_64-1.2.8.so',
                 a.parent_evidence/'official/libcatboostmodel-linux-x86_64-1.2.10.so'):
        files[str(path)]=sha(path)
    source={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*') if p.is_file()}
    # Record previous-engine source dependencies as well as the current compiler.
    for name in ('tree_residual','certified_trees'):
        for p in (ROOT/'experiments'/name).glob('*.cpp'):source[str(p.relative_to(ROOT))]=sha(p)
    write(a.out/'LOCK.json',{'tasks':TASKS,'arms':ARMS,'chunks':CHUNKS,'repeats':REPEATS,'cycles':CYCLES,'seed':SEED,
        'jobs':schedule(),'files':files,'source':source,'platform':platform.platform(),'python':sys.version,
        'cpu':next(s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),
        'cpu_affinity':min(os.sched_getaffinity(0)),
        'gate':{'batch32_geometric_ratio_limit':1.10,'batch32_max_task_ratio_limit':1.25,'baseline':'full16'},
        'scope':'10 whole-dataset jobs/cell; input conversion, inference, fresh labels, output comparison included; loading and proof excluded'})

def task(a):
    lock=json.loads((a.out/'LOCK.json').read_text())
    for path,h in lock['files'].items():
        if sha(path)!=h:raise ValueError('frozen comparison artifact changed')
    for path,h in lock['source'].items():
        if sha(ROOT/path)!=h:
            amendment=json.loads((a.out/'RESUMPTION.json').read_text())
            if path!='experiments/tree_total/benchmark.py' or h!=amendment['original_sha256'] or sha(ROOT/path)!=amendment['resumed_sha256']:
                raise ValueError('source changed after benchmark lock')
    os.sched_setaffinity(0,{lock['cpu_affinity']})
    t=a.task
    prior=[];logpath=a.out/(t+'.jsonl')
    if logpath.exists():
        if not a.resume:raise ValueError('existing timing log requires explicit resume')
        prior=[json.loads(line) for line in logpath.read_text().splitlines()]
        expected_jobs=[tuple(j) for j in lock['jobs'] if j[0]==t]
        if [(r['task'],r['repeat'],r['chunk'],r['arm']) for r in prior]!=expected_jobs[:len(prior)]:
            raise ValueError('saved measurements are not the exact schedule prefix')
    folder=a.parent_sdk/'models'/t;src=(folder/'model.json').read_bytes()
    models={layout:VerifiedTotal(src,(a.models/t/(layout+'.sctt')).read_bytes()) for layout in ('flat','interned')}
    m=models['interned'].info;d=m['features'];D=m['maximum'];c=m['classes']
    proof=VerifiedCompact.from_files(folder/'model.json',folder/'model-16.sct')
    resid=VerifiedResidual.from_files(folder/'model.json',a.residual_sdk/'models'/t/'model.scr')
    data=bytearray((folder/'input.u8').read_bytes());n=len(data)//d
    expected=list(struct.unpack('<'+'i'*n,(folder/'indices.i32').read_bytes()))
    oldlib=a.parent_evidence/'native-register/trees.so'
    with ExitStack() as stack:
        official={v:stack.enter_context(TreeSession(oldlib,official_model=folder/'model.cbm',
                    official_library=a.parent_evidence/'official'/f'libcatboostmodel-linux-x86_64-{v}.so',
                    features=d,maximum=D,classes=c)) for v in ('1.2.8','1.2.10')}
        full=stack.enter_context(TreeSession(oldlib,first=proof,official_model=folder/'model.cbm',
                    official_library=a.parent_evidence/'official/libcatboostmodel-linux-x86_64-1.2.10.so'))
        residual=stack.enter_context(ResidualSession(resid,a.residual_library))
        current={k:stack.enter_context(TotalSession(v,a.library)) for k,v in models.items()}
        exported=ExportSession(a.parent_evidence/'replay'/t/'cpp')
        fn={'official128':official['1.2.8'].predict_buffer,'official1210':official['1.2.10'].predict_buffer,
            'exported_cpp':exported.predict_buffer,'full16':lambda q:full.predict_buffer(q,fallback=True),
            'residual_adaptive':residual.predict_buffer,
            'total_flat':current['flat'].predict_buffer,'total_interned':current['interned'].predict_buffer,
            'exact_flat':lambda q:current['flat'].predict_buffer(q,policy='exact'),
            'exact_interned':lambda q:current['interned'].predict_buffer(q,policy='exact')}
        def invoke(arm,chunk):
            output=[]
            with memoryview(data) as v:
                for start in range(0,n,chunk):output.extend(fn[arm](v[start*d:min(n,start+chunk)*d]))
            return output
        for arm in ARMS:
            if invoke(arm,256)!=expected:raise ValueError('native comparator mismatch '+arm)
        save=lambda path,value: None if a.resume and path.exists() else write(path,value)
        save(a.out/(t+'-models.json'),{'indices_sha256':digest(expected),'rows':n,
             'new':{k:v.info for k,v in current.items()},'old':full.info,
             'new_work':current['interned'].inspect_buffer(data)['work'],'residual_work':residual.inspect_buffer(data)['work']})
        with logpath.open('a' if a.resume else 'x') as log:
            seen=0
            for taskname,rep,chunk,arm in lock['jobs']:
                if taskname!=t:continue
                seen+=1
                if seen<=len(prior):continue
                wall=time.perf_counter_ns();cpu=time.process_time_ns()
                for cycle in range(CYCLES):
                    pred=invoke(arm,chunk)
                    if pred!=expected:raise ValueError('timed output mismatch')
                cpu=time.process_time_ns()-cpu;wall=time.perf_counter_ns()-wall
                row={'task':t,'repeat':rep,'chunk':chunk,'arm':arm,'rows':n,'cycles':CYCLES,
                     'wall_ns':wall,'cpu_ns':cpu,'indices_sha256':digest(pred)}
                log.write(json.dumps(row)+'\n');log.flush()
                if arm==ARMS[-1]:print(t,rep,chunk,flush=True)
    write(a.out/(t+'-complete.json'),{'status':'COMPLETE','timing_rows':len(CHUNKS)*len(ARMS)*REPEATS})

def summarize(a):
    obs=[]
    for t in TASKS:
        if json.loads((a.out/(t+'-complete.json')).read_text())['status']!='COMPLETE':raise ValueError('unfinished task')
        obs.extend(json.loads(s) for s in (a.out/(t+'.jsonl')).read_text().splitlines())
    if [(r['task'],r['repeat'],r['chunk'],r['arm']) for r in obs]!=schedule():raise ValueError('incomplete or reordered grid')
    tables={};ratios=[]
    for t in TASKS:
        n=next(r['rows'] for r in obs if r['task']==t)
        timing={metric:{str(b):{arm:statistics.median(r[metric+'_ns'] for r in obs if (r['task'],r['chunk'],r['arm'])==(t,b,arm))/(n*CYCLES*1000)
                  for arm in ARMS} for b in CHUNKS} for metric in ('wall','cpu')}
        ratio=timing['wall']['32']['total_interned']/timing['wall']['32']['full16'];ratios.append(ratio)
        paired=[]
        for rep in range(REPEATS):
            cells={r['arm']:r['wall_ns'] for r in obs if (r['task'],r['repeat'],r['chunk'])==(t,rep,32)}
            paired.append(cells['total_interned']/cells['full16'])
        tables[t]={'us_per_row':timing,'batch32_total_over_prior_full16':ratio,'batch32_paired_ratios':paired,
                   'batch32_paired_regressions':sum(r>1 for r in paired)}
    g=statistics.geometric_mean(ratios)
    write(a.out/'SUMMARY.json',{'status':'COMPLETE','cells':len(obs),'tasks':tables,'geometric_ratio':g,'max_task_ratio':max(ratios),
            'performance_gate':g<=1.10 and max(ratios)<=1.25,'repeated_predictions':sum(r['rows']*r['cycles'] for r in obs),
            'scope':'recorded same-host complete warm jobs; no training, novel accuracy, cold-start or service-tail claim'})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['initialize','task','summarize']);p.add_argument('--task',choices=TASKS);p.add_argument('--resume',action='store_true')
    for name in ('out','parent-sdk','parent-evidence','residual-sdk','residual-library','models','library'):p.add_argument('--'+name,type=Path)
    a=p.parse_args();globals()[a.operation](a)
