"""Fixed complete-call comparison, raw uint8 codes to new label lists.

No fitting or selection. Direct-exp is a numerical diagnostic and is checked
against its own pre-recorded output, not assumed identical to product-table RBF.
"""
from __future__ import annotations
from contextlib import ExitStack
import argparse,hashlib,json,os,platform,random,statistics,sys,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.integer_interactions.session import InteractionSession
from experiments.learned_metric.session import MetricSession
from experiments.learned_metric.controls import ControlSession
TASKS=('letter','pendigits','satellite')
FAMILIES=('uniform','diagonal','full','whitening','local_supervised','local_unsupervised')
ARMS=FAMILIES+('local_scalar','local_exhaustive','local_direct_exp','diagonal_projected','parent_uniform','parent_nca','linear','mlp')
CHUNKS=(1,32,256);REPEATS=7;SEED=2026092904

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(p):return hashlib.sha256(np.asarray(p,dtype='<i8').tobytes()).hexdigest()
def write(p,o):
    with p.open('x') as f:json.dump(o,f,indent=2,allow_nan=False)
def data_path(root,t,part):return root/('satellite/data/'+part+'.npz' if t=='satellite' else 'datasets/'+t+'-'+part+'.npz')
def parent_model(root,t,a):return root/('satellite/models/'+a if t=='satellite' else 'models/'+t+'-'+a)
def parent_pred(root,t,a):return root/('satellite/evaluation/'+a+'-predictions.npz' if t=='satellite' else 'evaluation/'+t+'/'+a+'-predictions.npz')

def summarize(records,quality):
    result={}
    for task in TASKS:
        n=quality[task]['uniform']['rows']
        med={str(c):{a:statistics.median(r['ns'] for r in records if r['task']==task and r['chunk']==c and r['arm']==a) for a in ARMS} for c in CHUNKS}
        cost=med['32'];gain=quality[task]['full_vs_diagonal']['points'];ratio=cost['full']/cost['diagonal']
        secondary=quality[task]['local_vs_diagonal']['points'];sr=cost['local_supervised']/cost['diagonal']
        paired={a:[] for a in ('full','local_supervised')}
        for repeat in range(REPEATS):
            r={v['arm']:v['ns'] for v in records if v['task']==task and v['chunk']==32 and v['repeat']==repeat}
            for a in paired:paired[a].append(r[a]/r['diagonal'])
        result[task]={'medians_ns':med,'batch32_us_per_row':{a:v/n/1000 for a,v in cost.items()},
            'primary_gain_points':gain,'primary_cost_ratio':ratio,'primary_gate':gain>=.5 and ratio<=1.25,
            'secondary_gain_points':secondary,'secondary_cost_ratio':sr,'secondary_gate':secondary>=.5 and sr<=1.25,
            'paired_cost_ratios':paired,'paired_regressions':{a:sum(x>1 for x in v) for a,v in paired.items()}}
    return {'timing_cells':len(records),'checked_predictions':sum(r['rows'] for r in records),'tasks':result,
            'primary_two_task_gate':sum(v['primary_gate'] for v in result.values())>=2,
            'secondary_two_task_gate':sum(v['secondary_gate'] for v in result.values())>=2,
            'scope':'secondary success cannot replace the primary; exposed historical test partitions'}

def main(a):
    a.out.mkdir(parents=True,exist_ok=False)
    lock=json.loads((a.models/'FINAL_LOCK.json').read_text())
    for f,h in lock['files'].items():
        if sha(a.models/f)!=h:raise ValueError('changed final model')
    files={}
    for root,prefix in ((a.models,'models'),(a.evaluation,'evaluation'),(a.controls,'controls')):
        for p in root.rglob('*'):
            if p.is_file():files[prefix+'/'+p.relative_to(root).as_posix()]=sha(p)
    for task in TASKS:
        files['parent/'+data_path(a.parent,task,'test').relative_to(a.parent).as_posix()]=sha(data_path(a.parent,task,'test'))
        for arm in ('uniform','nca','linear','mlp'):
            p=parent_model(a.parent,task,arm)/('model.sgm' if arm in ('uniform','nca') else 'weights.npz')
            files['parent/'+p.relative_to(a.parent).as_posix()]=sha(p)
            p=parent_pred(a.parent,task,arm);files['parent/'+p.relative_to(a.parent).as_posix()]=sha(p)
    src=[p for p in Path(__file__).parent.iterdir() if p.suffix in ('.cpp','.py','.md')]
    src += [ROOT/'experiments/learned_metric'/x for x in ('runtime.cpp','session.py','controls.py')]
    src += [p for p in (ROOT/'spectra/_native/ovo').rglob('*') if p.suffix in ('.cpp','.hpp')]
    jobs=[(t,r,c,arm) for t in TASKS for r in range(REPEATS) for c in CHUNKS for arm in ARMS]
    # Shuffle within each task/repetition, keeping every cell and all failures.
    rng=random.Random(SEED);schedule=[]
    for task in TASKS:
        for rep in range(REPEATS):
            group=[(task,rep,c,arm) for c in CHUNKS for arm in ARMS];rng.shuffle(group);schedule.extend(group)
    write(a.out/'protocol.json',{'tasks':TASKS,'arms':ARMS,'chunks':CHUNKS,'repeats':REPEATS,'seed':SEED,'schedule':schedule,
        'files':files,'source':{p.relative_to(ROOT).as_posix():sha(p) for p in src},
        'libraries':{'interaction':sha(a.library),'parent':sha(a.parent_library)},
        'cpu':next(l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),
        'affinity':sorted(os.sched_getaffinity(0)),'platform':platform.platform(),'python':sys.version,
        'scope':'all raw-code validation, projection, kernel, voting, interface and fresh labels; loading/table setup/feature extraction excluded'})
    records=[];quality=json.loads((a.evaluation/'quality.json').read_text())
    with threadpool_limits(1),(a.out/'rows.jsonl').open('x') as log:
        for task in TASKS:
            with np.load(data_path(a.parent,task,'test')) as f:q=f['q'].copy()
            expected={}
            for family in FAMILIES:
                with np.load(a.evaluation/f'{task}-{family}-predictions.npz') as f:
                    expected[family]=f['prediction'].tolist()
                    if family=='local_supervised':expected['local_direct_exp']=f['direct_prediction'].tolist()
            expected['local_scalar']=expected['local_exhaustive']=expected['local_supervised']
            expected['diagonal_projected']=expected['diagonal']
            for arm in ('uniform','nca','linear','mlp'):
                with np.load(parent_pred(a.parent,task,arm)) as f:expected['parent_'+arm if arm in ('uniform','nca') else arm]=f['prediction'].tolist()
            with ExitStack() as stack:
                new={family:stack.enter_context(InteractionSession(a.models/(task+'-'+family)/'model.sik',a.library)) for family in FAMILIES}
                old={arm:stack.enter_context(MetricSession(parent_model(a.parent,task,arm)/'model.sgm',a.parent_library)) for arm in ('uniform','nca')}
                ctl={arm:ControlSession(a.controls/(task+'-'+arm)) for arm in ('linear','mlp')}
                def invoke(arm,chunk):
                    result=[]
                    for start in range(0,len(q),chunk):
                        x=q[start:start+chunk]
                        if arm in new:y=new[arm].predict_buffer(x)
                        elif arm in ('linear','mlp'):y=ctl[arm].predict_buffer(x)
                        elif arm.startswith('parent_'):y=old[arm[7:]].predict_buffer(x)
                        elif arm=='diagonal_projected':y=new['diagonal'].predict_buffer(x,mode='projected')
                        else:y=new['local_supervised'].predict_buffer(x,mode={'local_scalar':'scalar_integer','local_exhaustive':'exhaustive','local_direct_exp':'direct_exp'}[arm])
                        result.extend(y)
                    return result
                for arm in ARMS:
                    if invoke(arm,256)!=expected[arm]:raise ValueError('warmup mismatch '+task+' '+arm)
                for _,repeat,chunk,arm in (j for j in schedule if j[0]==task):
                    start=time.perf_counter_ns();pred=invoke(arm,chunk);elapsed=time.perf_counter_ns()-start
                    if pred!=expected[arm]:raise ValueError('timed mismatch '+task+' '+arm)
                    record={'task':task,'repeat':repeat,'chunk':chunk,'arm':arm,'rows':len(q),'ns':elapsed,'output_sha256':digest(pred)}
                    log.write(json.dumps(record)+'\n');log.flush();records.append(record)
                print(task,'complete',flush=True)
    result=summarize(records,quality);write(a.out/'summary.json',result)
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('parent','models','evaluation','controls','library','parent-library','out'):p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
