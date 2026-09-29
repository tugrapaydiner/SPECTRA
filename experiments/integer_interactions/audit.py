"""Independent stdlib audit of selection, model locks, outputs and timing arithmetic.

No learner/runtime imports, unpickling, fitting or timed-code aggregation. Does
not authenticate clocks or make repeatedly consumed benchmarks independent.
"""
from __future__ import annotations
import argparse,hashlib,json,math,random,statistics,struct,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.learned_metric.audit import arrays,rows,outcome
TASKS=('letter','pendigits','satellite');BASE=('uniform','diagonal','full','whitening')
EXTRA=('local_supervised','local_unsupervised');FAMILIES=BASE+EXTRA
ARMS=FAMILIES+('local_scalar','local_exhaustive','local_direct_exp','diagonal_projected','parent_uniform','parent_nca','linear','mlp')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unique(pairs):
    result={}
    for k,v in pairs:
        if k in result:raise ValueError('duplicate JSON field')
        result[k]=v
    return result
def constant(x):raise ValueError('nonfinite JSON constant')
def read(p):return json.loads(Path(p).read_text(),object_pairs_hook=unique,parse_constant=constant)
def lines(p):return [json.loads(x,object_pairs_hook=unique,parse_constant=constant) for x in Path(p).read_text().splitlines()]
def inside(root,name):
    p=Path(root);relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('escaping path')
    for part in relative.parts:
        p=p/part
        if p.is_symlink():raise ValueError('symlink not accepted')
    if not p.is_file():raise ValueError('missing member '+name)
    return p

def same(a,b):
    if type(a)!=type(b):raise ValueError('different value type')
    if type(b) is dict:
        if set(a)!=set(b):raise ValueError('different field inventory')
        for k in b:same(a[k],b[k])
    elif type(b) is list:
        if len(a)!=len(b):raise ValueError('different sequence length')
        for x,y in zip(a,b):same(x,y)
    elif type(b) is float:
        if not math.isfinite(a) or not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('different numeric value')
    elif a!=b:raise ValueError('different value')

def data_path(root,t,part):return root/('satellite/data/'+part+'.npz' if t=='satellite' else 'datasets/'+t+'-'+part+'.npz')
def prediction_path(root,t,a):return root/('satellite/evaluation/'+a+'-predictions.npz' if t=='satellite' else 'evaluation/'+t+'/'+a+'-predictions.npz')
def hash_predictions(pred):return hashlib.sha256(struct.pack('<'+str(len(pred))+'q',*pred)).hexdigest()
def grid(t):return [(c,g) for c in (1.,10.,100.) for g in ((2.,8.,32.,128.) if t=='satellite' else (.5,2.,8.,32.))]

def audit(root,source):
    root=Path(root);source=Path(source);parent=root/'parent'
    observations={};targets={};test_codes={};test_mask={};selection_cells=0
    for task in TASKS:
        train=arrays(data_path(parent,task,'train'));test=arrays(data_path(parent,task,'test'))
        targets[task]=test['y']['values'];test_codes[task]=rows(test['q']);trainrows=rows(train['q']);keys=set(trainrows)
        test_mask[task]=[r not in keys for r in test_codes[task]]
        for folder,families in (('selection',BASE),('local_selection',EXTRA)):
            selection=root/folder;lock=read(selection/'LOCK.json')
            bound_sources=lock['source'] if folder=='selection' else {'learning.py':lock['source_sha256'],'local_study.py':lock['script_sha256']}
            for name,h in bound_sources.items():
                if sha(inside(root/'source_snapshots'/folder,name))!=h:raise ValueError('selection source mismatch')
            bound_inputs=lock['inputs'] if folder=='selection' else lock['input_sha256']
            for t,h in bound_inputs.items():
                if sha(data_path(parent,t,'train'))!=h:raise ValueError('selection input mismatch')
            rs=[r for r in lines(selection/'rows.jsonl') if r['task']==task]
            expected={(s,a,c,g) for s in (611,977,1543) for a in families for c,g in grid(task)}
            seen=set()
            splits={}
            for seed in (611,977,1543):
                split=arrays(selection/f'{task}-{seed}-split.npz');fit=split['fit']['values'];val=split['validation']['values']
                if len(fit)!=len(set(fit)) or len(val)!=len(set(val)) or set(fit)&set(val):raise ValueError('row split overlap')
                if any(type(i) is not int or not 0<=i<len(trainrows) for i in fit+val):raise ValueError('split index outside training')
                if {trainrows[i] for i in fit}&{trainrows[i] for i in val}:raise ValueError('exact-feature leakage')
                splits[seed]=(fit,val)
            for r in rs:
                key=r['seed'],r['arm'],r['C'],r['gamma']
                if key not in expected or key in seen:raise ValueError('selection inventory')
                seen.add(key);fit,val=splits[r['seed']]
                p=arrays(selection/f"{task}-{r['seed']}-{r['arm']}-C{r['C']:g}-g{r['gamma']:g}.npz")
                y=[train['y']['values'][i] for i in val]
                if p['expected']['values']!=y or len(p['prediction']['values'])!=len(y):raise ValueError('validation target mismatch')
                if r['correct']!=sum(a==b for a,b in zip(p['prediction']['values'],y)) or r['fit_rows']!=len(fit) or r['validation_rows']!=len(val):raise ValueError('validation count mismatch')
                if type(r['supports']) is not int or not 1<=r['supports']<=len(fit):raise ValueError('invalid supports')
            if seen!=expected:raise ValueError('incomplete selection')
            selection_cells+=len(rs);chosen=read(selection/'selected.json')[task]
            for family in families:
                ranks=[]
                for j,(c,g) in enumerate(grid(task)):
                    group=[r for r in rs if r['arm']==family and r['C']==c and r['gamma']==g]
                    ranks.append((sum(r['correct'] for r in group)/sum(r['validation_rows'] for r in group),-sum(r['supports'] for r in group),-j,c,g))
                z=max(ranks);same(chosen[family],{'C':z[3],'gamma':z[4],'validation_accuracy':z[0]})
    final=read(root/'models/FINAL_LOCK.json')
    for name,h in final['files'].items():
        if sha(inside(root/'models',name))!=h:raise ValueError('final model lock mismatch')
    for name,h in final['source'].items():
        if sha(inside(root/'source_snapshots/refit',name))!=h:raise ValueError('final fitting source mismatch')
    if len(final['fits'])!=18:raise ValueError('wrong final fit inventory')
    for r in final['fits']:
        selected=read(root/('local_selection' if r['arm'] in EXTRA else 'selection')/'selected.json')[r['task']][r['arm']]
        same({k:r[k] for k in selected},selected)
    opening=read(root/'evaluation/EVALUATION_OPENING.json')
    if opening['final_lock_sha256']!=sha(root/'models/FINAL_LOCK.json'):raise ValueError('wrong opened model lock')
    for key,h in opening['data'].items():
        t,part=key.rsplit('-',1)
        if sha(data_path(parent,t,part))!=h:raise ValueError('evaluation data changed')
    for name,h in opening['source'].items():
        if sha(inside(source/'experiments/integer_interactions',name))!=h:raise ValueError('evaluated source changed')
    quality=read(root/'evaluation/quality.json')
    for task in TASKS:
        y=targets[task];mask=test_mask[task]
        for family in FAMILIES:
            a=arrays(root/'evaluation'/f'{task}-{family}-predictions.npz');pred=a['prediction']['values']
            if a['expected']['values']!=y or a['nonoverlap']['values']!=mask:raise ValueError('evaluation row identity changed')
            observations[task,family]=pred;observations[task,family+'-direct']=a['direct_prediction']['values']
            report=quality[task][family]
            for k,v in outcome(pred,y).items():same(report[k],v)
            for k,v in outcome([p for p,m in zip(pred,mask) if m],[p for p,m in zip(y,mask) if m]).items():same(report['nonoverlap'][k],v)
        for name,a,b in [('full_vs_diagonal','full','diagonal'),('full_vs_uniform','full','uniform'),('whitening_vs_diagonal','whitening','diagonal'),('local_vs_diagonal','local_supervised','diagonal'),('local_vs_unsupervised','local_supervised','local_unsupervised')]:
            pa,pb=observations[task,a],observations[task,b];win=sum(a==t and b!=t for a,b,t in zip(pa,pb,y));lose=sum(a!=t and b==t for a,b,t in zip(pa,pb,y))
            n=win+lose;k=min(win,lose);pv=min(1.,2*sum(math.comb(n,j) for j in range(k+1))/2**n) if n else 1.
            same({k:quality[task][name][k] for k in ('points','candidate_only_correct','baseline_only_correct','paired_pvalue')},
                 {'points':100*(win-lose)/len(y),'candidate_only_correct':win,'baseline_only_correct':lose,'paired_pvalue':pv})
    protocol=read(root/'benchmark/protocol.json')
    for k,v in [('tasks',list(TASKS)),('arms',list(ARMS)),('chunks',[1,32,256]),('repeats',7),('seed',2026092904)]:same(protocol[k],v)
    for name,h in protocol['source'].items():
        if sha(inside(source,name))!=h:raise ValueError('timed source mismatch')
    for name,h in protocol['files'].items():
        if sha(inside(root,name))!=h:raise ValueError('timed artifact mismatch')
    if protocol['libraries']!={'interaction':sha(root/'build-final/interaction.so'),'parent':sha(root/'parent-final/metric.so')}:raise ValueError('timed library mismatch')
    rng=random.Random(2026092904);expected=[]
    for task in TASKS:
        for repeat in range(7):
            jobs=[(task,repeat,c,a) for c in (1,32,256) for a in ARMS];rng.shuffle(jobs);expected+=jobs
    same(protocol['schedule'],[list(x) for x in expected])
    recorded=lines(root/'benchmark/rows.jsonl')
    if len(recorded)!=len(expected):raise ValueError('missing timed cell')
    for r,key in zip(recorded,expected):
        if set(r)!={'task','repeat','chunk','arm','rows','ns','output_sha256'}:raise ValueError('timing schema differs')
        if tuple(r[k] for k in ('task','repeat','chunk','arm'))!=key or any(type(r[k]) is not int for k in ('repeat','chunk','rows','ns')) or r['ns']<=0:raise ValueError('timing order/type differs')
        t,rep,c,a=key
        if a.startswith('parent_') or a in ('linear','mlp'):
            pred=arrays(prediction_path(parent,t,a[7:] if a.startswith('parent_') else a))['prediction']['values']
        else:
            family={'local_scalar':'local_supervised','local_exhaustive':'local_supervised','local_direct_exp':'local_supervised-direct','diagonal_projected':'diagonal'}.get(a,a)
            pred=observations[t,family]
        if r['rows']!=len(targets[t]) or r['output_sha256']!=hash_predictions(pred):raise ValueError('timed output mismatch')
    result={}
    for task in TASKS:
        costs={str(c):{a:statistics.median(r['ns'] for r in recorded if r['task']==task and r['chunk']==c and r['arm']==a) for a in ARMS} for c in (1,32,256)}
        x=costs['32'];gain=quality[task]['full_vs_diagonal']['points'];ratio=x['full']/x['diagonal'];sg=quality[task]['local_vs_diagonal']['points'];sr=x['local_supervised']/x['diagonal']
        paired={a:[] for a in ('full','local_supervised')}
        for rep in range(7):
            rr={r['arm']:r['ns'] for r in recorded if r['task']==task and r['chunk']==32 and r['repeat']==rep}
            for a in paired:paired[a].append(rr[a]/rr['diagonal'])
        result[task]={'medians_ns':costs,'batch32_us_per_row':{a:v/len(targets[task])/1000 for a,v in x.items()},
            'primary_gain_points':gain,'primary_cost_ratio':ratio,'primary_gate':gain>=.5 and ratio<=1.25,
            'secondary_gain_points':sg,'secondary_cost_ratio':sr,'secondary_gate':sg>=.5 and sr<=1.25,
            'paired_cost_ratios':paired,'paired_regressions':{a:sum(x>1 for x in v) for a,v in paired.items()}}
    summary={'timing_cells':len(recorded),'checked_predictions':sum(r['rows'] for r in recorded),'tasks':result,
        'primary_two_task_gate':sum(v['primary_gate'] for v in result.values())>=2,
        'secondary_two_task_gate':sum(v['secondary_gate'] for v in result.values())>=2,
        'scope':'secondary success cannot replace the primary; exposed historical test partitions'}
    same(read(root/'benchmark/summary.json'),summary)
    fidelity=read(root/'evaluation/fidelity.json')
    if {(r['task'],r['arm']) for r in fidelity}!={(t,a) for t in TASKS for a in FAMILIES} or len(fidelity)!=18:raise ValueError('score-observation inventory')
    total=0
    for r in fidelity:
        c=len(set(targets[r['task']]));count=len(targets[r['task']])*c*(c-1)//2
        if r['pair_scores']!=count or r['original_sha256']!=r['candidate_sha256']:raise ValueError('recorded score parity mismatch')
        total+=count
    return {'status':'PASS','selection_cells':selection_cells,'timing_cells':len(recorded),'model_input_pairs':sum(len(v) for v in targets.values())*6,
        'independently_observed_pair_scores':total,'primary_two_task_gate':summary['primary_two_task_gate'],'secondary_two_task_gate':summary['secondary_two_task_gate'],
        'scope':'recorded identity, complete inventories and derived arithmetic; not independent execution, clock authentication or fresh data'}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','source','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();report=audit(a.root,a.source)
    with a.out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
