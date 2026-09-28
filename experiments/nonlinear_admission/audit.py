"""Recompute selection, held-out metrics, hashes and the complete timing grid.

This separate checker never loads pickles or calls any model. It verifies recorded
predictions/identities and arithmetic, not authenticated clocks or external adoption.
"""
from __future__ import annotations
import argparse,hashlib,json,random,statistics
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    def unique(pairs):
        out={}
        for k,v in pairs:
            if k in out:raise ValueError('duplicate JSON key')
            out[k]=v
        return out
    return json.loads(Path(p).read_text(),object_pairs_hook=unique,parse_constant=lambda t:(_ for _ in ()).throw(ValueError(t)))
def acc(y,p):
    if p.shape!=y.shape or not np.issubdtype(p.dtype,np.integer):raise ValueError('prediction geometry/type')
    return float(np.mean(y==p))
def macro(y,p):
    values=[]
    for c in np.union1d(y,p):
        tp=np.sum((y==c)&(p==c));fp=np.sum((y!=c)&(p==c));fn=np.sum((y==c)&(p!=c))
        values.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return float(np.mean(values))
def digest(p):return hashlib.sha256(np.asarray(p,dtype='<i8').tobytes()).hexdigest()
def almost(a,b):
    if not np.isfinite(a) or abs(a-b)>1e-12:raise ValueError('derived metric differs')

def audit(root,builds,run,source):
    freeze=read(root/'MODEL_FREEZE.json');selected=read(root/'SELECTED.json');data=read(root/'DATA.json')
    if sha(root/'SELECTED.json')!=freeze['selected_sha256'] or sha(root/'DATA.json')!=freeze['data_sha256']:raise ValueError('freeze chain changed')
    for name,wanted in freeze['files'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=wanted:raise ValueError('frozen model changed')
    for name,record in data['acquisition']['files'].items():
        original=root/'original'/name
        if original.stat().st_size!=record['bytes'] or sha(original)!=record['sha256']:raise ValueError('original archive differs')
    quality=read(root/'evaluation/QUALITY.json')
    if quality['freeze_sha256']!=sha(root/'MODEL_FREEZE.json'):raise ValueError('quality not bound to models')
    events=[json.loads(s) for s in (root/'events.jsonl').read_text().splitlines()]
    kinds=[e['event'] for e in events]
    if kinds.count('test_opened')!=1 or kinds.index('selection_frozen')>kinds.index('models_frozen') or kinds.index('models_frozen')>kinds.index('test_opened'):raise ValueError('freeze chronology invalid')
    frozen_predictions={};tasks=('isolet','gas')
    for task in tasks:
        for name,entry in data['splits'][task].items():
            if sha(root/'data'/task/(name+'.npz'))!=entry['sha256']:raise ValueError('data identity differs')
        with np.load(root/'data'/task/'validation.npz',allow_pickle=False) as z:val=z['y']
        records=selected['records'][task]
        expected_cfg=[{'family':'linear','C':c} for c in (.01,.1,1.,10.)]
        expected_cfg += [{'family':'svm','C':c,'gamma_multiplier':g} for c in (1.,10.) for g in (.25,1.,4.)]
        expected_cfg += [{'family':'mlp','width':w,'seed':z} for w in (64,256) for z in (101,202,303)]
        expected_cfg += [{'family':'tree','leaves':n} for n in (15,31)]
        if [r['job']['config'] for r in records]!=expected_cfg:raise ValueError('development model inventory differs')
        for r in records:
            if r['status'] not in ('COMPLETE','TIMEOUT','ERROR'):raise ValueError('invalid fit disposition')
            if r['status']=='TIMEOUT' and r['process_wall_seconds']<120:raise ValueError('invalid censored fit')
        for i,r in enumerate(records):
            if r['status']=='COMPLETE':
                p=np.load(root/'development'/task/f'{i:02d}'/'validation_predictions.npy',allow_pickle=False)
                almost(acc(val,p),r['fit']['validation_accuracy']);almost(macro(val,p),r['fit']['validation_macro_f1'])
        for family in ('linear','svm','tree'):
            completed=[r for r in records if r['status']=='COMPLETE' and r['job']['config']['family']==family]
            winner=max(completed,key=lambda r:r['fit']['validation_accuracy'])
            if winner['job']['config']!=selected['selected'][task]['configs'][family]:raise ValueError('selection differs')
        widths={}
        for width in (64,256):
            rr=[r for r in records if r['job']['config'].get('width')==width]
            widths[width]=float(np.mean([r['fit']['validation_accuracy'] for r in rr])) if len(rr)==3 and all(r['status']=='COMPLETE' for r in rr) else -1.
        if max(widths,key=widths.get)!=selected['selected'][task]['configs']['mlp']['width']:raise ValueError('seed cherry-picking')
        cfg=selected['selected'][task]['configs']
        val_scores={f:max(r['fit']['validation_accuracy'] for r in records if r['status']=='COMPLETE' and r['job']['config']==cfg[f]) for f in ('svm','linear','tree')}
        val_mlp=widths[cfg['mlp']['width']]
        val_gate=val_scores['svm']-max(val_scores['linear'],val_mlp)>=.01
        if val_gate!=selected['selected'][task]['svm_admitted_validation'] or val_gate!=quality['tasks'][task]['svm_admitted_validation']:raise ValueError('validation gate differs')
        for name,record in freeze['records'][task].items():
            family=record['job']['config']['family'];expected_config=dict(cfg[family])
            if family=='mlp':expected_config['seed']=int(name.split('-')[1])
            if record['job']['config']!=expected_config or record['status']!='COMPLETE':raise ValueError('refit selection differs')
        with np.load(root/'data'/task/'test.npz',allow_pickle=False) as z:y=z['y'];groups=z['groups']
        if len(y)!=({'isolet':1559,'gas':4070}[task]):raise ValueError('test size differs')
        if task=='gas' and (list(np.unique(groups,return_counts=True)[0])!=[9,10] or list(np.unique(groups,return_counts=True)[1])!=[470,3600]):raise ValueError('temporal test boundaries differ')
        frozen_predictions[task]={}
        for name,q in quality['tasks'][task]['models'].items():
            p=np.load(root/'evaluation'/(task+'-'+name+'.npy'),allow_pickle=False)
            almost(acc(y,p),q['accuracy']);almost(macro(y,p),q['macro_f1'])
            if int(np.sum(y==p))!=q['correct']:raise ValueError('correct count differs')
            for g in np.unique(groups):almost(acc(y[groups==g],p[groups==g]),q['groups'][str(g)]['accuracy'])
            frozen_predictions[task][name]=p
        mean=float(np.mean([quality['tasks'][task]['models']['mlp-'+str(s)]['accuracy'] for s in (101,202,303)]))
        almost(mean,quality['tasks'][task]['mlp_mean_accuracy'])
        margin=quality['tasks'][task]['models']['svm']['accuracy']-max(mean,quality['tasks'][task]['models']['linear']['accuracy'])
        almost(margin,quality['tasks'][task]['svm_margin_over_best_linear_mlp'])
        if (margin>=.01)!=quality['tasks'][task]['svm_admitted_test']:raise ValueError('test gate changed')
        if (quality['tasks'][task]['models']['svm']['accuracy']>=quality['tasks'][task]['models']['tree']['accuracy'])!=quality['tasks'][task]['svm_not_worse_than_tree']:raise ValueError('tree comparison differs')
    b=read(builds/'BUILD.json')
    for name,wanted in b['model_files'].items():
        if sha(builds/name)!=wanted:raise ValueError('model export changed')
    protocol=read(run/'PROTOCOL.json');inventory=read(run/'INVENTORY.json');summary=read(run/'SUMMARY.json')
    if sha(root/'MODEL_FREEZE.json')!=protocol['model_freeze_sha256'] or sha(builds/'BUILD.json')!=protocol['build_sha256']:raise ValueError('timing bindings differ')
    if protocol['seed']!=20260928 or protocol['repeats']!=7 or protocol['jobs']!=[['prepared',1],['prepared',32],['prepared',256],['with_scaling',32]]:raise ValueError('fixed timing protocol differs')
    for name,wanted in protocol['sources'].items():
        if sha(source/name)!=wanted:raise ValueError('measured source changed')
    generated=read(builds/'GENERATED_SELECTION.json')
    if sha(builds/'GENERATED_SELECTION.json')!=protocol['generated_selection_sha256']:raise ValueError('generated selection changed')
    if sha(source/'experiments/nonlinear_admission/COMPARATOR_AMENDMENT.md')!=protocol['amendment_sha256']:raise ValueError('amendment differs')
    for task,entry in generated['models'].items():
        if 'binary' in entry:
            old=Path(entry['binary']);parent='generated_guarded' if 'generated_guarded' in old.parts else 'generated'
            if sha(builds/parent/task/old.name)!=entry['sha256']:raise ValueError('generated binary changed')
    for key,path in b['libraries'].items():
        # Relocate by artifact basename, not an arbitrary saved absolute path.
        p=builds/'spectra'/Path(path).name if key=='spectra' else builds/Path(path).name
        if sha(p)!=b['sha256'][key]:raise ValueError('measured binary changed')
    rows=[json.loads(l) for l in (run/'timings.jsonl').read_text().splitlines()]
    rng=random.Random(20260928);expected=[];buckets={}
    for task in tasks:
        expected_arms=['spectra','spectra_exhaustive','libsvm']
        if 'binary' in generated['models'][task]:expected_arms.append('generated_c')
        for name in ('linear','mlp-101','mlp-202','mlp-303'):expected_arms.extend([name,name+'_sklearn'])
        expected_arms.append('tree_sklearn')
        if inventory[task]['arms']!=expected_arms:raise ValueError('omitted or reordered model arm')
        arms=inventory[task]['arms'];jobs=[(scope,batch,arm) for scope,batch in [('prepared',1),('prepared',32),('prepared',256),('with_scaling',32)] for arm in arms]
        for repeat in range(7):
            order=list(jobs);rng.shuffle(order)
            for scope,batch,arm in order:expected.append((task,repeat,scope,batch,arm))
    if len(rows)!=len(expected):raise ValueError('missing timed attempt')
    for r,e in zip(rows,expected):
        if any(type(r[k]) is not int for k in ('repeat','batch','rows','ns')):raise ValueError('invalid timing field type')
        if tuple(r[k] for k in ('task','repeat','scope','batch','arm'))!=e or type(r['ns']) is not int or r['ns']<=0:raise ValueError('timing order/number')
        t,repeat,scope,batch,arm=e;name='svm' if arm in ('spectra','spectra_exhaustive','libsvm','generated_c') else 'tree' if arm=='tree_sklearn' else arm
        if name.endswith('_sklearn'):name=name[:-8]
        p=frozen_predictions[t][name]
        if r['rows']!=len(p) or r['predictions_sha256']!=digest(p):raise ValueError('timed output changed')
        buckets.setdefault((t,scope,batch,arm),[]).append(r['ns'])
    for task in tasks:
        for scope,batch in [('prepared',1),('prepared',32),('prepared',256),('with_scaling',32)]:
            q=summary['tasks'][task][scope+'-'+str(batch)]
            values={arm:statistics.median(buckets[task,scope,batch,arm])/inventory[task]['rows']/1000 for arm in inventory[task]['arms']}
            for arm,v in values.items():almost(v,q['us_per_row'][arm])
            almost(values['libsvm']/values['spectra'],q['libsvm_over_spectra'])
            almost(values['spectra_exhaustive']/values['spectra'],q['exhaustive_over_spectra'])
    if len(rows)!=summary['timing_records'] or sum(r['rows'] for r in rows)!=summary['repeated_predictions']:raise ValueError('summary counts differ')
    return {'status':'PASS','tasks':2,'timing_records':len(rows),'unique_task_test_rows':sum(len(frozen_predictions[t]['svm']) for t in tasks),
      'models':sum(len(p) for p in frozen_predictions.values()),'scope':'recorded selection, metrics, full grid and hashes; not independently authenticated performance'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','builds','run','source','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();result=audit(a.root,a.builds,a.run,a.source)
    with a.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
