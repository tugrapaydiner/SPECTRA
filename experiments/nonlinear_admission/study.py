"""Predeclared two-task admission study. No test predictions before model freeze.

Run prepare, develop, refit, evaluate in that order. Training runs in isolated,
bounded children. All model pickles are locally produced trusted artifacts only.
"""
from __future__ import annotations
import argparse, hashlib, io, json, os, pickle, platform, resource, signal, subprocess, sys, time, warnings, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

TASKS = ('isolet', 'gas')
SEEDS = (101, 202, 303)
PROTOCOL_COMMIT = 'd98cc128de88a8765203560ee87ce9b39f34429f'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f: json.dump(value, f, indent=2, allow_nan=False)
def now(): return datetime.now(timezone.utc).isoformat()
def event(root, kind, **values):
    with (root/'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps({'utc':now(),'event':kind,**values},allow_nan=False)+'\n')
def write_data(folder, name, x, y, groups):
    np.savez_compressed(folder/(name+'.npz'), x=np.ascontiguousarray(x,dtype=np.float64),
                        y=np.asarray(y,dtype=np.int64), groups=np.asarray(groups,dtype=np.int64))
def load(path):
    with np.load(path, allow_pickle=False) as z: return z['x'], z['y'], z['groups']
def sparse_rows(raw):
    x=[];y=[]
    for line in raw.decode('ascii').splitlines():
        tokens=line.split()
        if not tokens: continue
        label=int(tokens[0]); row=np.zeros(128,dtype=np.float64); seen=set()
        for token in tokens[1:]:
            k,v=token.split(':');k=int(k)-1
            if not 0<=k<128 or k in seen: raise ValueError('invalid or duplicate feature')
            seen.add(k);row[k]=float(v)
        if not np.isfinite(row).all() or label not in range(1,7): raise ValueError('invalid gas row')
        x.append(row);y.append(label)
    return np.asarray(x), np.asarray(y,dtype=np.int64)

def prepare(a):
    from sklearn.model_selection import train_test_split
    a.out.mkdir(parents=True,exist_ok=False)
    archive=zipfile.ZipFile(a.archive)
    acquisition=json.loads(archive.read('ACQUISITION.json'))
    origin=a.out/'original';origin.mkdir(); ids={}
    for filename in ('isolet.zip','gas.zip'):
        raw=archive.read(filename);expected=acquisition['files'][filename]
        if len(raw)!=expected['bytes'] or hashlib.sha256(raw).hexdigest()!=expected['sha256']:
            raise ValueError('acquisition identity differs')
        (origin/filename).write_bytes(raw);ids[filename]=expected
    for task in TASKS: (a.out/'data'/task).mkdir(parents=True)
    z=zipfile.ZipFile(origin/'isolet.zip')
    def iso(name):
        raw=z.read(name)
        p=subprocess.run(['gzip','-cd'],input=raw,capture_output=True,timeout=60,check=True)
        matrix=np.loadtxt(io.BytesIO(p.stdout),delimiter=',',dtype=np.float64)
        if matrix.shape[1]!=618 or not np.isfinite(matrix).all():raise ValueError('invalid ISOLET geometry')
        y=matrix[:,-1].astype(np.int64)
        if not np.array_equal(y,matrix[:,-1]) or not np.isin(y,np.arange(1,27)).all():raise ValueError('invalid ISOLET labels')
        return matrix[:,:-1], y
    x,y=iso('isolet1+2+3+4.data.Z'); tx,ty=iso('isolet5.data.Z')
    if (len(x),len(tx))!=(6238,1559): raise ValueError('unexpected ISOLET row counts')
    fit,val=train_test_split(np.arange(len(x)),test_size=.2,stratify=y,random_state=20260928)
    folder=a.out/'data/isolet';np.savez_compressed(folder/'development_indices.npz',fit=fit,validation=val)
    write_data(folder,'fit',x[fit],y[fit],np.zeros(len(fit),int))
    write_data(folder,'validation',x[val],y[val],np.zeros(len(val),int))
    write_data(folder,'refit',x,y,np.zeros(len(y),int));write_data(folder,'test',tx,ty,np.zeros(len(ty),int))
    for name in z.namelist():
        if name.endswith(('.names','.info')):(origin/name).write_bytes(z.read(name))
    z=zipfile.ZipFile(origin/'gas.zip');parts=[]
    for batch in range(1,11):
        names=[n for n in z.namelist() if n.endswith('/batch'+str(batch)+'.dat') or n=='batch'+str(batch)+'.dat']
        if len(names)!=1:raise ValueError('missing gas batch')
        bx,by=sparse_rows(z.read(names[0]));parts.append((bx,by,np.full(len(by),batch)))
    if sum(len(y) for _,y,_ in parts)!=13910:raise ValueError('unexpected gas total')
    folder=a.out/'data/gas'
    for name,selected in [('fit',parts[:6]),('validation',parts[6:8]),('refit',parts[:8]),('test',parts[8:])]:
        write_data(folder,name,*(np.concatenate([p[i] for p in selected]) for i in range(3)))
    details={t:{p.stem:{'rows':len(load(p)[1]),'features':load(p)[0].shape[1],'sha256':sha(p)}
               for p in (a.out/'data'/t).glob('*.npz') if p.stem!='development_indices'} for t in TASKS}
    save(a.out/'DATA.json',{'protocol_commit':PROTOCOL_COMMIT,'acquisition':acquisition,'archive_sha256':sha(a.archive),
         'utc':now(),'splits':details,'script_sha256':sha(__file__),'test_predictions':False})
    event(a.out,'data_prepared',details=details)
    print(json.dumps(details,indent=2))

def configs():
    jobs=[]
    for c in (.01,.1,1.,10.):jobs.append({'family':'linear','C':c})
    for c in (1.,10.):
        for g in (.25,1.,4.):jobs.append({'family':'svm','C':c,'gamma_multiplier':g})
    for width in (64,256):
        for seed in SEEDS:jobs.append({'family':'mlp','width':width,'seed':seed})
    for leaves in (15,31):jobs.append({'family':'tree','leaves':leaves})
    return jobs

def make_model(config,d):
    from sklearn.svm import LinearSVC,SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    f=config['family']
    if f=='linear':return LinearSVC(C=config['C'],dual='auto',max_iter=10000,random_state=101)
    if f=='svm':return SVC(C=config['C'],gamma=config['gamma_multiplier']/d,cache_size=128,probability=False,break_ties=False)
    if f=='mlp':return MLPClassifier(hidden_layer_sizes=(config['width'],),activation='relu',solver='adam',
        batch_size=128,learning_rate_init=.001,alpha=.0001,max_iter=300,early_stopping=True,
        validation_fraction=.1,n_iter_no_change=20,random_state=config['seed'])
    if f=='tree':return HistGradientBoostingClassifier(max_leaf_nodes=config['leaves'],max_iter=200,
        learning_rate=.1,l2_regularization=.1,early_stopping=False,random_state=101)
    raise ValueError('invalid family')

def fit(a):
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score,f1_score
    from threadpoolctl import threadpool_limits
    cfg=json.loads(a.job.read_text()); x,y,_=load(a.root/'data'/cfg['task']/('refit.npz' if cfg['stage']=='refit' else 'fit.npz'))
    with threadpool_limits(1):
        cpu=time.process_time(); wall=time.perf_counter();scaler=StandardScaler().fit(x);x=scaler.transform(x)
        model=make_model(cfg['config'],x.shape[1])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(x,y)
        record={'job':cfg,'fit_wall_seconds':time.perf_counter()-wall,'fit_cpu_seconds':time.process_time()-cpu,
           'warnings':[str(w.message) for w in caught],'iterations':int(np.max(model.n_iter_)) if hasattr(model,'n_iter_') else None,
           'utc':now(),'script_sha256':sha(__file__),'training_rows':len(y)}
        if cfg['stage']=='development':
            vx,vy,_=load(a.root/'data'/cfg['task']/'validation.npz'); pred=model.predict(scaler.transform(vx))
            record.update(validation_accuracy=float(accuracy_score(vy,pred)),validation_macro_f1=float(f1_score(vy,pred,average='macro',zero_division=0)))
            np.save(a.job.parent/'validation_predictions.npy',pred,allow_pickle=False)
        with (a.job.parent/'model.pkl').open('xb') as f:pickle.dump({'scaler':scaler,'model':model},f,protocol=5)
        record['pickle_sha256']=sha(a.job.parent/'model.pkl');save(a.job.parent/'fit.json',record)
        print(json.dumps(record),flush=True)

def restrict():
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})

def child(root,folder,task,config,stage):
    folder.mkdir(parents=True,exist_ok=False)
    job={'task':task,'config':config,'stage':stage};save(folder/'job.json',job)
    args=[sys.executable,str(Path(__file__).resolve()),'fit','--root',str(root),'--job',str(folder/'job.json')]
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    start=time.perf_counter();p=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,preexec_fn=restrict,start_new_session=True)
    status='COMPLETE'
    try:out,err=p.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        status='TIMEOUT';os.killpg(p.pid,signal.SIGKILL);out,err=p.communicate()
    if p.returncode and status=='COMPLETE':status='ERROR'
    (folder/'stdout.txt').write_bytes(out);(folder/'stderr.txt').write_bytes(err)
    record={'status':status,'returncode':p.returncode,'process_wall_seconds':time.perf_counter()-start,'command':args,'job':job}
    if status=='COMPLETE':record['fit']=json.loads((folder/'fit.json').read_text())
    save(folder/'execution.json',record);event(root,'fit_completed',folder=str(folder.relative_to(root)),status=status)
    print(task,stage,config,status,record.get('fit',{}).get('validation_accuracy'),flush=True)
    return record

def choose(records):
    selected={}
    for family in ('linear','svm','tree'):
        valid=[r for r in records if r['status']=='COMPLETE' and r['job']['config']['family']==family]
        if not valid:raise ValueError('no completed '+family)
        selected[family]=max(valid,key=lambda r:r['fit']['validation_accuracy'])['job']['config']
    values={}
    for width in (64,256):
        matching=[r for r in records if r['job']['config'].get('width')==width]
        if len(matching)!=3 or any(r['status']!='COMPLETE' for r in matching): values[width]=-1.
        else:values[width]=float(np.mean([r['fit']['validation_accuracy'] for r in matching]))
    width=max(values,key=values.get)
    if values[width]<0:raise ValueError('no complete three-seed MLP')
    selected['mlp']={'family':'mlp','width':width}
    accuracy={f:max(r['fit']['validation_accuracy'] for r in records if r['status']=='COMPLETE' and r['job']['config']==selected[f]) for f in ('linear','svm','tree')}
    accuracy['mlp_mean']=values[width]
    return {'configs':selected,'validation_accuracy':accuracy,'svm_admitted_validation':
       accuracy['svm']-accuracy['linear']>=.01 and accuracy['svm']-accuracy['mlp_mean']>=.01}

def develop(a):
    records={};data=json.loads((a.root/'DATA.json').read_text())
    for task in TASKS:
        records[task]=[child(a.root,a.root/'development'/task/f'{i:02d}',task,c,'development') for i,c in enumerate(configs())]
    selected={t:choose(records[t]) for t in TASKS}
    save(a.root/'SELECTED.json',{'utc':now(),'protocol_commit':PROTOCOL_COMMIT,'data_sha256':sha(a.root/'DATA.json'),
       'selected':selected,'test_predictions':False,'records':records,'script_sha256':sha(__file__)})
    event(a.root,'selection_frozen',sha256=sha(a.root/'SELECTED.json'))
    print(json.dumps(selected,indent=2))

def refit(a):
    selected=json.loads((a.root/'SELECTED.json').read_text())['selected'];records={}
    for task in TASKS:
        cfgs=selected[task]['configs'];jobs=[(f,cfgs[f]) for f in ('linear','svm','tree')]
        jobs += [('mlp-'+str(s),{**cfgs['mlp'],'seed':s}) for s in SEEDS]
        records[task]={}
        for name,c in jobs:
            r=child(a.root,a.root/'final'/task/name,task,c,'refit');records[task][name]=r
            if r['status']!='COMPLETE':raise RuntimeError('final fit failed; test remains unopened')
    files={p.relative_to(a.root).as_posix():sha(p) for p in sorted((a.root/'final').rglob('*')) if p.is_file()}
    save(a.root/'MODEL_FREEZE.json',{'utc':now(),'protocol_commit':PROTOCOL_COMMIT,'files':files,
        'selected_sha256':sha(a.root/'SELECTED.json'),'data_sha256':sha(a.root/'DATA.json'),'test_predictions':False,
        'script_sha256':sha(__file__),'records':records})
    event(a.root,'models_frozen',sha256=sha(a.root/'MODEL_FREEZE.json'))

def verify_freeze(root):
    freeze=json.loads((root/'MODEL_FREEZE.json').read_text())
    if freeze['selected_sha256']!=sha(root/'SELECTED.json') or freeze['data_sha256']!=sha(root/'DATA.json'):
        raise ValueError('selection/data altered')
    for name,digest in freeze['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=digest:raise ValueError('frozen artifact differs')
    for task,entries in json.loads((root/'DATA.json').read_text())['splits'].items():
        for name,e in entries.items():
            if sha(root/'data'/task/(name+'.npz'))!=e['sha256']:raise ValueError('dataset changed')
    return freeze

def evaluate(a):
    from sklearn.metrics import accuracy_score,f1_score
    from threadpoolctl import threadpool_limits
    freeze=verify_freeze(a.root);folder=a.root/'evaluation';folder.mkdir(exist_ok=False)
    event(a.root,'test_opened',freeze_sha256=sha(a.root/'MODEL_FREEZE.json'))
    result={}
    with threadpool_limits(1):
        for task in TASKS:
            x,y,g=load(a.root/'data'/task/'test.npz');records={};predictions={}
            for name in freeze['records'][task]:
                with (a.root/'final'/task/name/'model.pkl').open('rb') as f:obj=pickle.load(f)
                p=obj['model'].predict(obj['scaler'].transform(x));predictions[name]=p
                np.save(folder/(task+'-'+name+'.npy'),p,allow_pickle=False)
                groups={str(v):{'rows':int(np.sum(g==v)),'correct':int(np.sum(p[g==v]==y[g==v])),
                    'accuracy':float(accuracy_score(y[g==v],p[g==v]))} for v in np.unique(g)}
                records[name]={'rows':len(y),'correct':int(np.sum(p==y)),'accuracy':float(accuracy_score(y,p)),
                    'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),'groups':groups}
            mlp_mean=float(np.mean([records['mlp-'+str(s)]['accuracy'] for s in SEEDS]))
            gain=records['svm']['accuracy']-max(records['linear']['accuracy'],mlp_mean)
            selected=json.loads((a.root/'SELECTED.json').read_text())['selected'][task]
            rx,_,_=load(a.root/'data'/task/'refit.npz')
            signatures={hashlib.sha256(row.tobytes()).digest() for row in rx}
            overlap=sum(hashlib.sha256(row.tobytes()).digest() in signatures for row in x)
            result[task]={'models':records,'mlp_mean_accuracy':mlp_mean,'svm_margin_over_best_linear_mlp':gain,
             'svm_admitted_validation':selected['svm_admitted_validation'],'svm_admitted_test':gain>=.01,
             'svm_not_worse_than_tree':records['svm']['accuracy']>=records['tree']['accuracy'],
             'exact_feature_rows_seen_in_development':overlap}
    save(folder/'QUALITY.json',{'utc':now(),'freeze_sha256':sha(a.root/'MODEL_FREEZE.json'),'tasks':result,
        'scope':'one official speaker-group test and two later gas batches; not production adoption or independent groups bootstrap'})
    event(a.root,'test_evaluated',quality_sha256=sha(folder/'QUALITY.json'))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['prepare','fit','develop','refit','evaluate'])
    for n in ('archive','out','root','job'):p.add_argument('--'+n,type=Path)
    a=p.parse_args();globals()[a.operation](a)
