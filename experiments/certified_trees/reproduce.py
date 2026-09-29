"""New fixed-source replay; never infer recovery of old model bytes or accuracy."""
from __future__ import annotations
import argparse,hashlib,io,json,platform,sys,time,zipfile
from pathlib import Path
TASKS=('letter','pendigits','satellite','optdigits')
PARAMS=dict(iterations=256,depth=6,learning_rate=.08,l2_leaf_reg=3,
            bootstrap_type='No',random_strength=0,random_seed=20260929,
            thread_count=1,loss_function='MultiClass',allow_writing_files=False,verbose=False)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,record):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(record,f,indent=2,allow_nan=False)
def load(data,task,part):
    import numpy as np
    if task=='optdigits':
        with zipfile.ZipFile(data/'optdigits.zip') as z:raw=z.read('optdigits.tra' if part=='train' else 'optdigits.tes')
        m=np.loadtxt(io.BytesIO(raw),delimiter=',',dtype=np.int64)
        if m.shape!=(3823 if part=='train' else 1797,65) or (m[:,:-1]<0).any() or (m[:,:-1]>16).any():raise ValueError('optical schema')
        return m[:,:-1].astype(np.uint8),m[:,-1],16
    with np.load(data/f'{task}-{part}.npz',allow_pickle=False) as z:q=z['q'].copy();y=z['y'].copy();maximum=int(z['maximum'])
    if q.dtype!=np.uint8 or q.ndim!=2 or y.shape!=(len(q),) or (q>maximum).any():raise ValueError('input schema')
    return q,y,maximum

def fit(args):
    import catboost,numpy,sklearn
    if catboost.__version__!='1.2.8':raise ValueError('fixed source version is CatBoost1.2.8')
    args.out.mkdir(parents=True,exist_ok=False)
    inputs={p.name:sha(p) for p in args.data.iterdir() if p.is_file()}
    write(args.out/'FIT_PROTOCOL.json',{'parameters':PARAMS,'inputs':inputs,'script_sha256':sha(__file__),
        'public_protocol_commit':'0d35521b05e155a7e1d589bb7147f5d9a0ee3b91',
        'scope':'new fixed-recipe compatibility models, no tuning/no new holdout',
        'python':sys.version,'catboost':catboost.__version__,'numpy':numpy.__version__,'sklearn':sklearn.__version__,'platform':platform.platform()})
    records=[]
    for task in TASKS:
        q,y,D=load(args.data,task,'train');dest=args.out/task;dest.mkdir()
        start=time.process_time();wall=time.perf_counter()
        model=catboost.CatBoostClassifier(**PARAMS).fit(q,y)
        seconds=time.process_time()-start;elapsed=time.perf_counter()-wall
        for fmt in ('cbm','json','cpp'):model.save_model(str(dest/('model.'+fmt)),format=fmt)
        rec={'task':task,'training_rows':len(q),'features':q.shape[1],'maximum':D,
             'classes':model.classes_.tolist(),'fit_cpu_seconds':seconds,'fit_wall_seconds':elapsed,
             'files':{p.name:sha(p) for p in dest.iterdir() if p.is_file()}}
        write(dest/'FIT.json',rec);records.append(rec);print(task,seconds,flush=True)
    write(args.out/'MODEL_LOCK.json',{'models':records,'files':{p.relative_to(args.out).as_posix():sha(p) for p in args.out.rglob('*') if p.is_file()},
          'script_sha256':sha(__file__),'test_predictions_made':False,'scope':'all four new fixed models saved before any test prediction'})

def open_tests(args):
    import catboost,numpy as np
    lock=json.loads((args.models/'MODEL_LOCK.json').read_text())
    for n,h in lock['files'].items():
        if sha(args.models/n)!=h:raise ValueError('frozen model changed')
    args.out.mkdir(parents=True,exist_ok=False)
    write(args.out/'OPENING.json',{'model_lock_sha256':sha(args.models/'MODEL_LOCK.json'),'script_sha256':sha(__file__),
           'scope':'exposed original tests; compatibility-source quality, not learning gain'})
    results={}
    for task in TASKS:
        q,y,D=load(args.data,task,'test');model=catboost.CatBoostClassifier();model.load_model(str(args.models/task/'model.cbm'))
        scores=np.asarray(model.predict(q,prediction_type='RawFormulaVal',thread_count=1),dtype='<f8')
        indices=scores.argmax(1).astype('<i4');pred=model.classes_[indices]
        d=args.out/task;d.mkdir();q.tofile(d/'input.u8');np.save(d/'truth.npy',y);indices.tofile(d/'indices.i32');scores.tofile(d/'source_scores.f64')
        results[task]={'rows':len(q),'features':q.shape[1],'maximum':D,'correct':int((pred==y).sum()),
          'classes':model.classes_.tolist(),'accuracy':float((pred==y).mean()),'files':{p.name:sha(p) for p in d.iterdir()}}
        print(task,results[task]['correct'],len(y),flush=True)
    write(args.out/'QUALITY.json',results)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['fit','open_tests'])
    for k in ('data','models','out'):p.add_argument('--'+k,type=Path)
    a=p.parse_args();globals()[a.operation](a)
