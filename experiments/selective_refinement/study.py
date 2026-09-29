"""Model selection/refits using disjoint calibration groups, never test rows."""
from __future__ import annotations
import argparse,dataclasses,hashlib,json,pickle,sys,time,warnings
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from experiments.conditioned_heads.pilot import training
from experiments.budgeted_prototypes.learning import Settings,train,features,grouped_split
from experiments.budgeted_prototypes.export import encode
from experiments.budgeted_prototypes.float_control import export as export_float
from spectra.svm_export import export_prepared_svc
TASKS=('letter','pendigits','satellite','optdigits')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    with Path(p).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)

def source_snapshot(destination):
    root=Path(__file__).resolve().parents[2];destination.mkdir()
    files={}
    for name in ('selective_refinement','budgeted_prototypes','conditioned_heads','finite_kernel','adaptive_kernel'):
        for p in (root/'experiments'/name).glob('*'):
            if p.is_file() and p.suffix in ('.py','.cpp','.md'):
                relative=p.relative_to(root);dst=destination/relative;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes());files[str(relative)]=sha(p)
    for p in (root/'spectra').rglob('*'):
        if p.is_file() and p.suffix in ('.py','.cpp','.hpp'):
            relative=p.relative_to(root);dst=destination/relative;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes());files[str(relative)]=sha(p)
    return files

def grids(task,family):
    if family=='fast':return [{'gamma':g} for g in ((8.,32.) if task=='satellite' else (.125,.5,2.) if task=='optdigits' else (2.,8.))]
    if family=='strong':return [{'C':c,'gamma':g} for c in (1.,10.,100.) for g in ((.125,.5,2.) if task=='optdigits' else (.5,2.,8.))]
    return [{'width':w,'alpha':a} for w in (128,256) for a in (1e-5,.001)]

def settings(task,gamma,seed):
    return Settings(prototypes=256,gamma=gamma,epochs=200,reg=1e-6,seed=seed,arm='local')

def control(family,choice,seed):
    if family=='strong':return SVC(**choice,kernel='rbf',cache_size=128)
    return make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(choice['width'],)*2,
        alpha=choice['alpha'],learning_rate_init=.002,batch_size=128,max_iter=400,
        early_stopping=True,validation_fraction=.1,n_iter_no_change=30,random_state=seed))

def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    src=source_snapshot(a.out/'source')
    write(a.out/'LOCK.json',{'source':src,'parent':'c5af3032e1f9dd8908e922836dd7c06d4222b876',
        'training_inputs':{p.name:sha(p) for p in a.data.iterdir() if p.name.endswith('-train.npz') or p.name=='optdigits.zip'},
        'tasks':TASKS,'calibration_split_seed':20260930,'selection_split_seed':611,
        'grids':{t:{f:grids(t,f) for f in ('fast','strong','mlp')} for t in TASKS},
        'new_run':True,'timestamp_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()})
    all_selection=[];all_fits=[];total=time.process_time()
    with threadpool_limits(1):
        for task in TASKS:
            q,y,D=training(a.data,task);folder=a.out/task;folder.mkdir()
            _,groups=np.unique(q,axis=0,return_inverse=True)
            development,calibration=next(StratifiedGroupKFold(5,shuffle=True,random_state=20260930).split(q,y,groups))
            fit0,val0=grouped_split(q[development],y[development],611,6000,2000)
            fit,val=development[fit0],development[val0]
            assert not set(groups[development])&set(groups[calibration])
            np.savez_compressed(folder/'roles.npz',development=development,calibration=calibration,fit=fit,validation=val)
            choices={}
            for family in ('fast','strong','mlp'):
                records=[]
                for index,choice in enumerate(grids(task,family)):
                    dest=folder/f'{family}-choice{index}';dest.mkdir();begin=time.process_time()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        if family=='fast':
                            s=settings(task,choice['gamma'],611)
                            arrays,report=train(q[fit],y[fit],D,s)
                            pred=arrays['classes'][(features(q[val],arrays,D,s)@arrays['head']+arrays['bias']).argmax(1)]
                            np.savez_compressed(dest/'model.npz',**arrays);write(dest/'training.json',report)
                        else:
                            model=control(family,choice,611);model.fit(q[fit].astype(float)/D,y[fit]);pred=model.predict(q[val].astype(float)/D)
                            (dest/'model.pkl').write_bytes(pickle.dumps(model,protocol=4))
                    np.savez_compressed(dest/'validation.npz',prediction=pred,expected=y[val])
                    record={'task':task,'family':family,'index':index,'choice':choice,'correct':int(np.sum(pred==y[val])),
                            'fit_rows':len(fit),'validation_rows':len(val),'cpu_seconds':time.process_time()-begin,
                            'warnings':[str(w.message) for w in caught]}
                    write(dest/'record.json',record);records.append(record);all_selection.append(record)
                    print(task,family,choice,record['correct'],'/',len(val),round(record['cpu_seconds'],2),flush=True)
                chosen=max(records,key=lambda r:(r['correct'],-r['index']));choices[family]=chosen
            write(folder/'SELECTION.json',choices)
            for family in ('fast','strong','mlp'):
                choice=choices[family]['choice'];dest=folder/family;dest.mkdir();begin=time.process_time()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    if family=='fast':
                        s=settings(task,choice['gamma'],20260930);arrays,report=train(q[development],y[development],D,s)
                        np.savez_compressed(dest/'model.npz',**arrays);write(dest/'settings.json',dataclasses.asdict(s));write(dest/'training.json',report)
                        (dest/'model.spp').write_bytes(encode(arrays,D,s))
                    else:
                        model=control(family,choice,20260930);model.fit(q[development].astype(float)/D,y[development])
                        (dest/'model.pkl').write_bytes(pickle.dumps(model,protocol=4))
                        if family=='strong':
                            export_prepared_svc(model,dest/'model.srt')
                            write(dest/'training.json',{'support_vectors':len(model.support_)})
                        else:
                            scaler,network=model.steps[0][1],model.steps[1][1]
                            arrays={**{f'w{i}':v for i,v in enumerate(network.coefs_)},**{f'b{i}':v for i,v in enumerate(network.intercepts_)},
                                    'mean':scaler.mean_,'scale':scaler.scale_,'classes':network.classes_,'maximum':D}
                            np.savez_compressed(dest/'weights.npz',**arrays);(dest/'model.sfn').write_bytes(export_float(arrays))
                            write(dest/'training.json',{'iterations':int(network.n_iter_),'early_stop_scores':network.validation_scores_})
                rec={'task':task,'family':family,'choice':choice,'training_rows':len(development),'calibration_rows':len(calibration),
                     'maximum':D,'features':q.shape[1],'cpu_seconds':time.process_time()-begin,'warnings':[str(w.message) for w in caught]}
                write(dest/'fit.json',rec);all_fits.append(rec);print('FINAL',task,family,round(rec['cpu_seconds'],2),flush=True)
    write(a.out/'FINAL_MODELS.json',{'selection':all_selection,'fits':all_fits,'model_count':12,
        'cpu_seconds':time.process_time()-total,'selection_lock_sha256':sha(a.out/'LOCK.json'),
        'files':{str(p.relative_to(a.out)):sha(p) for p in a.out.rglob('*') if p.is_file() and 'source' not in p.relative_to(a.out).parts},
        'calibration_not_used_in_fitting':True,'official_tests_not_opened':True})
    print('ALL MODELS FROZEN',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
