"""Fit/validate only. This command never opens a holdout.npz file."""
from __future__ import annotations
import argparse, hashlib,json,pickle,time,warnings,sys
from pathlib import Path
import numpy as np
from sklearn.svm import SVC,LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.learned_finite.learning import distances,learn_metric,learn_mixture,table,GAMMA_MULTIPLIERS,CS
from experiments.learned_finite.model import encode
from experiments.learned_finite.data import sha,write_json

def fit_one(model,x,y):
    start=time.process_time()
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter('always');model.fit(x,y)
    return time.process_time()-start,[str(w.message) for w in log]

def run(a):
    a.out.mkdir(parents=True,exist_ok=False);all_records=[];total_cpu=time.process_time();wall=time.perf_counter()
    env={'python':sys.version,'versions':{n:__import__(n).__version__ for n in ('numpy','scipy','sklearn','torch')},
         'source_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},'holdout_opened':False}
    write_json(a.out/'TRAINING_START.json',env)
    with threadpool_limits(1):
      for name in ('car','semeion'):
        folder=a.data/name;manifest=json.loads((folder/'manifest.json').read_text())
        if sha(folder/'development.npz')!=manifest['development_sha256']:raise ValueError('development bytes differ')
        development=np.load(folder/'development.npz',allow_pickle=False)
        x=development['x'];y=development['y'];groups=development['feature_groups'];classes=manifest['labels']
        for seed in (101,202,303):
            output=a.out/f'{name}-{seed}';output.mkdir()
            split=manifest['inner_splits'][str(seed)];tr=np.array(split['fit']);va=np.array(split['validation'])
            fit=x[tr];valid=x[va];target=y[tr];wanted=y[va];records=[]
            weights,metric=learn_metric(fit,target,groups);write_json(output/'metric.json',metric)
            print(name,seed,'metric',metric['group_integer_weights'],'cpu',round(metric['cpu_seconds'],3),flush=True)
            for metric_name,w in (('isotropic',np.ones(x.shape[1],np.uint8)),('metric',weights)):
                D=distances(fit,fit,w);VD=distances(valid,fit,w);median=float(np.median(D[D>0]));scales=GAMMA_MULTIPLIERS/median
                beta,alignment=learn_mixture(D,target,scales);write_json(output/(metric_name+'-alignment.json'),alignment)
                options={ 'rbf':[(table([gamma],[1.],int(w.sum())),{'gamma':float(gamma)}) for gamma in scales],
                          'uniform':[(table(scales,np.ones(8)/8,int(w.sum())),{'scales':scales.tolist(),'mixture':[.125]*8})],
                          'aligned':[(table(scales,beta,int(w.sum())),{'scales':scales.tolist(),'mixture':beta.tolist()})] }
                for family,choices in options.items():
                    label=metric_name+'_'+family;best=None;cpu=0.;attempts=[]
                    for lut,params in choices:
                        gram=np.ascontiguousarray(lut[D]);vgram=np.ascontiguousarray(lut[VD])
                        for C in CS:
                            model=SVC(kernel='precomputed',C=C)
                            elapsed,log=fit_one(model,gram,target);cpu+=elapsed
                            pred=model.predict(vgram);correct=int((pred==wanted).sum());supports=len(model.support_)
                            attempt={'parameters':{**params,'C':C},'validation_correct':correct,'validation_rows':len(wanted),
                                     'supports':supports,'cpu_seconds':elapsed,'warnings':log};attempts.append(attempt)
                            rank=(correct,-supports)
                            if best is None or rank>best[0]:best=(rank,model,lut,params,C,pred.copy())
                    _,model,lut,params,C,pred=best
                    raw=encode(fit[model.support_],classes,w,lut,model.n_support_,model.dual_coef_,model.intercept_)
                    (output/(label+'.lfk')).write_bytes(raw)
                    np.savez_compressed(output/(label+'-reference.npz'),weights=w,table=lut,supports=fit[model.support_],
                                        counts=model.n_support_,dual=model.dual_coef_,bias=model.intercept_,validation_predictions=pred)
                    # A trusted local reference for final sklearn prediction, not deployment.
                    (output/(label+'.pkl')).write_bytes(pickle.dumps({'model':model,'fit':fit,'weights':w,'table':lut},protocol=4))
                    record={'family':label,'validation_correct':int((pred==wanted).sum()),'validation_rows':len(wanted),
                            'supports':len(model.support_),'cpu_seconds':cpu,'attempts':attempts,'selected':{**params,'C':C},
                            'model_bytes':len(raw),'model_sha256':hashlib.sha256(raw).hexdigest()}
                    write_json(output/(label+'.json'),record);records.append(record)
                    print(name,seed,label,record['validation_correct'],'/',len(wanted),'SV',record['supports'],flush=True)
            control_defs={
                'linear':[LinearSVC(C=C,dual='auto',max_iter=10000,random_state=seed) for C in CS],
                'mlp':[MLPClassifier(hidden_layer_sizes=h,alpha=alpha,max_iter=300,random_state=seed,early_stopping=False)
                       for h in ((32,),(64,),(32,32),(64,64)) for alpha in (.0001,.01)],
                'tree':[DecisionTreeClassifier(max_depth=depth,random_state=seed) for depth in (4,8,16,None)],
                'extra_trees':[ExtraTreesClassifier(n_estimators=200,n_jobs=1,random_state=seed)]}
            for family,models in control_defs.items():
                best=None;attempts=[];cpu=0.
                for i,model in enumerate(models):
                    elapsed,log=fit_one(model,fit,target);cpu+=elapsed
                    pred=model.predict(valid);correct=int((pred==wanted).sum())
                    attempts.append({'parameters':model.get_params(),'validation_correct':correct,'cpu_seconds':elapsed,'warnings':log})
                    if best is None or correct>best[0]:best=(correct,model,pred.copy())
                correct,model,pred=best;raw=pickle.dumps(model,protocol=4);(output/(family+'.pkl')).write_bytes(raw)
                record={'family':family,'validation_correct':correct,'validation_rows':len(wanted),'cpu_seconds':cpu,
                        'attempts':attempts,'model_sha256':hashlib.sha256(raw).hexdigest(),'pickle_bytes':len(raw)}
                write_json(output/(family+'.json'),record);records.append(record)
                print(name,seed,family,correct,'/',len(wanted),flush=True)
            np.savez_compressed(output/'validation.npz',x=valid,y=wanted,original_indices=development['original_indices'][va])
            write_json(output/'selection.json',{'dataset':name,'seed':seed,'development_sha256':manifest['development_sha256'],
                         'fit_indices':tr.tolist(),'validation_indices':va.tolist(),'models':records})
            all_records.append({'dataset':name,'seed':seed,'models':records,'metric_cpu_seconds':metric['cpu_seconds']})
    record={'status':'ALL_MODELS_SELECTED_BEFORE_TEST','training_cpu_seconds':time.process_time()-total_cpu,
            'wall_seconds':time.perf_counter()-wall,'records':all_records,
            'files':{str(p.relative_to(a.out)):sha(p) for p in sorted(a.out.rglob('*')) if p.is_file()}}
    write_json(a.out/'TRAINING_COMPLETE.json',record)
    print('TOTAL CPU',record['training_cpu_seconds'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
