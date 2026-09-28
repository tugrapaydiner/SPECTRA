"""Pre-test interaction amendment; preserve every earlier selection unchanged."""
from __future__ import annotations
import argparse,json,pickle,time,sys
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.learned_finite.learning import distances,table,GAMMA_MULTIPLIERS,CS,align_tables,spectral_tables
from experiments.learned_finite.train import fit_one
from experiments.learned_finite.model import encode
from experiments.learned_finite.data import sha,write_json

def run(a):
    a.out.mkdir(parents=True,exist_ok=False);started=time.process_time();results=[]
    original=json.loads((a.original/'TRAINING_COMPLETE.json').read_text())
    for f,expected in original['files'].items():
        if sha(a.original/f)!=expected:raise ValueError('original selection modified')
    with threadpool_limits(1):
      for name in ('car','semeion'):
        folder=a.data/name;meta=json.loads((folder/'manifest.json').read_text());data=np.load(folder/'development.npz',allow_pickle=False)
        x=data['x'];y=data['y'];d=x.shape[1]
        for seed in (101,202,303):
            dest=a.out/f'{name}-{seed}';dest.mkdir();split=meta['inner_splits'][str(seed)]
            fit=x[split['fit']];valid=x[split['validation']];target=y[split['fit']];wanted=y[split['validation']]
            old=np.load(a.original/f'{name}-{seed}'/'metric_rbf-reference.npz',allow_pickle=False)
            w=np.maximum(old['weights'],1).astype(np.uint8)
            D=distances(fit,fit,w);VD=distances(valid,fit,w);scales=GAMMA_MULTIPLIERS/np.median(D[D>0])
            options={'metric_floor':('hamming',w,D,VD,[(table([g],[1.],int(w.sum())),{'gamma':float(g)}) for g in scales])}
            I=(fit.astype(np.int32)@fit.astype(np.int32).T);VI=valid.astype(np.int32)@fit.astype(np.int32).T
            scale=float(np.median(I[I>0]));grid=np.arange(d+1,dtype=np.float64);poly=[]
            for degree in (2,3,4,5):
                for offset in (0.,1.):poly.append(((grid/scale+offset)**degree,{'degree':degree,'offset':offset,'scale':scale}))
            options['polynomial']=('intersection',np.ones(d,np.uint8),I,VI,poly)
            mixed,mixinfo=align_tables(np.stack([p[0] for p in poly]),I,target)
            options['polynomial_aligned']=('intersection',np.ones(d,np.uint8),I,VI,[(mixed,mixinfo)])
            H=distances(fit,fit,np.ones(d));VH=distances(valid,fit,np.ones(d));scales=GAMMA_MULTIPLIERS/np.median(H[H>0])
            orders=[r for r in (1,2,3,4,6,8,12,16) if r<=d]
            bases=np.concatenate((spectral_tables(d,orders),np.stack([table([g],[1.],d) for g in scales])))
            lut,info=align_tables(bases,H,target);info.update(orders=orders,scales=scales.tolist())
            options['spectral_aligned']=('hamming',np.ones(d,np.uint8),H,VH,[(lut,info)])
            records=[]
            for family,(kind,weights,D,VD,choices) in options.items():
                best=None;attempts=[];cpu=0.
                for lut,params in choices:
                    K=np.ascontiguousarray(lut[D]);V=np.ascontiguousarray(lut[VD])
                    for C in CS:
                        model=SVC(kernel='precomputed',C=C);elapsed,log=fit_one(model,K,target);cpu+=elapsed
                        pred=model.predict(V);correct=int((pred==wanted).sum());supports=len(model.support_)
                        attempts.append({'parameters':{**params,'C':C},'validation_correct':correct,'supports':supports,'cpu_seconds':elapsed,'warnings':log})
                        rank=correct,-supports
                        if best is None or rank>best[0]:best=(rank,model,lut,params,C,pred.copy())
                _,model,lut,params,C,pred=best
                raw=encode(fit[model.support_],meta['labels'],weights,lut,model.n_support_,model.dual_coef_,model.intercept_,kind)
                (dest/(family+'.lfk')).write_bytes(raw)
                (dest/(family+'.pkl')).write_bytes(pickle.dumps({'model':model,'fit':fit,'weights':weights,'table':lut,'signature':kind},protocol=4))
                np.savez_compressed(dest/(family+'-reference.npz'),weights=weights,table=lut,supports=fit[model.support_],counts=model.n_support_,dual=model.dual_coef_,bias=model.intercept_,validation_predictions=pred)
                record={'family':family,'signature':kind,'validation_correct':int((pred==wanted).sum()),'validation_rows':len(wanted),'supports':len(model.support_),
                        'cpu_seconds':cpu,'attempts':attempts,'selected':{**params,'C':C},'model_sha256':sha(dest/(family+'.lfk')),'model_bytes':len(raw)}
                write_json(dest/(family+'.json'),record);records.append(record)
                print(name,seed,family,record['validation_correct'],'/',len(wanted),'SV',len(model.support_),flush=True)
            write_json(dest/'selection.json',{'dataset':name,'seed':seed,'models':records});results.append({'dataset':name,'seed':seed,'models':records})
    record={'status':'EXTENSIONS_SELECTED_BEFORE_TEST','cpu_seconds':time.process_time()-started,'records':results,
            'original_training_receipt_sha256':sha(a.original/'TRAINING_COMPLETE.json'),
            'files':{str(p.relative_to(a.out)):sha(p) for p in sorted(a.out.rglob('*')) if p.is_file()}}
    write_json(a.out/'EXTENSIONS_COMPLETE.json',record);print('CPU',record['cpu_seconds'])
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('data','original','out'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
