"""Training-only head-conditioning pilot; every arm shares frozen geometry/start."""
from __future__ import annotations
import argparse, dataclasses, hashlib, io, json, os, time, zipfile
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.budgeted_prototypes.learning import Settings,train,features,grouped_split
from experiments.budgeted_prototypes.export import encode
from .optimizer import fit_head
TASKS=('letter','pendigits','satellite','optdigits')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    with Path(p).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)

def training(data,task):
    if task=='optdigits':
        with zipfile.ZipFile(data/'optdigits.zip') as z:
            m=np.loadtxt(io.BytesIO(z.read('optdigits.tra')),delimiter=',',dtype=int)
        assert m.shape==(3823,65) and m[:,:64].min()>=0 and m[:,:64].max()<=16
        return m[:,:64].astype(np.uint8),m[:,64],16
    with np.load(data/(task+'-train.npz')) as d:
        return d['q'],d['y'],int(d['maximum'])

def run(args):
    args.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]
    source_files=sorted((root/'experiments/conditioned_heads').glob('*.py'))+[
        root/'experiments/budgeted_prototypes'/n for n in ('learning.py','export.py','session.py')]
    source={str(f.relative_to(root)):sha(f) for f in source_files}
    write(args.out/'LOCK.json',{'source':source,'inputs':{p.name:sha(p) for p in args.data.iterdir() if p.name.endswith('-train.npz') or p.name=='optdigits.zip'},
                              'tasks':TASKS,'split_seed':611,'role':'training partitions only',
                              'new_run_not_prior_bytes':True,'timestamp_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()})
    results=[];start=time.process_time()
    with threadpool_limits(1):
        for task in TASKS:
            q,y,D=training(args.data,task);fit,validation=grouped_split(q,y,611,4000,1500)
            folder=args.out/task;folder.mkdir();np.savez_compressed(folder/'split.npz',fit=fit,validation=validation)
            s=Settings(prototypes=256,gamma=32. if task=='satellite' else .5 if task=='optdigits' else 8.,epochs=80,reg=1e-6,seed=611,arm='local')
            arrays,report=train(q[fit],y[fit],D,s,final_head=False)
            np.savez_compressed(folder/'frozen_geometry_and_start.npz',**arrays);write(folder/'geometry_fit.json',report)
            labels,targets=np.unique(y[fit],return_inverse=True)
            F=features(q[fit],arrays,D,s);V=features(q[validation],arrays,D,s)
            print(task,'geometry complete',round(report['cpu_seconds'],3),flush=True)
            outputs={}
            for tag,method,iterations,start_tag in [
                ('raw150','raw',150,None),('raw300','raw',300,None),
                ('diagonal150','diagonal',150,None),('cholesky150','cholesky',150,None),
                ('raw_continue150','raw',150,'raw150'),('cholesky_continue150','cholesky',150,'raw150')]:
                h,b=outputs[start_tag][:2] if start_tag else (arrays['head'],arrays['bias'])
                nh,nb,r=fit_head(F,targets,h,b,ridge=s.reg,method=method,maxiter=iterations)
                pred=labels[(V@nh+nb).argmax(1)]
                final={**arrays,'head':nh,'bias':nb}
                np.savez_compressed(folder/(tag+'.npz'),**final)
                (folder/(tag+'.spp')).write_bytes(encode(final,D,s))
                np.savez_compressed(folder/(tag+'-predictions.npz'),prediction=pred,expected=y[validation])
                result={'task':task,'tag':tag,'correct':int(np.sum(pred==y[validation])),'validation_rows':len(validation),**r,
                        'start':start_tag or 'same_geometry_training_head'}
                write(folder/(tag+'.json'),result);outputs[tag]=(nh,nb,r);results.append(result)
                print(task,tag,'objective',round(r['objective'],6),'grad',f"{r['original_gradient_inf']:.2g}",'correct',result['correct'], '/',len(validation),'CPU',round(r['cpu_seconds'],3),r['message'],flush=True)
    write(args.out/'RESULTS.json',{'rows':results,'cpu_seconds':time.process_time()-start,'scope':'training-only diagnosis, no official test opened'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
