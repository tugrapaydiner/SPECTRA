"""Independent integer/ordered-score replay. No learning or model selection."""
from __future__ import annotations
import argparse,hashlib,json,pickle,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.learned_finite.model import Model
from experiments.learned_finite.session import Session

def reference(model,x):
    # Direct coordinate definition, not native bitplane or training dot formula.
    support=np.zeros((model.n,model.d),dtype=np.uint8)
    for k in range(model.n):
        for f in range(model.d):support[k,f]=(model.supports[k*model.words+f//64]>>(f%64))&1
    w=np.asarray(model.weights,dtype=np.uint16);ker=np.empty((len(x),model.n))
    lut=np.asarray(model.table)
    for start in range(0,len(x),32):
        rows=np.asarray(x[start:start+32],dtype=np.uint8)
        if model.kind:terms=rows[:,None,:]&support[None,:,:]
        else:terms=rows[:,None,:]^support[None,:,:]
        sig=(terms*w).sum(axis=2,dtype=np.uint32);ker[start:start+len(rows)]=lut[sig]
    out=np.empty((len(x),len(model.pairs)),dtype=np.float64);votes=np.zeros((len(x),model.c),int)
    for p,(i,j,terms,bias) in enumerate(model.pairs):
        score=np.zeros(len(x),dtype=np.float64)
        for k,a in terms:score=score+float(a)*ker[:,k]
        score=score+float(bias);out[:,p]=score
        winner=np.where(score>0,i,j) if model.c>2 else (score>=0).astype(int)
        votes[np.arange(len(x)),winner]+=1
    pred=np.array(model.labels)[votes.argmax(axis=1)]
    return pred.tolist(),out

def validate_development(data,roots,library,out):
    report=[]
    for root in roots:
      for folder in sorted(root.glob('*-*')):
        if not folder.is_dir():continue
        name,seed=folder.name.rsplit('-',1)
        meta=json.loads((data/name/'manifest.json').read_text());dev=np.load(data/name/'development.npz',allow_pickle=False)
        x=dev['x'][meta['inner_splits'][seed]['validation']]
        for path in sorted(folder.glob('*.lfk')):
            model=Model.load(path);wanted,margins=reference(model,x)
            original=np.load(path.with_name(path.stem+'-reference.npz'),allow_pickle=False)['validation_predictions']
            labels=np.array(meta['labels'])[original].tolist()
            if wanted!=labels:raise AssertionError('independent vs sklearn '+str(path))
            with Session(path,library) as s:
                for mode in ('selective','exhaustive','scalar'):
                    if s.predict_buffer(x,mode)!=wanted:raise AssertionError('native '+str(path)+' '+mode)
                observed=s.margins(x)
                if observed!=margins.astype('<f8').tobytes():raise AssertionError('margin bytes '+str(path))
            report.append({'model':folder.name+'/'+path.name,'rows':len(x),'margins':margins.size,
                           'margin_sha256':hashlib.sha256(observed).hexdigest(),'model_sha256':model.sha256})
    result={'status':'PASS','models':len(report),'model_input_pairs':sum(r['rows'] for r in report),
            'ordered_margins':sum(r['margins'] for r in report),'details':report,'scope':'development replay only; no holdout data opened'}
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='details'},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--root',type=Path,action='append',required=True)
    p.add_argument('--library',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    validate_development(a.data,a.root,a.library,a.out)
