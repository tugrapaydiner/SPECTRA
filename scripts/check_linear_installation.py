"""Exercise a generated linear evaluator under isolated installed-only Python.

Use the numerical-framework-free venv created by check_svm_installation.py. This
synthetic known-coefficient model is a packaging fixture, not a training result.
"""
from array import array
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys


def main(out):
    import spectra
    from spectra.linear import _TEMPLATE, _canonical, _identity, FORMAT, build_linear, CompiledLinear
    if not Path(spectra.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError('requires installed package outside the source tree')
    forbidden={'numpy','pandas','sklearn','torch','jax'}
    if any(importlib.util.find_spec(name) for name in forbidden):raise ValueError('numerical frameworks must be absent')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);checks=0
    for classes in (2,3):
        d=3;o=1 if classes==2 else classes
        coef=[[float((c+1)*(f-1)) for f in range(d)] for c in range(o)];bias=[-.25*c for c in range(o)]
        labels=[f'class-{c}' for c in range(classes)]
        raw=struct.pack('<'+'d'*(o*d+o),*([v for row in coef for v in row]+bias))
        params=hashlib.sha256(raw).hexdigest();identity=_identity(d,labels,params)
        text=_TEMPLATE.replace('@D@',str(d)).replace('@C@',str(classes)).replace('@O@',str(o)).replace('@IDENTITY@',identity)
        text=text.replace('@W@',','.join('{'+','.join(coef[c][f].hex() for c in range(o))+'}' for f in range(d))).replace('@B@',','.join(x.hex() for x in bias))
        meta={'format':FORMAT,'features':d,'classes':classes,'outputs':o,'labels':labels,
              'parameters_sha256':params,'model_sha256':identity,'source_sha256':hashlib.sha256(text.encode()).hexdigest()}
        folder=out/f'model-{classes}';folder.mkdir();(folder/'model.cpp').write_bytes(text.encode());(folder/'parameters.f64').write_bytes(raw);(folder/'model.json').write_bytes(_canonical(meta))
        lib=build_linear(folder,out/f'build-{classes}');w=CompiledLinear(folder,lib);checks+=1
        rows=[[0.,1.,2.],[2.,1.,0.],[1.,1.,1.]];scores=[];expected=[]
        for row in rows:
            values=[]
            for c in range(o):
                s=0.
                for f in range(d):s+=row[f]*coef[c][f]
                values.append(s+bias[c])
            scores.extend(values);idx=(1 if values[0]>0 else 0) if classes==2 else max(range(classes),key=lambda i:values[i]);expected.append(labels[idx])
        packed=array('d',(x for r in rows for x in r))
        assert w.predict_buffer(packed)==expected;checks+=1
        assert w.scores_buffer(packed)==scores;checks+=1
        assert w.predict_many(rows)==expected;checks+=1
        assert w.predict_buffer(array('d'))==[];checks+=1
        try:w.predict([float('nan')]*3)
        except ValueError:checks+=1
        else:raise AssertionError('NaN accepted')
        try:w.predict_buffer(array('f',[0.]*3))
        except ValueError:checks+=1
        else:raise AssertionError('float32 silently cast')
        packed.append(0.);checks+=1
    if forbidden&sys.modules.keys():raise ValueError('framework imported')
    report={'status':'PASS','checks':checks,'package':spectra.__file__,'scope':'two fixed synthetic models compiled and executed from installed-only source; not benchmark data'}
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
