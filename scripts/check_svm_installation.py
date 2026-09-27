"""Compile and execute SVM inference from a wheel in a framework-free fresh venv."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

PROBE = r'''
import importlib.util,json,struct,sys,zlib
from pathlib import Path
from spectra.svm import Session,build_runtime,verify_certificate
assert not any(importlib.util.find_spec(name) for name in ('torch','numpy','sklearn'))
assert Path(__import__('spectra').__file__).resolve().is_relative_to(Path(sys.prefix))
library=build_runtime(Path.cwd()/'native')
checks=2
for classes in (2,3,10):
    count=classes
    payload=struct.pack('<d',.5)+struct.pack('<'+'I'*classes,*([1]*classes))
    values=[0.]*(16*count+(classes-1)*count+classes*(classes-1)//2)
    payload+=struct.pack('<'+'d'*len(values),*values)
    model=Path.cwd()/f'model-{classes}.srt'
    model.write_bytes(struct.pack('<8sIIII',b'SPCSVM01',classes,count,len(payload),zlib.crc32(payload))+payload)
    with Session(model,library,tables=True) as session:
        result=session.predict_with_certificate([0.]*16)
        assert result.class_index==classes-1
        assert verify_certificate(classes,result.class_index,result.pair_outcomes)
        assert session.predict_many([[0.]*16,[1.]*16])==[classes-1]*2
        checks+=3
# Exercise the new generic format and shared lifetime from this installed wheel.
from spectra.svm_shared import PreparedModel
labels=['class-a','class-b','class-c'];features=3;count=3
meta=json.dumps({'labels':labels},separators=(',',':')).encode()
payload=struct.pack('<dIII',.5,1,1,1)+struct.pack('<18d',*([0.]*18))
body=meta+payload
model=Path.cwd()/'generic.srt'
model.write_bytes(struct.pack('<8sIIIIII',b'SPCSVM02',3,count,features,len(meta),len(payload),zlib.crc32(body))+body)
owner=PreparedModel(model,library,input_dtype='float64')
a,b=owner.session(),owner.session()
assert a.info['model_id']==b.info['model_id'];checks+=1
assert owner.features==3 and owner.labels==tuple(labels);checks+=1
owner.close();model.unlink()
assert a.predict([0.,0.,0.])=='class-c';checks+=1
assert b.predict_many([[0.,0.,0.],[1.,1.,1.]])==['class-c']*2;checks+=1
p=a.predict_with_certificate([0.,0.,0.],schedule='cost_aware')
assert p.label=='class-c' and verify_certificate(3,p.class_index,p.pair_outcomes);checks+=1
a.close()
assert b.predict_index([0.,0.,0.])==2;checks+=1
# The deployment probe uses only inert schema/model bytes, not sklearn or pandas.
from array import array
assert b.predict_buffer(array('d',[0.]*6))==['class-c']*2;checks+=1
assert b.predict_buffer(array('d'))==[];checks+=1
try:
    b.predict_buffer(array('f',[0.]*3))
except ValueError:checks+=1
else:raise AssertionError('float32 buffer was accepted')
assert 'numpy' not in sys.modules;checks+=1
b.close()
from spectra.svm_pipeline import Preprocessor, PreparedPipeline, SCHEMA
import hashlib
folder=Path.cwd()/'pipeline';folder.mkdir()
model=folder/'model.srt'
model.write_bytes(struct.pack('<8sIIIIII',b'SPCSVM02',3,count,features,len(meta),len(payload),zlib.crc32(body))+body)
plan={'schema':SCHEMA,'columns':['number','category'],'features':3,
      'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
      'operations':[{'kind':'numeric','column':0,'fill':(2.).hex(),'mean':(1.).hex(),'scale':(2.).hex()},
                    {'kind':'onehot','column':1,'categories':['a','b'],'unknown':'ignore'}]}
(folder/'preprocessing.json').write_text(json.dumps(plan))
prep=Preprocessor.load(folder/'preprocessing.json')
assert prep.transform([[None,'a']]).tolist()==[.5,1.,0.];checks+=1
owner=PreparedPipeline(folder,library);worker=owner.session()
assert worker.predict_many([[1.,'a'],[None,'unknown']])==['class-c']*2;checks+=1
assert worker.predict_with_certificate([1.,'a']).label=='class-c';checks+=1
try:worker.predict([float('inf'),'a'])
except ValueError:checks+=1
else:raise AssertionError('infinite raw feature accepted')
owner.close();assert worker.predict([1.,'b'])=='class-c';checks+=1
assert worker.predict_many([])==[];checks+=1
worker.close()
# Compile the optional CPython extension from the actual installed source too.
from spectra.svm_preprocess_native import build_preprocessor, NativePreprocessor
prelibrary=build_preprocessor(Path.cwd()/'preprocessing-native');checks+=1
native=NativePreprocessor(prep,prelibrary)
assert native.transform([[None,'a'],[1.,'unknown']]).tobytes()==prep.transform([[None,'a'],[1.,'unknown']]).tobytes();checks+=1
owner=PreparedPipeline(folder,library,preprocessor_library=prelibrary);worker=owner.session()
owner.close()
assert worker.predict_many([[1.,'a'],[None,'unknown']])==['class-c']*2;checks+=1
assert worker.predict_with_certificate([1.,'b']).label=='class-c';checks+=1
try:worker.predict([float('inf'),'a'])
except ValueError:checks+=1
else:raise AssertionError('compiled pipeline accepted infinity')
assert worker.predict_many([])==[];checks+=1
assert worker.predict_fused([[1.,'a'],[None,'unknown']]*129)==['class-c']*258;checks+=1
assert worker.predict_fused(iter([[1.,'a']]))==['class-c'];checks+=1
assert worker.predict_fused([],tile_rows=1)==[];checks+=1
try:worker.predict_fused([[1.,'a']]*129+[[float('inf'),'a']],tile_rows=1)
except ValueError:checks+=1
else:raise AssertionError('fused pipeline accepted late infinity')
worker.close()
try:worker.predict_fused([[1.,'a']])
except ValueError:checks+=1
else:raise AssertionError('fused pipeline used a closed worker')
# Binary streaming must execute from the installed wheel, not only source tests.
bmeta=b'{"labels":[10,20]}'
bpayload=struct.pack('<dII9d',.5,1,1,*([0.]*9))
bbody=bmeta+bpayload
bpath=Path.cwd()/'binary-stream.srt'
bpath.write_bytes(struct.pack('<8sIIIIII',b'SPCSVM02',2,2,3,len(bmeta),len(bpayload),zlib.crc32(bbody))+bbody)
with PreparedModel(bpath,library,input_dtype='float64') as bm,bm.session() as bw:
    assert bw.predict_buffer(array('d',[0.]*39),schedule='binary_stream')==[20]*13;checks+=1
    assert bw.predict_with_certificate([0.]*3,schedule='binary_stream').label==20;checks+=1
    assert bw.predict_buffer(array('d'),schedule='binary_stream')==[];checks+=1
with PreparedPipeline(folder,library,preprocessor_library=prelibrary) as pm,pm.session() as pw:
    assert pw.predict_fused([[1.,'a']]*5,schedule='binary_stream')==['class-c']*5;checks+=1
# Reentrant close must not destroy a pointer already borrowed by a request.
with PreparedModel(bpath,library,input_dtype='float64') as lm:
    lw=lm.session();run=lw._lib.sp_worker_run
    def closing_run(*args):
        lw.close()
        assert lw._handle is not None
        return run(*args)
    lw._lib.sp_worker_run=closing_run
    try:
        assert lw.predict_with_certificate([0.]*3).label==20;checks+=1
        assert lw._handle is None;checks+=1
    finally:
        lw._lib.sp_worker_run=run;lw.close()
from spectra.svm_receipt import create_receipt,verify_receipt
with PreparedModel(bpath,library,input_dtype='float64') as rm,rm.session() as rw:
    receipt=create_receipt(rw,[0.]*3,schedule='binary_stream')
assert verify_receipt(bpath,receipt,expected_input=[0.]*3,input_dtype='float64')['verified'];checks+=1
rp=Path.cwd()/'decision.json';rp.write_text(json.dumps(receipt))
import subprocess
replay='import sys;from spectra.svm_receipt import load_receipt,verify_receipt; r=verify_receipt(sys.argv[1],load_receipt(sys.argv[2]),expected_input=[0.]*3,input_dtype="float64");assert r["verified"];assert not ({"ctypes","numpy","sklearn","torch","spectra.svm_shared"}&sys.modules.keys())'
subprocess.run([sys.executable,'-I','-c',replay,str(bpath),str(rp)],check=True);checks+=1
assert not ({'pandas','torch','numpy','sklearn'} & sys.modules.keys());checks+=1
assert not ({'torch','numpy','sklearn'} & sys.modules.keys())
checks+=1
print(json.dumps({'status':'PASS','checks':checks,'package':__import__('spectra').__file__,
                  'native_build':json.loads((Path.cwd()/'native/build.json').read_text()),
                  'preprocessor_build':json.loads((Path.cwd()/'preprocessing-native/build.json').read_text())}))
'''


def check(wheel: Path, out: Path) -> dict:
    wheel = wheel.resolve(strict=True);out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    envdir = out / 'venv';venv.EnvBuilder(with_pip=True).create(envdir)
    python = envdir / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    environment = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME')}
    commands = [[str(python), '-I', '-m', 'pip', 'install', '--no-index', '--no-deps', str(wheel)],
                [str(python), '-I', '-c', PROBE]]
    results = []
    for index, command in enumerate(commands):
        process = subprocess.run(command, cwd=out, env=environment, capture_output=True, text=True)
        (out / f'command-{index}.json').write_text(json.dumps({'command': command,
          'returncode': process.returncode,'stdout':process.stdout,'stderr':process.stderr},indent=2)+'\n')
        if process.returncode:
            raise RuntimeError(f'installed-wheel check failed; see {out}/command-{index}.json')
        results.append(process.stdout)
    report = json.loads(results[-1]);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(check(args.wheel,args.out),indent=2))
