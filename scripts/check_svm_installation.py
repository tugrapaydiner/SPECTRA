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
b.close()
assert not ({'torch','numpy','sklearn'} & sys.modules.keys())
checks+=1
print(json.dumps({'status':'PASS','checks':checks,'package':__import__('spectra').__file__,
                  'native_build':json.loads((Path.cwd()/'native/build.json').read_text())}))
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
