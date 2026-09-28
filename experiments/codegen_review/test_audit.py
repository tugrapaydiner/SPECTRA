"""Small synthetic records exercise the auditor; no clocks/models are fabricated as evidence."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random

import pytest

spec=importlib.util.spec_from_file_location('generation_audit',Path(__file__).with_name('audit.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def records(tmp_path):
    run=tmp_path/'run';models=tmp_path/'models';packages=tmp_path/'packages';prior=tmp_path/'prior'
    script=tmp_path/'benchmark.py';script.write_text('# synthetic fixture only\n')
    jobs=[(n,r,v) for n in module.NAMES for r in range(1 if n=='har' else 3) for v in module.VARIANTS]
    random.Random(20260928).shuffle(jobs)
    ids={};pkg={};rows=[]
    for name in module.NAMES:
        p=models/name/'model.pkl';p.parent.mkdir(parents=True);p.write_bytes(b'not a real model')
        ids[name]=module.sha(p)
        if name!='har':
            p=prior/name/'model.c';p.parent.mkdir(parents=True);p.write_bytes(b'/* synthetic */\n')
    for v in module.VARIANTS:
        p=packages/(v+'-review')/'m2cgen/__init__.py';p.parent.mkdir(parents=True);p.write_bytes(b'# fixture\n')
        pkg[v]={'m2cgen/__init__.py':module.sha(p)}
    put(run/'protocol.json',{'jobs':jobs,'seed':20260928,'cap_seconds':120,
        'script_sha256':module.sha(script),'models':ids,'packages':pkg})
    for name,repeat,variant in jobs:
        p=run/f'{name}-{repeat}-{variant}';p.mkdir()
        row={'model':name,'repeat':repeat,'variant':variant,'status':'COMPLETE','returncode':0,'process_seconds':1.}
        if name=='har' and variant=='stock':
            row.update(status='TIMEOUT',returncode=-9,process_seconds=120.1)
        else:
            (p/'model.c').write_bytes(b'/* synthetic */\n')
            obs={'generation_ns':100,'generation_cpu_ns':100,'assembly_ns':10,
                 'source_bytes':16,'peak_process_kib':100,'source_sha256':module.sha(p/'model.c')}
            put(p/'generation.json',obs);row.update(obs)
        rows.append(row)
    (run/'attempts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return run,models,packages,script,prior


def test_valid_synthetic_inventory(records):
    result=module.audit(*records)
    assert (result['attempts'],result['completed'],result['timeouts'])==(57,56,1)


@pytest.mark.parametrize('damage',['missing','order','time','source','prior','package','script','model','cap','timeout','duplicate','nonfinite'])
def test_corrupted_generation_receipts_rejected(records,damage):
    run,models,packages,script,prior=records
    path=run/'attempts.jsonl';rows=[json.loads(l) for l in path.read_text().splitlines()]
    if damage=='missing':rows.pop()
    elif damage=='order':rows[0],rows[1]=rows[1],rows[0]
    elif damage=='time':rows[0]['generation_ns']=999999
    elif damage=='source':
        r=next(r for r in rows if r['status']=='COMPLETE')
        (run/f"{r['model']}-{r['repeat']}-{r['variant']}"/'model.c').write_text('changed')
    elif damage=='prior':(prior/module.NAMES[0]/'model.c').write_text('changed')
    elif damage=='package':(packages/'stock-review/m2cgen/__init__.py').write_text('changed')
    elif damage=='script':script.write_text('changed')
    elif damage=='model':(models/module.NAMES[0]/'model.pkl').write_text('changed')
    elif damage=='cap':
        p=run/'protocol.json';doc=json.loads(p.read_text());doc['cap_seconds']=121;put(p,doc)
    elif damage=='timeout':next(r for r in rows if r['status']=='TIMEOUT')['process_seconds']=2.
    elif damage=='duplicate':
        path.write_text(path.read_text().replace('"repeat":', '"repeat":-100,"repeat":',1))
    else:path.write_text(path.read_text().replace('"process_seconds": 1.0','"process_seconds": NaN',1))
    if damage not in ('duplicate','nonfinite'):
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):module.audit(*records)
