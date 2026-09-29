"""Deliberately corrupt disposable evidence copies and require rejection."""
from __future__ import annotations
import argparse,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.integer_interactions.audit import audit

def main(a):
    shutil.copytree(a.root,a.work)
    audit(a.work,a.source)
    def obj(p,fn):
        data=json.loads(p.read_text());fn(data);return json.dumps(data).encode()
    def row(p,fn):
        data=p.read_text().splitlines();fn(data);return ('\n'.join(data)+'\n').encode()
    cases=[
        ('missing-primary','selection/rows.jsonl',lambda p:row(p,lambda r:r.pop())),
        ('duplicate-primary','selection/rows.jsonl',lambda p:row(p,lambda r:r.append(r[0]))),
        ('missing-secondary','local_selection/rows.jsonl',lambda p:row(p,lambda r:r.pop())),
        ('changed-choice','selection/selected.json',lambda p:obj(p,lambda r:r['letter']['full'].update(C=999.))),
        ('changed-secondary-choice','local_selection/selected.json',lambda p:obj(p,lambda r:r['letter']['local_supervised'].update(gamma=999.))),
        ('changed-model','models/letter-full/model.sik',lambda p:p.read_bytes()+b'x'),
        ('changed-fitting-source','source_snapshots/refit/learning.py',lambda p:p.read_bytes()+b'\n# damaged'),
        ('changed-test-data','parent/datasets/letter-test.npz',lambda p:p.read_bytes()+b'x'),
        ('changed-quality','evaluation/quality.json',lambda p:obj(p,lambda r:r['letter']['full'].update(correct=4000))),
        ('changed-paired-result','evaluation/quality.json',lambda p:obj(p,lambda r:r['letter']['local_vs_diagonal'].update(points=10.))),
        ('missing-timing','benchmark/rows.jsonl',lambda p:row(p,lambda r:r.pop())),
        ('wrong-time-type','benchmark/rows.jsonl',lambda p:p.read_bytes().replace(b'"ns": ',b'"ns": true,"old_ns": ',1)),
        ('wrong-output-digest','benchmark/rows.jsonl',lambda p:p.read_bytes().replace(b'"output_sha256": "',b'"output_sha256": "0',1)),
        ('forged-success','benchmark/summary.json',lambda p:obj(p,lambda r:r.update(primary_two_task_gate=True))),
        ('altered-library','build-final/interaction.so',lambda p:p.read_bytes()+b'changed'),
        ('altered-reference-margin','evaluation/fidelity.json',lambda p:obj(p,lambda r:r[0].update(original_sha256='0'*64))),
    ]
    observations=[]
    for name,path,change in cases:
        p=a.work/path;before=p.read_bytes()
        try:
            modified=change(p)
            if modified==before:raise AssertionError('ineffective corruption')
            p.write_bytes(modified)
            try:audit(a.work,a.source)
            except ValueError as e:observations.append({'case':name,'rejected':True,'reason':str(e)})
            else:raise AssertionError('accepted corruption '+name)
        finally:p.write_bytes(before)
    result={'status':'PASS','cases':observations,'rejected':len(observations),'scope':'copied receipt corruption, not new model accuracy cases'}
    with a.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','source','work','out'):p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
