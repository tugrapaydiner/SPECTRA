"""Mutate copied evidence only, requiring the independent checker to reject it."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
from .audit import audit


def run(root,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);copy=out/'copy';shutil.copytree(root,copy)
    audit(copy)
    def change_json(path,fn):
        obj=json.loads(path.read_text());fn(obj);return json.dumps(obj).encode()
    cases=[
        ('short-timing','results/benchmark/letter.jsonl',lambda p:b'\n'.join(p.read_bytes().splitlines()[:-1])+b'\n'),
        ('wrong-output','results/benchmark/letter.jsonl',lambda p:p.read_bytes().replace(b'"indices_sha256": "',b'"indices_sha256": "f',1)),
        ('negative-time','results/benchmark/letter.jsonl',lambda p:p.read_bytes().replace(b'"wall_ns": ',b'"wall_ns": -',1)),
        ('wrong-speedup','results/benchmark/SUMMARY.json',lambda p:change_json(p,lambda x:x.update(geometric_ratio=.5))),
        ('false-gate','results/benchmark/SUMMARY.json',lambda p:change_json(p,lambda x:x.update(performance_gate=False))),
        ('changed-prefix','results/benchmark/letter_prefix.jsonl',lambda p:p.read_bytes()+b'{}\n'),
        ('wrong-order','results/benchmark/LOCK.json',lambda p:change_json(p,lambda x:x['jobs'].reverse())),
        ('changed-model','results/replay/letter/interned.sctt',lambda p:p.read_bytes()[:-1]+bytes([p.read_bytes()[-1]^1])),
        ('false-coverage','results/replay/letter/result.json',lambda p:change_json(p,lambda x:x['models']['interned']['work']['total'].update(unresolved=1))),
        ('changed-stress-score','results/replay/letter/uniform-scores.f64',lambda p:p.read_bytes()[:-1]+bytes([p.read_bytes()[-1]^1])),
        ('missing-proof','results/independent.json',lambda p:change_json(p,lambda x:x['files'].pop())),
        ('nonisolated-resource','results/resources/LOCK.json',lambda p:change_json(p,lambda x:x.update(isolated=False))),
        ('empty-resource-schedule','results/resources/LOCK.json',lambda p:change_json(p,lambda x:x.update(jobs=[]))),
        ('short-resource-schedule','results/resources/LOCK.json',lambda p:change_json(p,lambda x:x['jobs'].pop())),
        ('long-resource-schedule','results/resources/LOCK.json',lambda p:change_json(p,lambda x:x['jobs'].append(x['jobs'][0]))),
        ('empty-resource-output','results/resources/process-0.json',lambda p:change_json(p,lambda x:x.update(stdout='{}'))),
    ]
    results=[]
    for name,file,mutate in cases:
        path=copy/file;before=path.read_bytes()
        try:
            after=mutate(path)
            if after==before:raise AssertionError('empty mutation')
            path.write_bytes(after)
            try:audit(copy)
            except ValueError as error:results.append({'case':name,'rejected':True,'reason':str(error)})
            else:raise AssertionError('corruption accepted: '+name)
        finally:path.write_bytes(before)
    result={'status':'PASS','cases':results,'rejected':len(results),'scope':'deliberately corrupted copies, not new examples'}
    (out/'report.json').write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    print(json.dumps(run(**vars(p.parse_args())),indent=2))
