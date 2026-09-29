"""Reject altered copies of actual receipts; never mutate original evidence."""
from pathlib import Path
import argparse,json,shutil
from .audit import audit

def run(root,source,out):
    out.mkdir(parents=True,exist_ok=False);copy=out/'copy';shutil.copytree(root,copy)
    def changejson(p,fn):
        doc=json.loads(p.read_text());fn(doc);return json.dumps(doc).encode()
    cases=[
      ('missing-row','benchmark_layouts/rows.jsonl',lambda p:b'\n'.join(p.read_bytes().splitlines()[:-1])+b'\n'),
      ('false-output','benchmark_layouts/letter-register_16_official.indices',lambda p:b'\xff\xff\xff\xff'+p.read_bytes()[4:]),
      ('false-certification','benchmark_layouts/letter-register_16_tiled_end.indices',lambda p:b'\x7f\x00\x00\x00'+p.read_bytes()[4:]),
      ('negative-time','benchmark_layouts/rows.jsonl',lambda p:p.read_bytes().replace(b'"ns": ',b'"ns": -',1)),
      ('reordered-row','benchmark/rows.jsonl',lambda p:b'\n'.join([p.read_bytes().splitlines()[1],p.read_bytes().splitlines()[0]]+p.read_bytes().splitlines()[2:])+b'\n'),
      ('blank-bindings','benchmark_layouts/PROTOCOL.json',lambda p:changejson(p,lambda x:x.update(source={}))),
      ('missing-official','benchmark_layouts/PROTOCOL.json',lambda p:changejson(p,lambda x:x.update(official={}))),
      ('missing-candidate','benchmark_layouts/PROTOCOL.json',lambda p:changejson(p,lambda x:x.update(new_libraries={}))),
      ('duplicate-json','benchmark_layouts/PROTOCOL.json',lambda p:p.read_bytes().replace(b'"seed":',b'"seed":0,"seed":',1)),
      ('nonfinite-json','benchmark_layouts/rows.jsonl',lambda p:p.read_bytes().replace(b'"ns": ',b'"ns": 1e999, "ignored": ',1)),
      ('false-median','benchmark_layouts/SUMMARY.json',lambda p:changejson(p,lambda x:x['tasks']['letter']['us_per_row']['32'].update(register_16_official=.001))),
      ('changed-packed','compiled/letter-16.sct',lambda p:p.read_bytes()[:-1]+bytes([p.read_bytes()[-1]^1])),
      ('changed-source','frozen_models/letter/model.json',lambda p:p.read_bytes()+b' '),
      ('wrong-opening','evaluation/OPENING.json',lambda p:changejson(p,lambda x:x.update(model_lock_sha256='0'*64))),
      ('forged-quality','evaluation/QUALITY.json',lambda p:changejson(p,lambda x:x['letter'].update(correct=4000))),
      ('changed-library','native-register-final/trees.so',lambda p:p.read_bytes()[:-1]+bytes([p.read_bytes()[-1]^1]))]
    records=[]
    audit(copy,source)
    for name,path,transform in cases:
        p=copy/path;old=p.read_bytes()
        try:
            new=transform(p);assert new!=old;p.write_bytes(new)
            try:audit(copy,source)
            except ValueError as e:records.append({'case':name,'rejected':True,'reason':str(e)})
            else:raise AssertionError('corruption accepted: '+name)
        finally:p.write_bytes(old)
    result={'status':'PASS','rejected':len(records),'cases':records,'auditor_scope':'copied evidence corruption, not empirical model tests'}
    (out/'RESULT.json').write_text(json.dumps(result,indent=2));shutil.rmtree(copy);return result
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','source','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.root,a.source,a.out),indent=2))
