"""Corrupt disposable COPIES of completed receipts; require explicit rejection.

These are auditor tests, not new model observations. Original evidence is never
modified. The test directory must be fresh and is retained for inspection.
"""
from __future__ import annotations
import argparse,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.learned_metric.audit import audit as audit_primary
from experiments.learned_metric.audit_satellite import audit as audit_satellite


def main(a):
 a.work.mkdir(parents=True,exist_ok=False)
 copied=a.work/'evidence';shutil.copytree(a.root,copied)
 audit_primary(copied,a.source);audit_satellite(copied/'satellite',a.source)
 def alter_json(path,fn):
  obj=json.loads(path.read_text());fn(obj);return json.dumps(obj).encode()
 def line_change(path,fn):
  rows=path.read_text().splitlines();fn(rows);return ('\n'.join(rows)+'\n').encode()
 cases=[
 ('primary-missing-selection','selection/selection.jsonl',lambda p:line_change(p,lambda r:r.pop()),False),
 ('primary-duplicate-selection','selection/selection.jsonl',lambda p:line_change(p,lambda r:r.append(r[0])),False),
 ('primary-wrong-selection','selection/selected.json',lambda p:alter_json(p,lambda o:o['letter']['nca'].update(C=999.)),False),
 ('primary-wrong-quality','evaluation/quality.json',lambda p:alter_json(p,lambda o:o['letter']['nca'].update(correct=4000)),False),
 ('primary-truncated-timing','benchmark/rows.jsonl',lambda p:line_change(p,lambda r:r.pop()),False),
 ('primary-output-digest','benchmark/rows.jsonl',lambda p:p.read_bytes().replace(b'"prediction_sha256": "',b'"prediction_sha256": "0',1),False),
 ('primary-forged-promotion','benchmark/summary.json',lambda p:alter_json(p,lambda o:o['tasks']['letter'].update(joint_gate=True)),False),
 ('primary-changed-source-binding','benchmark/protocol.json',lambda p:alter_json(p,lambda o:o['source'].update({'experiments/learned_metric/runtime.cpp':'0'*64})),False),
 ('primary-changed-model','models/letter-nca/model.sgm',lambda p:p.read_bytes()[:-1]+bytes([p.read_bytes()[-1]^1]),False),
 ('satellite-missing-selection','satellite/selection/selection.jsonl',lambda p:line_change(p,lambda r:r.pop()),True),
 ('satellite-wrong-selection','satellite/selection/selected.json',lambda p:alter_json(p,lambda o:o['nca'].update(gamma=99.)),True),
 ('satellite-wrong-quality','satellite/evaluation/quality.json',lambda p:alter_json(p,lambda o:o['nca'].update(correct=2000)),True),
 ('satellite-truncated-timing','satellite/evaluation/timings.jsonl',lambda p:line_change(p,lambda r:r.pop()),True),
 ('satellite-output-digest','satellite/evaluation/timings.jsonl',lambda p:p.read_bytes().replace(b'"prediction_sha256": "',b'"prediction_sha256": "0',1),True),
 ('satellite-changed-model','satellite/models/nca/model.sgm',lambda p:p.read_bytes()[:-1]+bytes([p.read_bytes()[-1]^1]),True),
 ('satellite-changed-official-data','satellite/acquisition/satellite.zip',lambda p:p.read_bytes()[:-1]+bytes([p.read_bytes()[-1]^1]),True)]
 observations=[]
 for name,relative,change,satellite in cases:
  path=copied/relative;original=path.read_bytes()
  try:
   changed=change(path)
   if changed==original:raise AssertionError('mutation did not alter bytes')
   path.write_bytes(changed)
   try:
    (audit_satellite(copied/'satellite',a.source) if satellite else audit_primary(copied,a.source))
   except ValueError as error:
    observations.append({'case':name,'rejected':True,'reason':str(error)})
   else:raise AssertionError('corruption accepted: '+name)
  finally:path.write_bytes(original)
 report={'status':'PASS','negative_cases':len(observations),'cases':observations,'scope':'deliberate copied-receipt corruption, not new task data or clock authentication'}
 with a.out.open('x') as f:json.dump(report,f,indent=2)
 print(json.dumps(report,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('root','source','work','out'):p.add_argument('--'+key,type=Path,required=True)
 main(p.parse_args())
