"""Verify corruption rejection only on private evidence copies, never originals."""
from pathlib import Path
import argparse,json,shutil,sys,zipfile
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.budgeted_prototypes.audit import audit

def run(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    copy=out/'copy';copy.mkdir()
    names=['selection','selection_optical','selection_final','models','evaluation','source_snapshots','benchmark','benchmark_strong','benchmark_final','models_fp32','float-native','native','native-packet','native-register','controls-native','blas-native','finite-native']
    for n in names:shutil.copytree(root/n,copy/n)
    for n in ['prior/datasets','prior/satellite/data','new_data']:shutil.copytree(root/n,copy/n)
    audit(copy)
    def json_change(p,f):
        x=json.loads(p.read_text());f(x);return json.dumps(x).encode()
    cases=[
      ('selection-choice','selection_final/selected.json',lambda p:json_change(p,lambda x:x['letter']['local']['choice'].update(prototypes=2))),
      ('selection-count','selection/job-000/record.json',lambda p:json_change(p,lambda x:x.update(correct=99999))),
      ('amendment-count','selection_optical/job-000/record.json',lambda p:json_change(p,lambda x:x.update(correct=99999))),
      ('model-lock','models/FINAL_LOCK.json',lambda p:json_change(p,lambda x:x.update(models=23))),
      ('model-bytes','models/letter-local/model.spp',lambda p:p.read_bytes()[:-1]+bytes([p.read_bytes()[-1]^1])),
      ('input','evaluation/letter/input.u8',lambda p:bytes([p.read_bytes()[0]^1])+p.read_bytes()[1:]),
      ('quality','evaluation/quality.json',lambda p:json_change(p,lambda x:x['optdigits']['arms']['local'].update(correct=1797))),
      ('opening','evaluation/TEST_OPENING.json',lambda p:json_change(p,lambda x:x.update(final_lock_sha256='0'*64))),
      ('new-official-data','new_data/optdigits.zip',lambda p:p.read_bytes()[:-1]+bytes([p.read_bytes()[-1]^1])),
      ('source','source_snapshots/strong_benchmark/runtime.cpp',lambda p:p.read_bytes()+b'\n// mutation\n'),
      ('missing-timing','benchmark_strong/timings.jsonl',lambda p:b'\n'.join(p.read_bytes().splitlines()[:-1])+b'\n'),
      ('timing-digest','benchmark_strong/timings.jsonl',lambda p:p.read_bytes().replace(b'"prediction_sha256": "',b'"prediction_sha256": "f',1)),
      ('negative-time','benchmark_strong/timings.jsonl',lambda p:p.read_bytes().replace(b'"ns": ',b'"ns": -',1)),
      ('promoted-failure','benchmark_strong/summary.json',lambda p:json_change(p,lambda x:x.update(primary_two_task_gate=True))),
      ('layout-claim','benchmark_strong/summary.json',lambda p:json_change(p,lambda x:x.update(layout_ratio=.1))),
      ('native-binary','native-register/prototypes.so',lambda p:p.read_bytes()[:-1]+bytes([p.read_bytes()[-1]^1]))]
    observations=[]
    for name,file,change in cases:
        p=copy/file;raw=p.read_bytes()
        try:
            changed=change(p)
            if changed==raw:raise AssertionError('corruption is not a change')
            p.write_bytes(changed)
            try:audit(copy)
            except (ValueError,zipfile.BadZipFile) as e:observations.append({'case':name,'rejected':True,'reason':str(e)})
            else:raise AssertionError('accepted altered receipt '+name)
        finally:p.write_bytes(raw)
    result={'status':'PASS','corrupted_copies_rejected':len(observations),'cases':observations,'scope':'adversarial receipt tests, not new classifier examples'}
    (out/'report.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.root,a.out),indent=2))
