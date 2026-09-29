"""Alter only disposable copies, require rejection by the recorded-data auditor."""
from __future__ import annotations
import argparse,json,shutil,tempfile,zipfile
from pathlib import Path
from .audit import audit,sha

def run(root,out):
    root=Path(root);out=Path(out)
    if out.exists():raise FileExistsError(out)
    audit(root)
    def change_json(path,fn):
        d=json.loads(path.read_text());fn(d);return json.dumps(d).encode()
    cases=[
        ('model-bytes','refinement_fit/letter/fast/model.spp',lambda p:p.read_bytes()[:-1]+bytes([p.read_bytes()[-1]^1])),
        ('selected-setting','refinement_fit/letter/SELECTION.json',lambda p:change_json(p,lambda x:x['fast']['choice'].update(gamma=999))),
        ('missing-training-row','inputs/letter-train.npz',lambda p:p.read_bytes()[:-1]),
        ('calibration-threshold','refinement_calibration/POLICIES.json',lambda p:change_json(p,lambda x:x['letter']['0.01'].update(threshold=0))),
        ('calibration-bound','refinement_calibration/letter/calibration.json',lambda p:change_json(p,lambda x:x['rows'][0].update(upper_added_harm=0))),
        ('missing-score','refinement_evaluation/letter/predictions.npz',lambda p:p.read_bytes()[:-1]),
        ('test-identity','refinement_evaluation/letter/input.u8',lambda p:bytes([p.read_bytes()[0]^1])+p.read_bytes()[1:]),
        ('missing-time','refinement_benchmark/timings.jsonl',lambda p:b'\n'.join(p.read_bytes().splitlines()[:-1])+b'\n'),
        ('wrong-output','refinement_benchmark/timings.jsonl',lambda p:p.read_bytes().replace(b'"prediction_sha256": "',b'"prediction_sha256": "f',1)),
        ('negative-time','refinement_benchmark/timings.jsonl',lambda p:p.read_bytes().replace(b'"ns": ',b'"ns": -',1)),
        ('duplicate-field','refinement_benchmark/timings.jsonl',lambda p:p.read_bytes().replace(b'"ns": ',b'"ns": 1,"ns": ',1)),
        ('invented-promotion','refinement_benchmark/RESULTS.json',lambda p:change_json(p,lambda x:x.update(two_task_gate=True))),
        ('invented-speed','refinement_benchmark/RESULTS.json',lambda p:change_json(p,lambda x:x['tasks']['letter'].update(primary_over_strong=.0001))),
        ('erased-failure','refinement_benchmark/RESULTS.json',lambda p:change_json(p,lambda x:x['tasks']['pendigits'].update(loss_against_strong_points=0))),
        ('native-bytes','refinement-native-final/refinement.so',lambda p:p.read_bytes()[:-1]+bytes([p.read_bytes()[-1]^1])),
        ('timed-source','refinement_benchmark/source/experiments/selective_refinement/bench.py',lambda p:p.read_bytes()+b'\n# altered copy\n')]
    records=[]
    with tempfile.TemporaryDirectory(prefix='spectra-audit-negative-') as tmp:
        copy=Path(tmp)
        for name in ('inputs','refinement_fit','refinement_calibration','refinement_evaluation','refinement_benchmark','refinement-native-final','mlp-native'):
            shutil.copytree(root/name,copy/name)
        audit(copy)
        for name,relative,mutation in cases:
            p=copy/relative;old=p.read_bytes()
            try:
                new=mutation(p)
                if new==old:raise AssertionError('test did not mutate bytes')
                p.write_bytes(new)
                try:audit(copy)
                except (ValueError,KeyError,EOFError,zipfile.BadZipFile) as error:
                    records.append({'case':name,'rejected':True,'error_type':type(error).__name__,'reason':str(error)})
                else:raise AssertionError('corruption was accepted: '+name)
            finally:p.write_bytes(old)
        audit(copy)
    result={'status':'PASS','corrupted_copies_rejected':len(records),'cases':records,
            'auditor_sha256':sha(Path(__file__).with_name('audit.py')),
            'scope':'disposable-record corruption tests, not new accuracy evidence or authenticated clocks'}
    with out.open('x') as f:json.dump(result,f,indent=2)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.root,a.out),indent=2))
