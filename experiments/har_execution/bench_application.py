"""Parameterized reproduction of the recorded process benchmark; no model tuning."""
from pathlib import Path
import hashlib,json,subprocess,time,sys,random,os,statistics
import argparse
parser=argparse.ArgumentParser(description='Run the fixed complete-file application comparison in a fresh study root.')
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[2])
args=parser.parse_args();root=args.root.resolve();source=args.source.resolve();out=root/'application';out.mkdir(exist_ok=False)
import numpy as np
import sysconfig
x=np.loadtxt(root/'inputs/har/test/X_test.txt');inp=out/'features.jsonl'
with inp.open('xb') as f:
    for row in x:f.write((json.dumps(row.tolist(),separators=(',',':'))+'\n').encode())
params={
'native_linear':(root/'linear-export',root/'linear-build/libspectra_linear.so'),
'native_rbf':(root/'models/rbf_c10/model.srt',root/'build-baseline-avx2/libspectra_svm.so'),
'sklearn_linear':(root/'models/linear_c1/model.joblib',None),
'numpy_linear':(root/'linear-export',None),
'sklearn_rbf':(root/'models/rbf_c10/model.joblib',None)}
files=[source/'experiments/har_execution/application.py',source/'spectra/linear.py',inp]
(root/'application/LOCK.json').write_text(json.dumps({'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'rounds':3,'seed':4927,'scope':'post-test application check; frozen input/model; no tuning'},indent=2))
def prediction_file(name):
    candidates=list((root/'evaluation').glob('*pred*'))+list((root/'linear-evaluation').glob('*pred*'))
    return candidates
print(prediction_file(''))
rng=random.Random(4927);results=[]
for repeat in range(3):
    names=list(params);rng.shuffle(names)
    for arm in names:
        model,library=params[arm];dest=out/f'{arm}-{repeat}.jsonl'
        command=[sys.executable,'-I','-S',str(source/'experiments/har_execution/application.py'),'--source',str(source),'--backend',arm,'--model',str(model),'--input',str(inp),'--output',str(dest)]
        if library:command+=['--library',str(library)]
        if not arm.startswith('native'):command+=['--dependency-site',sysconfig.get_path('purelib')]
        env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        start=time.perf_counter_ns();p=subprocess.run(command,capture_output=True,text=True,env=env);cold=time.perf_counter_ns()-start
        receipt={'command':command,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'parent_complete_ns':cold}
        (out/f'{arm}-{repeat}-execution.json').write_text(json.dumps(receipt,indent=2))
        if p.returncode:raise RuntimeError(p.stderr)
        result=json.loads(p.stdout);result.update(repeat=repeat,parent_complete_ns=cold)
        expected=json.loads((root/'linear-evaluation/predictions.json').read_text()) if 'linear' in arm else json.loads((root/'evaluation/rbf_c10-predictions.json').read_text())
        all_records=[json.loads(line) for line in dest.read_text().splitlines()]
        assert [v['label'] for v in all_records[1:-1]]==expected
        assert all_records[-1]['input_sha256']==hashlib.sha256(inp.read_bytes()).hexdigest()
        result['predictions_verified']=len(expected);results.append(result)
        print(arm,repeat,'cold',cold/1e6,'ms','inner',result['processing_ns']/1e6,'ms','RSS',result['memory_kib'],flush=True)
(out/'RESULTS.json').write_text(json.dumps(results,indent=2))
summary={arm:{'median_cold_ms':statistics.median(r['parent_complete_ns']/1e6 for r in results if r['backend']==arm),'median_processing_ms':statistics.median(r['processing_ns']/1e6 for r in results if r['backend']==arm),'peak_kib_range':[min(r['memory_kib']['VmHWM'] for r in results if r['backend']==arm),max(r['memory_kib']['VmHWM'] for r in results if r['backend']==arm)]} for arm in params}
(out/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
