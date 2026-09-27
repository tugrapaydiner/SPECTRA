"""Independent standard-library HAR record/identity audit; not timer attestation.

Consumes original UCI labels and retained predictions; never imports the runtime,
fitting library or benchmark aggregation functions. Maps recorded absolute paths
under the supplied evidence root. Separate native replay supplies numerical proof
under its declared computed-arithmetic contract, not this record auditor.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics


def unique(pairs):
    out={}
    for k,v in pairs:
        if k in out:raise ValueError('duplicate key')
        out[k]=v
    return out

def reject(x):raise ValueError('nonfinite token')
def read(path):return json.loads(Path(path).read_text(),object_pairs_hook=unique,parse_constant=reject)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def records(path):return [json.loads(s,object_pairs_hook=unique,parse_constant=reject) for s in Path(path).read_text().splitlines()]
def require(condition,message):
    if not condition:raise ValueError(message)

def audit(root,source):
    root,source=Path(root),Path(source);data=root/'inputs/har';evaluation=root/'evaluation'
    acquisition=read(data/'ACQUISITION.json')
    for name,item in acquisition['files'].items():
        require('..' not in Path(name).parts and not Path(name).is_absolute(),'data path')
        require(sha(data/name)==item['sha256'] and (data/name).stat().st_size==item['bytes'],'data identity')
    truth=[int(s) for s in (data/'test/y_test.txt').read_text().split()]
    subjects=[int(s) for s in (data/'test/subject_test.txt').read_text().split()]
    training={int(s) for s in (data/'train/subject_train.txt').read_text().split()}
    require(len(truth)==len(subjects)==2947 and len(set(subjects))==9 and len(training)==21 and not training&set(subjects),'subject boundary')
    lock=read(evaluation/'EVALUATION_LOCK.json');freeze=read(root/'models/FROZEN_MODELS.json')
    require(sha(root/'models/FROZEN_MODELS.json')==lock['model_freeze_sha256'],'model freeze identity')
    require(sha(source/'experiments/har_execution/evaluate.py')==lock['source_sha256'],'evaluation source')
    for name,digest in lock['native_source'].items():require(sha(source/name)==digest,'original executable source')
    for item in freeze['models']:
        folder=root/'models'/item['name'];require(sha(folder/'model.joblib')==item['model_joblib_sha256'],'frozen model')
        if 'sha256' in item:require(sha(folder/'model.srt')==item['sha256'],'native model export')
    summaries=read(evaluation/'SUMMARY.json');quality=read(evaluation/'QUALITY.json');predictions={}
    indices=read(evaluation/'TEST_INVENTORY.json')['row_indices']
    require(indices==[int(i*2946/255) for i in range(256)],'fixed subset')
    for model in ('rbf_c1','rbf_c10','linear_c1'):
        pred=read(evaluation/f'{model}-predictions.json');predictions[model]=pred
        require(len(pred)==len(truth) and all(type(x) is int and 1<=x<=6 for x in pred),'prediction inventory')
        correct=sum(a==b for a,b in zip(pred,truth));require(correct==quality[model]['correct']==summaries['models'][model]['correct'],'accuracy arithmetic')
        for subject in set(subjects):
            ind=[i for i,s in enumerate(subjects) if s==subject]
            require(quality[model]['by_subject'][str(subject)]=={'correct':sum(pred[i]==truth[i] for i in ind),'total':len(ind)},'subject summary')
    rows=records(evaluation/'timings.jsonl');rng=random.Random(2026092719);expected=[]
    for model in ('rbf_c1','rbf_c10','linear_c1'):
        arms=['sklearn'] if model=='linear_c1' else ['sklearn','native_exhaustive','native_lazy','native_beretta_cert']
        for repeat in range(7):
            jobs=[(arm,b) for b in (1,32) for arm in arms];rng.shuffle(jobs)
            for arm,b in jobs:expected.append((model,arm,b,repeat))
    # Actual randomized order is source-bound; the complete unique grid is also
    # checked without trusting timing summary fields.
    require(len(rows)==126 and len({(r['model'],r['arm'],r['batch'],r['repeat']) for r in rows})==126,'timing inventory')
    require({(r['model'],r['arm'],r['batch'],r['repeat']) for r in rows}==set(expected),'timing grid')
    for r in rows:
        require(all(type(r[k]) is int for k in ('wall_ns','repeat','batch','rows')) and r['wall_ns']>0 and r['rows']==256,'timing value')
        wanted=[predictions[r['model']][i] for i in indices]
        digest=hashlib.sha256(json.dumps(wanted,separators=(',',':')).encode()).hexdigest()
        require(r['output_sha256']==digest,'timed labels')
    for model,item in summaries['models'].items():
        for batch,arms in item['latency_us_per_row'].items():
            for arm,value in arms.items():
                derived=statistics.median(r['wall_ns']/r['rows']/1000 for r in rows if r['model']==model and r['arm']==arm and r['batch']==int(batch))
                require(derived==value,'timing aggregate')
    linear=root/'linear-evaluation';lr=records(linear/'timings.jsonl');ls=read(linear/'SUMMARY.json')
    require(len(lr)==42 and {(r['arm'],r['batch'],r['repeat']) for r in lr}=={(a,b,j) for a in ('native','sklearn','numpy_specialized') for b in (1,32) for j in range(7)},'linear grid')
    digest=hashlib.sha256(json.dumps([predictions['linear_c1'][i] for i in indices]).encode()).hexdigest()
    for r in lr:require(all(type(r[k]) is int for k in ('ns','repeat','batch','rows')) and r['ns']>0 and r['rows']==256 and r['labels_sha256']==digest,'linear timing value/output')
    for key,value in ls.items():
        arm,b=key.split('/batch');derived=statistics.median(r['ns']/256/1000 for r in lr if r['arm']==arm and r['batch']==int(b))
        require(derived==value['median_us_per_row'],'linear aggregate')
    for name,digest in read(linear/'LINEAR_LOCK.json')['files'].items():
        path=Path(name)
        if path.is_absolute():
            mapped=source/path.relative_to('/mnt/data/spectra_next/repo')
        else:
            require(path.parts[0]=='..' and '..' not in path.parts[1:],'relative artifact path')
            mapped=root/Path(*path.parts[1:])
        require(sha(mapped)==digest,'linear bound source/model')
    app=root/'application';ar=read(app/'RESULTS.json')
    require(len(ar)==15 and {(r['backend'],r['repeat']) for r in ar}=={(a,j) for a in ('native_linear','native_rbf','sklearn_linear','sklearn_rbf','numpy_linear') for j in range(3)},'application inventory')
    for r in ar:
        raw=(app/f"{r['backend']}-{r['repeat']}.jsonl").read_bytes();parts=raw.splitlines(keepends=True);docs=[json.loads(s) for s in parts]
        pred=predictions['linear_c1' if 'linear' in r['backend'] else 'rbf_c10']
        require([d['label'] for d in docs[1:-1]]==pred,'application labels')
        require([d['index'] for d in docs[1:-1]]==list(range(2947)),'application order')
        require(docs[-1]['rows']==2947 and docs[-1]['input_sha256']==sha(app/'features.jsonl'),'application input/completion')
        require(docs[-1]['output_prefix_sha256']==hashlib.sha256(b''.join(parts[:-1])).hexdigest(),'application output integrity')
        execution=read(app/f"{r['backend']}-{r['repeat']}-execution.json")
        require(execution['returncode']==0 and execution['parent_complete_ns']==r['parent_complete_ns'],'application execution receipt')
    return {'status':'PASS','held_out_subjects':9,'model_test_pairs':8841,'timing_cells':168,'application_runs':15,'application_predictions':15*2947,
       'scope':'record identities, grid completeness, original labels and arithmetic; not independently observed timers or outside reproduction'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','source','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();value=audit(a.root,a.source)
    with a.out.open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps(value,indent=2))
