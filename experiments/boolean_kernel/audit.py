"""Independent standard-library audit of the fixed seven-model timing matrix.

Requires original evidence identities, all binary/source bindings and every planned
measurement. This checks records and arithmetic, not clock integrity or outsiders'
reproduction. It deliberately does not import the benchmark or inference runtime.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

MODELS=('wine-101','wdbc-101','chess-101','penguins-101','titanic-101','zoo-101','har')
ROWS=(45,143,799,86,328,26,2947)
CHUNKS=(1,32,256)
SEED=20260928
REPEATS=11
ANCHOR='95f6a37315746502eed1a165e17e89e9dbd0c472733c89537ce894f58e446a00'
BASE='c9f8bb6a2ca30d6840c69b463fd6dc1ea046ad6f'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def unique(pairs):
    out={}
    for k,v in pairs:
        if k in out: raise ValueError('duplicate JSON key')
        out[k]=v
    return out

def reject(_): raise ValueError('nonfinite JSON')
def load(path): return json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=unique,parse_constant=reject)
def label_hash(labels): return hashlib.sha256(json.dumps(labels,separators=(',',':')).encode()).hexdigest()
def member(root,name):
    path=Path(root)
    if type(name) is not str or '\\' in name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('unsafe inventory path')
    for part in Path(name).parts:
        path=path/part
        if path.is_symlink(): raise ValueError('linked inventory path')
    if not path.is_file(): raise ValueError('missing inventory path')
    return path

def audit(run,models,source,controls,library):
    run,models,source,controls,library=map(Path,(run,models,source,controls,library))
    manifest=models.parent/'SHA256.json'
    if sha(manifest)!=ANCHOR: raise ValueError('original manifest identity differs')
    anchor=load(manifest)
    required={f'{m}-protocol.json' for m in MODELS}|{f'{m}-{r:02d}.json' for m in MODELS for r in range(REPEATS)}
    if {p.name for p in run.iterdir()}!=required: raise ValueError('incomplete or extra run inventory')
    bindings={p.name:sha(p) for p in [library,*sorted(controls.glob('*.so'))] if not p.name.startswith('observe')}
    sources={p.relative_to(source).as_posix() for p in (source/'spectra').rglob('*.py')}
    sources|={p.relative_to(source).as_posix() for suffix in ('*.hpp','*.cpp') for p in (source/'spectra/_native').rglob(suffix)}
    sources|={'experiments/boolean_kernel/bench.py','experiments/boolean_kernel/PROTOCOL.md','experiments/native_baselines/runtime.py'}
    all_rows=[]; medians={}; regressions={}
    for ordinal,(model,n) in enumerate(zip(MODELS,ROWS)):
        arms=['parent_default','parent_stream','off_beretta_cert','off_exhaustive','packed_beretta_cert','packed_exhaustive','lookup_beretta_cert','lookup_exhaustive','libsvm', 'linear_model' if model=='har' else 'generated_c']
        p=load(run/f'{model}-protocol.json')
        if (p['format']!='spectra.boolean_kernel.bench.v1' or p['model']!=model or p['models']!=list(MODELS)
            or type(p['seed']) is not int or p['seed']!=SEED or type(p['repeats']) is not int or p['repeats']!=REPEATS
            or p['chunks']!=list(CHUNKS) or p['arms']!=arms or p['base_commit']!=BASE):
            raise ValueError('protocol differs')
        if set(p['source_sha256'])!=sources: raise ValueError('source inventory differs')
        for name,digest in p['source_sha256'].items():
            if sha(member(source,name))!=digest: raise ValueError('source bytes differ: '+name)
        if p['libraries']!=bindings: raise ValueError('binary identity differs')
        if set(p['bindings'])!={'model.srt','X.f64','expected.json','libsvm.model'}: raise ValueError('model inventory differs')
        for filename,digest in p['bindings'].items():
            fp=member(models,f'{model}/{filename}'); original=anchor[f'models/{model}/{filename}']
            if sha(fp)!=digest or digest!=original['sha256'] or fp.stat().st_size!=original['bytes']:
                raise ValueError('model/input identity differs')
        labels=load(models/model/'expected.json')
        if len(labels)!=n: raise ValueError('expected prediction inventory differs')
        expected={arm:label_hash(labels) for arm in arms}
        if model=='har':
            rel='runs/matched/validation/har-linear.json'; fp=member(models.parent,rel)
            if sha(fp)!=anchor[rel]['sha256']: raise ValueError('linear prediction identity differs')
            linear=load(fp)
            if len(linear)!=n: raise ValueError('linear prediction count differs')
            expected['linear_model']=label_hash(linear)
        groups={}; repeats={}
        for r in range(REPEATS):
            planned=[(chunk,arm) for chunk in CHUNKS for arm in arms]
            random.Random(SEED+100*ordinal+r).shuffle(planned)
            rows=load(run/f'{model}-{r:02d}.json')
            if type(rows) is not list or len(rows)!=len(planned): raise ValueError('timing inventory differs')
            for row,(chunk,arm) in zip(rows,planned):
                correct={'model':model,'repeat':r,'chunk':chunk,'arm':arm,'rows':n,'output_sha256':expected[arm]}
                if type(row) is not dict or set(row)!=set(correct)|{'ns'}: raise ValueError('timing schema differs')
                for k,v in correct.items():
                    if type(row[k]) is not type(v) or row[k]!=v: raise ValueError('timing order/output differs')
                if type(row['ns']) is not int or row['ns']<=0: raise ValueError('invalid time')
                groups.setdefault((chunk,arm),[]).append(row['ns']); repeats[r,chunk,arm]=row['ns']; all_rows.append(row)
        medians[model]={str(c):{a:statistics.median(groups[c,a])/n/1000 for a in arms} for c in CHUNKS}
        regressions[model]={str(c):{reference:sum(repeats[r,c,'lookup_beretta_cert']>repeats[r,c,reference] for r in range(REPEATS)) for reference in ['parent_default','off_beretta_cert',('linear_model' if model=='har' else 'generated_c')]} for c in CHUNKS}
    c=medians['chess-101']['32']
    ratios={ref:c['lookup_beretta_cert']/c[ref] for ref in c if ref!='lookup_beretta_cert'}
    numerical = {'expected_original_pairs':sum(ROWS),'note':'fidelity is checked by a separate original-source observer, not inferred from timing hashes'}
    return {'status':'PASS','cells':len(all_rows),'repeated_predictions':sum(r['rows'] for r in all_rows),
        'source_files':len(sources),'model_input_pairs':sum(ROWS),'medians_us_per_row':medians,
        'paired_repeat_regressions':regressions,'chess_batch32_lookup_over':ratios,
        'primary_timing_gate':all(ratios[k]<=1/1.2 for k in ('off_beretta_cert','generated_c')),
        'numerical_scope':numerical,'scope':'recorded native complete-job latency; no fresh accuracy/production/external-replication claim'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('run','models','source','controls','library','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();result=audit(a.run,a.models,a.source,a.controls,a.library)
    with a.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('medians_us_per_row','paired_repeat_regressions')},indent=2))
