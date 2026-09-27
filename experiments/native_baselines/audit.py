"""Independent standard-library audit of frozen native-comparison evidence.

Checks source/model/binary/input identities, every timing cell and output digest,
and recalculates medians without importing benchmark/native/framework code.
It does not authenticate a maliciously rewritten complete packet or a timer.
"""
from __future__ import annotations
import argparse,hashlib,json,math,random,statistics,struct,zlib
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unique(pairs):
    r={}
    for k,v in pairs:
        if k in r:raise ValueError('duplicate key')
        r[k]=v
    return r

def reject(s):raise ValueError('nonfinite JSON token')
def read(p):return json.loads(Path(p).read_text(),object_pairs_hook=unique,parse_constant=reject)
def digest(v):return hashlib.sha256(json.dumps(v,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
def equal(a,b):
    if type(a)!=type(b):raise ValueError('type mismatch')
    if type(a) is dict:
        if a.keys()!=b.keys():raise ValueError('keys mismatch')
        for k in a:equal(a[k],b[k])
    elif type(a) is list:
        if len(a)!=len(b):raise ValueError('length mismatch')
        for x,y in zip(a,b):equal(x,y)
    elif type(a) is float:
        if not math.isfinite(a) or not math.isclose(a,b,rel_tol=1e-13,abs_tol=1e-13):raise ValueError('aggregate mismatch')
    elif a!=b:raise ValueError('value mismatch')

def audit_export(srt,text):
    raw=srt.read_bytes()
    magic,c,n,d,ml,size,crc=struct.unpack_from('<8sIIIIII',raw)
    if magic!=b'SPCSVM02' or zlib.crc32(raw[32:])!=crc or len(raw)!=32+ml+size:raise ValueError('model structure')
    offset=32+ml;gamma=struct.unpack_from('<d',raw,offset)[0];offset+=8
    counts=struct.unpack_from('<'+'I'*c,raw,offset);offset+=4*c
    values=struct.unpack_from('<'+'d'*((size-8-4*c)//8),raw,offset)
    sv=values[:n*d];co=values[n*d:n*d+(c-1)*n];bias=values[n*d+(c-1)*n:]
    lines=text.read_text().splitlines();at=lines.index('SV')
    h={s.split()[0]:s.split()[1:] for s in lines[:at]}
    if h['svm_type']!=['c_svc'] or h['kernel_type']!=['rbf'] or float(h['gamma'][0])!=gamma:raise ValueError('kernel export')
    if list(map(int,h['label']))!=list(range(c)) or tuple(map(int,h['nr_sv']))!=counts or int(h['total_sv'][0])!=n:raise ValueError('class export')
    sign=-1 if c==2 else 1
    if list(map(float,h['rho']))!=[-sign*v for v in bias]:raise ValueError('bias export')
    if len(lines[at+1:])!=n:raise ValueError('support export count')
    for i,line in enumerate(lines[at+1:]):
        row=line.split()
        if list(map(float,row[:c-1]))!=[sign*co[j*n+i] for j in range(c-1)]:raise ValueError('coefficient export')
        mapped={int(x.split(':')[0])-1:float(x.split(':')[1]) for x in row[c-1:]}
        if list(mapped)!=sorted(mapped) or any(k<0 or k>=d for k in mapped):raise ValueError('node order')
        for j in range(d):
            if mapped.get(j,0.)!=sv[i*d+j]:raise ValueError('support value export')
    return {'features':d,'classes':c,'supports':n,'export':'EXACT_VALUES'}

def audit(root,source):
    root=Path(root);source=Path(source);run=root/'benchmark';models=root/'models';build=root/'builds-final';valid=root/'validation'
    p=read(run/'protocol.json');freeze=read(models/'MODEL_FREEZE.json');validation=read(valid/'VALIDATION.json');builds=read(build/'builds.json')
    if p['seed']!=2026092709 or p['repeats']!=7 or p['chunks']!=[1,32,256]:raise ValueError('protocol change')
    if p['models']!=['wine-101','wdbc-101','chess-101','penguins-101','titanic-101','zoo-101','har']:raise ValueError('panel change')
    for actual,wanted in [(models/'MODEL_FREEZE.json',p['freeze_sha256']),(valid/'VALIDATION.json',p['validation_sha256']),(build/'builds.json',p['builds_sha256'])]:
        if sha(actual)!=wanted:raise ValueError('receipt identity')
    for name,wanted in p['source_sha256'].items():
        q=source/name
        if '..' in Path(name).parts or q.is_symlink() or sha(q)!=wanted:raise ValueError('source identity')
    for name,wanted in p['libraries_sha256'].items():
        if sha(build/name)!=wanted:raise ValueError('binary identity')
    rng=random.Random(2026092709);schedule=[];exports={};outputs={}
    for name,m in freeze['models'].items():
        for path,d in m['files'].items():
            if sha(models/name/path)!=d:raise ValueError('frozen model identity')
        if sha(models/name/'X.f64')!=p['inputs_sha256'][name]:raise ValueError('input identity')
        exports[name]=audit_export(models/name/'model.srt',models/name/'libsvm.model')
        arms=['libsvm']
        if builds['generated'][name]['status']=='PASS':arms+=['m2cgen']
        for target in ('portable','avx2'):
            arms += [f'spectra-{target}-{mode}' for mode in ('exhaustive','beretta_cert','binary_stream')]
        if name=='har':arms+=['linear_native']
        base=read(valid/f'{name}-sklearn.json')
        for arm in arms:
            fn=f'{name}-linear.json' if arm=='linear_native' else f'{name}-{arm}.json'
            labels=read(valid/fn);outputs[name,arm]=digest(labels)
            if len(labels)!=m['rows']:raise ValueError('output count')
            if arm!='linear_native' and sum(x!=y for x,y in zip(labels,base))!=validation['models'][name]['backends'][arm]['disagreements']:raise ValueError('disagreement summary')
        for repeat in range(7):
            jobs=[(chunk,arm) for chunk in (1,32,256) for arm in arms];rng.shuffle(jobs)
            for chunk,arm in jobs:schedule.append((name,repeat,chunk,arm,m['rows'],outputs[name,arm]))
    records=[json.loads(x,object_pairs_hook=unique,parse_constant=reject) for x in (run/'rows.jsonl').read_text().splitlines()]
    if len(records)!=len(schedule):raise ValueError('grid length')
    groups={}
    for r,e in zip(records,schedule):
        if set(r)!={'model','repeat','chunk','arm','rows','ns','output_sha256'}:raise ValueError('row schema')
        if tuple(r[k] for k in ('model','repeat','chunk','arm','rows','output_sha256'))!=e:raise ValueError('row order or output')
        if any(type(r[k]) is not int for k in ('repeat','chunk','rows','ns')) or r['ns']<=0:raise ValueError('row number')
        groups.setdefault((r['model'],r['arm'],r['chunk']),[]).append(r['ns'])
    result={}
    for name,m in freeze['models'].items():
        arms=sorted({a for n,a,c in groups if n==name});batch={}
        for chunk in (1,32,256):
            cost={a:statistics.median(groups[name,a,chunk]) for a in arms}
            batch[str(chunk)]={'job_median_ns':cost,'us_per_row':{a:v/m['rows']/1000 for a,v in cost.items()}}
        result[name]={'rows':m['rows'],'features':m['features'],'supports':m['supports'],'batches':batch}
    h=result['har']['batches']['32']['us_per_row'];s=h['spectra-avx2-beretta_cert'];complete=builds['generated']['har']['status']=='PASS'
    expected={'format':'spectra.native_comparison.v1','cells':len(records),'repeated_predictions':sum(r['rows'] for r in records),
       'models':result,'har_default_avx2_over_libsvm':s/h['libsvm'],'har_linear_speedup_over_default_avx2':s/h['linear_native'],
       'har_generated_baseline_complete':complete,'native_admission':'INCOMPLETE' if not complete else ('PASS' if s<=min(h['libsvm'],h['m2cgen'])/1.2 else 'FAIL'),
       'task_admission':'FAIL' if validation['models']['har']['linear_accuracy']>=validation['models']['har']['svc_accuracy'] and h['linear_native']<s else 'UNDETERMINED',
       'scope':'internal native API complete jobs; no process-startup or raw-sensor preprocessing advantage'}
    equal(read(run/'summary.json'),expected)
    return {'status':'PASS','timing_cells':len(records),'repeated_predictions':expected['repeated_predictions'],
            'exact_model_exports':exports,'source_files':len(p['source_sha256']),'native_admission':expected['native_admission'],'task_admission':expected['task_admission']}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=audit(a.root,a.source)
    with a.out.open('x') as f:json.dump(r,f,indent=2)
    print(json.dumps(r,indent=2))
