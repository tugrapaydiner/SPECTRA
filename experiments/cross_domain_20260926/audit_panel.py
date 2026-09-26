"""Independent standard-library audit of the frozen cross-domain receipts.

Imports no SPECTRA, NumPy, sklearn, training or timing implementation. This checks
source/model binding, exported coefficients, observations, vote certificates and
arithmetic summaries; it is not independent external replication.
"""
from __future__ import annotations
import argparse
import ast
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import struct
import sys
import zipfile

TASKS=('wine','vehicle','satellite','har','sensorless')
ARMS=('exhaustive_direct','cert_direct','exhaustive_tables','cert_tables','knockout_direct','libsvm')
PUBLIC=('spectra','libsvm','sklearn','logistic','mlp')
SEED=20260926


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read_npy(raw):
    if len(raw)<10 or raw[:6]!=b'\x93NUMPY' or raw[6] not in (1,2,3):raise ValueError('not a supported NPY')
    size=2 if raw[6]==1 else 4;offset=8+size
    length=int.from_bytes(raw[8:offset],'little');meta=ast.literal_eval(raw[offset:offset+length].decode())
    if set(meta)!={'descr','fortran_order','shape'} or meta['fortran_order']:raise ValueError('NPY metadata')
    shape=meta['shape'];dtype=meta['descr'];body=raw[offset+length:]
    if not isinstance(shape,tuple) or any(type(n)is not int or n<0 for n in shape):raise ValueError('NPY shape')
    codes={'<f8':('d',8),'<i8':('q',8),'<i4':('i',4),'<u4':('I',4),'|b1':('B',1)}
    if dtype in codes:
        code,width=codes[dtype]
        if sys.byteorder!='little':raise ValueError('audit host must be little-endian')
        if len(body)!=math.prod(shape)*width:raise ValueError('NPY byte inventory')
        values=memoryview(body).cast(code)
    elif dtype.startswith('<U'):
        width=int(dtype[2:])*4
        if width<4:raise ValueError('unicode width')
        values=[body[k:k+width].decode('utf-32-le').rstrip('\x00') for k in range(0,len(body),width)]
    else:raise ValueError('unsupported NPY dtype')
    if len(body)!=math.prod(shape)*width:raise ValueError('NPY byte inventory')
    return shape,values


def npz(path,name):
    with zipfile.ZipFile(path) as z:return read_npy(z.read(name+'.npy'))


def certificate(c,w,edges):
    if type(c)is not int or not 2<=c<=128 or type(w)is not int or not 0<=w<c or len(edges)!=c*(c-1)//2:return False
    wins=[0]*c;missing=[c-1]*c;k=0
    for a in range(c):
        for b in range(a+1,c):
            value=edges[k];k+=1
            if type(value)is not int or value not in (-1,a,b):return False
            if value!=-1:wins[value]+=1;missing[a]-=1;missing[b]-=1
    return all((wins[w]>wins[j]+missing[j]) if j<w else (wins[w]>=wins[j]+missing[j]) for j in range(c) if j!=w)


def check_rows(rows,positions,names,expected,label_mode=False):
    rng=random.Random(SEED);plan=[]
    for rep in range(11):
        order=list(positions);rng.shuffle(order)
        for i in order:
            modes=list(names);rng.shuffle(modes)
            plan.extend((rep,i,name) for name in modes)
    if len(rows)!=len(plan):raise ValueError('timing grid count')
    grouped=defaultdict(list)
    key='label' if label_mode else 'class_index'
    for row,cell in zip(rows,plan):
        if set(row)!={'repeat','case','arm','ns',key} or (row['repeat'],row['case'],row['arm'])!=cell:raise ValueError('timing schedule')
        if any(type(row[k])is not int for k in ('repeat','case','ns')) or row['ns']<=0:raise ValueError('timing scalar')
        rep,i,arm=cell
        if expected is not None and (not label_mode or arm in ('spectra','libsvm','sklearn')) and row[key]!=expected[i]:raise ValueError('timed output')
        grouped[arm,i].append(row['ns'])
    return grouped


def summarize(groups,positions,names):
    return {name:dict(mean_case_median_us=statistics.mean(statistics.median(groups[name,i]) for i in positions)/1000)
            for name in names}


def paired(groups,positions,candidate,baseline):
    a=[statistics.median(groups[candidate,i]) for i in positions]
    b=[statistics.median(groups[baseline,i]) for i in positions]
    rng=random.Random(SEED);ratios=[];n=len(a)
    for _ in range(2000):
        ids=rng.choices(range(n),k=n);ratios.append(sum(a[i] for i in ids)/sum(b[i] for i in ids))
    ratios.sort()
    def quantile(xs,q):
        t=(len(xs)-1)*q;l=int(t);h=min(l+1,len(xs)-1)
        return xs[l]+(xs[h]-xs[l])*(t-l)
    return dict(ratio=sum(a)/sum(b),speedup=sum(b)/sum(a),conditional_ci95=[quantile(ratios,.025),quantile(ratios,.975)],
                median_regressions=sum(x>y for x,y in zip(a,b)),
                empirical_p95_regressions=sum(quantile(sorted(groups[candidate,i]),.95)>quantile(sorted(groups[baseline,i]),.95) for i in positions))


def check_export(srt,text):
    raw=srt.read_bytes()
    magic,c,n,d,ls,sz,crc=struct.unpack('<8sIIIIII',raw[:32])
    if magic!=b'SPCSVM02':raise ValueError('model version')
    import zlib
    if len(raw)!=32+ls+sz or zlib.crc32(raw[32:])!=crc:raise ValueError('model integrity')
    labels=json.loads(raw[32:32+ls])['labels'];offset=32+ls
    gamma=struct.unpack_from('<d',raw,offset)[0];offset+=8
    counts=struct.unpack_from('<'+'I'*c,raw,offset);offset+=4*c
    floats=memoryview(raw[offset:]).cast('d')
    sv=floats[:d*n];coefs=floats[d*n:d*n+(c-1)*n];inter=floats[d*n+(c-1)*n:]
    lines=text.read_text().splitlines();split=lines.index('SV');header=dict(line.split(' ',1) for line in lines[:split]);body=lines[split+1:]
    if (int(header['nr_class'])!=c or int(header['total_sv'])!=n or float(header['gamma'])!=gamma or
        list(map(int,header['nr_sv'].split()))!=list(counts) or list(map(int,header['label'].split()))!=list(range(c)) or len(body)!=n):raise ValueError('LIBSVM header')
    if list(map(float,header['rho'].split()))!=[-v for v in inter]:raise ValueError('LIBSVM intercept')
    for k,line in enumerate(body):
        fields=line.split()
        if len(fields)!=c-1+d:raise ValueError('LIBSVM row width')
        if [float(v) for v in fields[:c-1]]!=[coefs[j*n+k] for j in range(c-1)]:raise ValueError('LIBSVM coefficients')
        for j,item in enumerate(fields[c-1:]):
            index,value=item.split(':')
            if int(index)!=j+1 or float(value)!=sv[k*d+j]:raise ValueError('LIBSVM support')
    return c,n,d,labels


def audit(root,source):
    panel=root/'panel';freeze=json.loads((panel/'TEST_OPENING.json').read_text())
    for name,sha in freeze['files'].items():
        if digest(panel/name)!=sha:raise ValueError('changed frozen artifact '+name)
    for name,sha in freeze['source'].items():
        if digest(source/name)!=sha:raise ValueError('changed numerical source '+name)
    acquisition=json.loads((root/'acquisition/acquisition.json').read_text())
    for r in acquisition['files']:
        path=root/'acquisition'/(r['name'] if r['name'].startswith('libsvm/') else 'data/'+r['name']+'.zip')
        if path.stat().st_size!=r['bytes'] or digest(path)!=r['sha256']:raise ValueError('source acquisition')
    reports={};timings=0;decisions=0
    for task in TASKS:
        folder=root/'formal'/task;model=panel/task/'models'
        meta=json.loads((folder/'metadata.json').read_text());split=json.loads((panel/task/'split.json').read_text())
        fitting=json.loads((model/'fitting.json').read_text());quality=json.loads((folder/'quality_fidelity.json').read_text())
        if fitting['selected']!=max(fitting['models'][:4],key=lambda r:r['validation_correct'])['name']:raise ValueError('validation selection')
        c,n,d,labels=check_export(model/'selected.spc',model/'selected.libsvm')
        if meta['labels']!=labels or meta['dimensions']!=d or meta['rows']!=len(split['test']):raise ValueError('metadata')
        expected_hashes={'model':digest(model/'selected.spc'),'libsvm_model':digest(model/'selected.libsvm'),
                         'libsvm_library':digest(root/'build/libpanel.so'),'spectra_library':digest(root/'build/spectra/libspectra_svm.so')}
        if meta['hashes']!=expected_hashes:raise ValueError('loaded identity')
        if any(set(split[a])&set(split[b]) for a,b in [('fit','validation'),('fit','test'),('validation','test')]):raise ValueError('split overlap')
        if 'subjects' in split and any(set(split['subjects'][a])&set(split['subjects'][b]) for a,b in [('fit','validation'),('fit','test'),('validation','test')]):raise ValueError('subject overlap')
        _,expected=npz(folder/'test_inputs.npz','expected');_,y=npz(folder/'test_inputs.npz','y')
        _,original_y=npz(panel/task/'original.npz','y')
        if list(y)!=[original_y[i] for i in split['test']]:raise ValueError('test labels')
        # Compare every timed/replayed feature with its original sealed row and preprocessing.
        _,ox=npz(panel/task/'original.npz','x');_,rx=npz(folder/'test_inputs.npz','raw');_,tx=npz(folder/'test_inputs.npz','x')
        _,mean=npz(model/'scaler.npz','mean');_,scale=npz(model/'scaler.npz','scale')
        for k,original_id in enumerate(split['test']):
            for j in range(d):
                value=ox[original_id*d+j]
                if rx[k*d+j]!=value or tx[k*d+j]!=(value-mean[j])/scale[j]:raise ValueError('feature or preprocessing mismatch')
        control_outputs={}
        for control in ('logistic','mlp'):
            _,cp=npz(folder/'control_predictions.npz',control)
            if len(cp)!=len(y):raise ValueError('control prediction length')
            control_outputs[control]=cp
            if quality['quality'][control]!={'correct':sum(a==b for a,b in zip(cp,y)),'rows':len(y)}:raise ValueError('control accuracy')
        correct=sum(labels[p]==label for p,label in zip(expected,y))
        if quality['quality']['selected']!={'correct':correct,'rows':len(expected)}:raise ValueError('accuracy summary')
        for arm in ARMS:
            _,pred=npz(folder/'predictions.npz',arm)
            if list(pred)!=list(expected) or quality['fidelity'][arm]!={'mismatches':0,'rows':len(expected)}:raise ValueError('full predictions')
            decisions+=len(expected)
        work=[json.loads(l) for l in (folder/'work.jsonl').read_text().splitlines()]
        if len(work)!=len(expected):raise ValueError('work inventory')
        for row,pred in zip(work,expected):
            if row['class_index']!=pred or not certificate(c,pred,row['certificate']):raise ValueError('vote certificate')
            if any(type(row[k])is not int or row[k]<0 for k in ('pairs','kernels','terms')):raise ValueError('work counter types')
            if row['pairs']!=sum(v!=-1 for v in row['certificate']) or not 0<=row['kernels']<=n:raise ValueError('work counters')
        positions=split['timing_test_positions']
        if meta['timing_positions']!=positions or meta['seed']!=SEED or meta['repeats']!=11:raise ValueError('timing spec')
        nr=[json.loads(l) for l in (folder/'native.jsonl').read_text().splitlines()]
        pg=[json.loads(l) for l in (folder/'public.jsonl').read_text().splitlines()]
        groups=check_rows(nr,positions,ARMS,expected)
        public=check_rows(pg,positions,PUBLIC,[labels[i] for i in expected],True)
        for row in pg:
            if row['arm'] in control_outputs and row['label']!=control_outputs[row['arm']][row['case']]:raise ValueError('timed control output')
        batch=[json.loads(l) for l in (folder/'batch64.jsonl').read_text().splitlines()]
        if len(batch)!=33:raise ValueError('batch coverage')
        rng=random.Random(SEED);batch_groups=defaultdict(list)
        for rep in range(11):
            order=['spectra','libsvm','sklearn'];rng.shuffle(order)
            for k,name in enumerate(order):
                row=batch[rep*3+k]
                if set(row)!={'repeat','arm','ns','rows','matched'} or row['repeat']!=rep or row['arm']!=name or row['rows']!=len(positions) or row['matched'] is not True or type(row['ns'])is not int or row['ns']<=0:raise ValueError('batch order or output')
                batch_groups[name].append(row['ns'])
        timings+=len(nr)+len(pg)+len(batch)
        reports[task]=dict(rows=len(expected),features=d,classes=c,supports=n,quality=quality['quality'],
              tables=quality['storage']['cert_tables']['model']['tables_enabled'],
              mean_kernels=statistics.mean(r['kernels'] for r in work),mean_pairs=statistics.mean(r['pairs'] for r in work),
              duplicate_test_features=len(split['duplicate_feature_test_ids']),
              native=summarize(groups,positions,ARMS),public=summarize(public,positions,PUBLIC),
              comparisons={base:paired(groups,positions,'cert_tables',base) for base in ('exhaustive_direct','cert_direct','libsvm','knockout_direct')},
              batch64_us_per_row={a:statistics.median(v)/len(positions)/1000 for a,v in batch_groups.items()})
    successes=sum(r['comparisons']['exhaustive_direct']['speedup']>=1.25 for r in reports.values())
    return dict(status='PASS',audit_scope='source/inventory/fidelity/arithmetic, not external reproduction or universal proof',
        timing_cells=timings,executor_test_decisions=decisions,unique_test_rows=sum(r['rows'] for r in reports.values()),
        qualifying_tasks=successes,primary_gate='PASS' if successes>=4 and all(r['comparisons']['exhaustive_direct']['ratio']<=1.1 for r in reports.values()) else 'FAIL',tasks=reports)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    report=audit(a.evidence,a.source)
    with a.out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='tasks'},indent=2))
