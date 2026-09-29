"""Standalone audit of fixed-source replay and complete timing records.

Checks identities, integer outputs and derived arithmetic, without importing the
candidate, unpickling models or executing native code. It is not clock attestation
or a second proof of every exact-rational compiler operation.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,math,random,statistics,struct
from pathlib import Path
TASKS=('letter','pendigits','satellite','optdigits')
BASE=('official_128','official_1210','export_cpp','int8_scalar_end','int16_scalar_end',
      'int8_tiled_end','int16_tiled_end','int8_scalar_early','int16_scalar_early',
      'int8_tiled_early','int16_tiled_early','refine_8_16','refine_8_16_official',
      'refine_16_official','refine_8_16_official_early')
EXTRA=('vector_8_16_official','vector_16_official','register_8_tiled_end','register_16_tiled_end',
       'register_8_16','register_8_16_official','register_16_official')
ORACLE_SHA='704556048f58844787a89c2079710abbce949cc7f5d427e821fcb25f64f5c6a5'

def check(test,message):
    if not test:raise ValueError(message)

def loads(raw):
    def unique(pairs):
        d={}
        for k,v in pairs:
            check(k not in d,'duplicate JSON key');d[k]=v
        return d
    def finite(v):
        f=float(v);check(math.isfinite(f),'nonfinite JSON number');return f
    def fail(_):raise ValueError('nonfinite JSON token')
    check(len(raw)<=64*1024**2,'JSON byte cap')
    return json.loads(raw,object_pairs_hook=unique,parse_float=finite,parse_constant=fail)

def read(p):return loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def child(root,name):
    check(type(name) is str and '\\' not in name,'invalid path')
    parts=Path(name);check(not parts.is_absolute() and '..' not in parts.parts,'escaping path')
    p=Path(root)
    for s in parts.parts:p/=s;check(not p.is_symlink(),'symlink in evidence')
    check(p.is_file(),'missing evidence file: '+name)
    return p

def bindings(mapping,root,required=()):
    check(type(mapping) is dict and bool(mapping) and set(required)<=set(mapping),'missing identity binding')
    for name,h in mapping.items():
        check(type(h) is str and len(h)==64 and all(c in '0123456789abcdef' for c in h),'invalid digest')
        check(sha(child(root,name))==h,'changed bound file: '+name)

def indices(path,n):
    raw=Path(path).read_bytes();check(len(raw)==4*n,'index count mismatch')
    return list(struct.unpack('<'+str(n)+'i',raw))

def truth(path,n):
    raw=Path(path).read_bytes();check(len(raw)>=10 and raw[:6]==b'\x93NUMPY','NPY magic')
    check(tuple(raw[6:8]) in ((1,0),(2,0)),'NPY version');sz=2 if raw[6]==1 else 4;o=8+sz
    check(len(raw)>=o,'NPY header size');length=int.from_bytes(raw[8:o],'little');check(0<length<65536 and o+length<=len(raw),'NPY header cap')
    try:h=ast.literal_eval(raw[o:o+length].decode('latin1'))
    except (SyntaxError,ValueError,RecursionError) as e:raise ValueError('NPY header malformed') from e
    check(h=={'descr':'<i8','fortran_order':False,'shape':(n,)} or h=={'descr':'<i4','fortran_order':False,'shape':(n,)},'unexpected truth array')
    fmt='q' if h['descr']=='<i8' else 'i';body=raw[o+length:];check(len(body)==n*struct.calcsize(fmt),'truth count')
    return struct.unpack('<'+str(n)+fmt,body)

def audit(root,source):
    root=Path(root);source=Path(source)
    check(sha(source/'reference/certificate_oracle.py')==ORACLE_SHA,'checkpoint oracle modified')
    models=root/'frozen_models';lock=read(models/'MODEL_LOCK.json')
    check(len(lock['models'])==4 and {r['task'] for r in lock['models']}==set(TASKS),'source model inventory')
    bindings(lock['files'],models,[t+'/model.'+ext for t in TASKS for ext in ('json','cbm','cpp')])
    check(read(root/'evaluation/OPENING.json')['model_lock_sha256']==sha(models/'MODEL_LOCK.json'),'source/test opening mismatch')
    quality=read(root/'evaluation/QUALITY.json');expected={};totals=0
    for t in TASKS:
        r=quality[t];n=r['rows'];d=r['features'];check(type(n) is int and n>0 and type(d) is int and 0<d<=256,'invalid data shape')
        bindings(r['files'],root/'evaluation'/t,('input.u8','indices.i32','truth.npy','source_scores.f64'))
        q=(root/'evaluation'/t/'input.u8').read_bytes();check(len(q)==n*d and max(q)<=r['maximum'],'raw domain/size')
        expected[t]=indices(root/'evaluation'/t/'indices.i32',n);labels=r['classes'];y=truth(root/'evaluation'/t/'truth.npy',n)
        check(all(0<=v<len(labels) for v in expected[t]),'source class range')
        correct=sum(labels[k]==v for k,v in zip(expected[t],y))
        check(correct==r['correct'] and math.isclose(correct/n,r['accuracy'],rel_tol=1e-15),'quality arithmetic mismatch');totals+=n
        for bits in (8,16):
            packed=root/'compiled'/f'{t}-{bits}.sct';raw=packed.read_bytes();rec=read(root/'compiled'/f'{t}-{bits}.receipt.json')
            check(sha(packed)==rec['packed_sha256'] and len(raw)==rec['bytes'],'compiled identity')
            check(raw[60:92].hex()==sha(models/t/'model.json')==rec['source_sha256'],'compiled source identity')
            check(raw[92:124].hex()==sha(root/'compiled'/f'{t}-{bits}.oracle.json')==rec['oracle_sha256'],'oracle compiled identity')
            check(rec['oracle_source_sha256']==ORACLE_SHA and rec['status']=='PASS','compiler oracle receipt')
    reports={}
    for name,arms,seed,snapshot in (('benchmark',BASE,2026092921,'baseline'),('benchmark_layouts',BASE+EXTRA,2026092923,'final')):
        run=root/name;p=read(run/'PROTOCOL.json');check(p['tasks']==list(TASKS) and p['arms']==list(arms) and p['chunks']==[1,32,256] and p['repeats']==7 and p['seed']==seed,'complete matrix protocol differs')
        bindings(p['source'],root/'snapshots'/snapshot,('runtime.cpp','session.py','packed.py','reference/certificate_oracle.py'))
        check(p['model_lock_sha256']==sha(models/'MODEL_LOCK.json') and p['quality_sha256']==sha(root/'evaluation/QUALITY.json'),'timing model binding')
        check(p['library_sha256']==sha(root/'native-avx2/trees.so'),'baseline runtime changed')
        check(type(p['official']) is dict and set(p['official'])=={'1.2.8','1.2.10'},'missing official-library binding')
        for v,h in p['official'].items():check(h==sha(root/'official'/f'libcatboostmodel-linux-x86_64-{v}.so'),'official library changed')
        if name=='benchmark_layouts':
            check(type(p['new_libraries']) is dict and set(p['new_libraries'])=={'vector','register'},'missing candidate-library binding')
            for v,h in p['new_libraries'].items():check(h==sha(root/f'native-{v}-final/trees.so'),'new runtime changed')
        bindings(p['compact'],root/'compiled',[f'{t}-{b}.sct' for t in TASKS for b in (8,16)])
        outputs={}
        for t in TASKS:
            for arm in arms:
                path=run/f'{t}-{arm}.indices';v=indices(path,len(expected[t]));check(all(x==-1 or x==y for x,y in zip(v,expected[t])),'wrong certified index')
                if 'official' in arm or arm=='export_cpp':check(v==expected[t],'missing full-coverage output')
                outputs[t,arm]=(sha(path),v.count(-1))
        wanted=[];rng=random.Random(seed)
        for t in TASKS:
            for rep in range(7):
                jobs=[(c,a) for c in (1,32,256) for a in arms];rng.shuffle(jobs);wanted.extend((t,rep,c,a) for c,a in jobs)
        rows=[loads(x) for x in (run/'rows.jsonl').read_bytes().splitlines()]
        check(len(rows)==len(wanted),'missing timing observation')
        for r,keys in zip(rows,wanted):
            check(tuple(r[k] for k in ('task','repeat','chunk','arm'))==keys,'timing order/identity')
            check(all(type(r[k]) is int for k in ('repeat','chunk','rows','ns','unresolved')) and r['ns']>0,'invalid timing value')
            digest,missing=outputs[r['task'],r['arm']]
            check(r['rows']==len(expected[r['task']]) and r['prediction_sha256']==digest and r['unresolved']==missing,'timed output binding')
        summary=read(run/'SUMMARY.json');check(summary['cells']==len(rows) and summary['underlying_rows']==totals,'aggregate count')
        for t in TASKS:
            for c in (1,32,256):
                for arm in arms:
                    value=statistics.median(r['ns'] for r in rows if r['task']==t and r['chunk']==c and r['arm']==arm)/len(expected[t])/1000
                    check(math.isclose(summary['tasks'][t]['us_per_row'][str(c)][arm],value,rel_tol=1e-12),'timing median differs')
        reports[name]={'cells':len(rows),'status':'PASS'}
    replay=read(root/'layout_replay_final/LAYOUT_REPLAY.json');check(replay['status']=='PASS' and len(replay['comparisons'])==48,'layout replay inventory')
    check(all(x['matches_original'] is True for x in replay['comparisons']),'layout replay failure')
    return {'status':'PASS','source_models':4,'underlying_rows':totals,'timing':reports,
      'compact16_certified':sum(len(expected[t])-indices(root/'benchmark_layouts'/f'{t}-register_16_tiled_end.indices',len(expected[t])).count(-1) for t in TASKS),
      'scope':'byte identities, retained output agreement and arithmetic; no attestation of clocks or independent generalization'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();report=audit(a.root,a.source)
    with a.out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
