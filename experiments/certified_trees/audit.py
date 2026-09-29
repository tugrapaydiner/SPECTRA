"""Standard-library evidence audit: full inventories, decisions and all timings.

Optional recompilation executes exact hashed compiler source bytes, not cached
bytecode. This is deterministic policy reconstruction, not an independent second
quantization algorithm. Native source-score tests are separate observations.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,math,random,statistics,struct,sys,types,zipfile
from pathlib import Path
TASKS=('letter','pendigits','satellite','optdigits')
BASE_ARMS=('q8_end','q8_early','q16_end','q16_early','q16_scalar','q16_full_end','q16_full_early','q8_q16','catboost_refine_end','catboost_refine_early','full_fp64','catboost_native','catboost_export')

def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def loads(text):
    def pairs(items):
        r={}
        for k,v in items:require(k not in r,'duplicate JSON key');r[k]=v
        return r
    def bad(_):raise ValueError('nonfinite JSON constant')
    def number(v):
        r=float(v);require(math.isfinite(r),'nonfinite JSON number');return r
    return json.loads(text,object_pairs_hook=pairs,parse_constant=bad,parse_float=number)
def read(p):return loads(Path(p).read_text(encoding='utf-8'))
def member(root,name):
    require(type(name) is str and '\\' not in name,'invalid inventory name');p=Path(name)
    require(not p.is_absolute() and '..' not in p.parts,'escaping path');result=Path(root)
    for part in p.parts:result/=part;require(not result.is_symlink(),'symlink inventory')
    require(result.is_file(),'missing file '+name);return result

def identities(root,record,required=()):
    require(type(record) is dict and bool(record) and set(required)<=set(record),'incomplete identity inventory')
    for n,h in record.items():require(type(h) is str and len(h)==64 and sha(member(root,n))==h,'identity differs '+n)
def near(a,b):require(type(a) in (int,float) and math.isfinite(a) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),'derived numeric value differs')

def npy(raw):
    require(len(raw)>=10 and raw[:6]==b'\x93NUMPY' and tuple(raw[6:8]) in ((1,0),(2,0),(3,0)),'NPY header')
    nsize=2 if raw[6]==1 else 4;pos=8+nsize;require(len(raw)>=pos,'short NPY length');n=int.from_bytes(raw[8:pos],'little')
    require(0<n<65536 and pos+n<=len(raw),'NPY header cap')
    try:h=ast.literal_eval(raw[pos:pos+n].decode('utf8' if raw[6]==3 else 'latin1'))
    except (SyntaxError,ValueError,UnicodeError,RecursionError) as e:raise ValueError('NPY syntax') from e
    require(type(h) is dict and set(h)=={'descr','fortran_order','shape'} and type(h['fortran_order']) is bool,'NPY metadata')
    shape=h['shape'];require(type(shape) is tuple and len(shape)<=4 and all(type(x) is int and x>=0 for x in shape),'NPY shape')
    fmts={'<i8':'q','<u8':'Q','<i4':'i','<u4':'I','|u1':'B','<f8':'d'};require(h['descr'] in fmts,'NPY dtype')
    count=math.prod(shape);fmt=fmts[h['descr']];body=raw[pos+n:]
    require(count<=2000000 and len(body)==count*struct.calcsize('<'+fmt),'NPY data cap')
    values=list(struct.unpack('<'+str(count)+fmt,body))
    if h['fortran_order'] and len(shape)>1:
        require(len(shape)==2,'only 2D Fortran arrays supported')
        nrows,ncols=shape;values=[values[i+j*nrows] for i in range(nrows) for j in range(ncols)]
        body=struct.pack('<'+str(count)+fmt,*values)
    return {'shape':shape,'values':values,'bytes':body}
def arrays(path):
    out={}
    with zipfile.ZipFile(path) as z:
        names=z.namelist();require(len(names)==len(set(names)) and len(names)<=80,'NPZ member inventory')
        require(sum(x.file_size for x in z.infolist())<=64*1024**2,'NPZ total cap')
        for n in names:
            require(n.endswith('.npy') and '/' not in n and '\\' not in n and z.getinfo(n).file_size<=16*1024**2,'NPZ member cap');out[n[:-4]]=npy(z.read(n))
    return out

def reconstruct(source,root):
    # Import the literal source that is hashed into the receipt, never .pyc.
    co_name='certificate_oracle';co=types.ModuleType(co_name);co.__file__=str(source/'certificate_oracle.py');sys.modules[co_name]=co
    raw=(source/'certificate_oracle.py').read_bytes();exec(compile(raw,co.__file__,'exec'),co.__dict__)
    packing=types.ModuleType('verified_packing');packing.__file__=str(source/'packing.py');packing.__package__='';sys.modules[packing.__name__]=packing
    exec(compile((source/'packing.py').read_bytes(),packing.__file__,'exec'),packing.__dict__)
    results=[]
    for t in TASKS:
        raw=(root/'models'/t/'source.json').read_bytes()
        for variant in ('q8','q16','full'):
            results.append({'task':t,'variant':variant,**packing.verify_binary(raw,(root/'compiled'/t/(variant+'.sct')).read_bytes())})
    return {'source_sha256':{n:sha(source/n) for n in ('certificate_oracle.py','packing.py')},'verified_binaries':results}

def audit(root,source,recompile=False):
    root=Path(root);source=Path(source);mlock=read(root/'models/FINAL_LOCK.json')
    identities(root/'models',mlock['files'],[f'{t}/source.{ext}' for t in TASKS for ext in ('cbm','json','cpp')])
    compiled=read(root/'compiled/COMPILED_LOCK.json');identities(root/'compiled',compiled['files'],[f'{t}/q{b}.sct' for t in TASKS for b in (8,16)])
    require(compiled['source_model_lock_sha256']==sha(root/'models/FINAL_LOCK.json'),'compiled source lock')
    fidelity=read(root/'fidelity/report.json');require(fidelity['status']=='PASS' and fidelity['compiled_lock_sha256']==sha(root/'compiled/COMPILED_LOCK.json'),'fidelity model lock')
    predicted={};counts={};total=0
    for t in TASKS:
        a=arrays(root/'fidelity'/f'{t}.npz');r=fidelity['tasks'][t];require(sha(root/'fidelity'/f'{t}.npz')==r['output_sha256'],'fidelity array identity')
        idx=a['source_indices']['values'];n=len(idx);total+=n;predicted[t]={};counts[t]={}
        original=arrays(root/'source_evaluation'/f'{t}.npz')
        require(idx==original['indices']['values'] and a['q']['bytes']==original['q']['bytes'],'source evaluation identity')
        require(r['source_json_sha256']==sha(root/'models'/t/'source.json'),'original model source differs')
        near(r['source_accuracy_correct'],sum(x==y for x,y in zip(original['prediction']['values'],original['y']['values'])))
        for b in (8,16):
            for label,cp in (('end',0),('early',16)):
                key=f'q{b}_cp{cp}';p=a[key+'_indices']['values'];steps=a[key+'_steps']['values'];approx=a[key+'_approx']['values'];obs=r['results'][key]
                require(len(p)==len(steps)==len(approx)==n,'prediction length')
                require(all(v==-1 or v==target for v,target in zip(p,idx)),'false accepted decision')
                near(obs['certified'],sum(v>=0 for v in p));near(obs['unresolved'],sum(v<0 for v in p));near(obs['mean_trees'],statistics.mean(steps));near(obs['naive_wrong_on_completed'],sum(v>=0 and v!=target for v,target in zip(approx,idx)))
                require(all(type(v) is int and 0<=v<=256 for v in steps),'invalid tree work count')
                predicted[t][f'q{b}_{label}']=p;counts[t][key]={'certified':sum(v>=0 for v in p),'unresolved':sum(v<0 for v in p),'naive_wrong':sum(v>=0 and v!=target for v,target in zip(approx,idx))}
        predicted[t]['q16_scalar']=predicted[t]['q16_end'];predicted[t]['q16_row_major']=predicted[t]['q16_end'];predicted[t]['q8_q16']=a['cascade_indices']['values']
        for k in BASE_ARMS+('refine_row_major','owned_refine'):
            if k not in predicted[t]:predicted[t][k]=idx
        # Preserve every inherited complete-call source-score observation. They are
        # recorded experimental checks, not an independent CPU-library proof here.
        for k in ('q8_hybrid0','q8_hybrid16','q16_hybrid0','q16_hybrid16'):
            require(a[k+'_indices']['values']==idx,'saved hybrid mismatch')
    require(total==11295,'complete four-task row count')
    results={}
    for dirname,seed,extra,snapshot in [('benchmark',2026092908,(),'sources_v1'),('benchmark_tiled',2026092918,('q16_row_major','refine_row_major'),'sources_v2'),('benchmark_final',2026092920,('q16_row_major','refine_row_major','owned_refine'),'sources_v3')]:
        run=root/dirname;p=read(run/'PROTOCOL.json');arms=BASE_ARMS+extra
        require(p['tasks']==list(TASKS) and p['arms']==list(arms) and p['batches']==[1,32,256] and p['repeats']==7 and p['seed']==seed,'complete benchmark specification')
        identities(root,p['files'],('models/FINAL_LOCK.json','compiled/COMPILED_LOCK.json'))
        identities(root/snapshot,p['sources'],('runtime.cpp','benchmark.py','session.py','controls.py'))
        obs=[loads(s) for s in (run/'observations.jsonl').read_text().splitlines()];order=[];rng=random.Random(seed)
        for t in TASKS:
            for rep in range(7):
                jobs=[(b,a) for b in (1,32,256) for a in arms];rng.shuffle(jobs);order.extend((t,rep,b,a) for b,a in jobs)
        require(len(obs)==len(order),'missing timing observations')
        for r,key in zip(obs,order):
            require(tuple(r[k] for k in ('task','repeat','batch','arm'))==key and all(type(r[k]) is int for k in ('repeat','batch','rows','ns')) and r['ns']>0,'timing identity/order')
            out=predicted[r['task']][r['arm']];digest=hashlib.sha256(struct.pack('<'+str(len(out))+'i',*out)).hexdigest()
            require(r['rows']==len(out) and r['output_sha256']==digest,'timed output differs')
        summary=read(run/'summary.json');near(summary['cells'],len(obs));ratios=[]
        for t in TASKS:
            n=len(predicted[t]['catboost_native'])
            for b in (1,32,256):
                for arm in arms:
                    value=statistics.median(r['ns'] for r in obs if (r['task'],r['batch'],r['arm'])==(t,b,arm))/n/1000
                    near(summary['tasks'][t]['us_per_row'][str(b)][arm],value)
            u=summary['tasks'][t]['us_per_row']['32']
            for a,v in summary['tasks'][t]['batch32_over_native'].items():near(v,u[a]/u['catboost_native'])
            ratios.append(u['owned_refine' if 'owned_refine' in arms else 'catboost_refine_end']/u['catboost_native'])
            for a in arms:near(summary['tasks'][t]['coverage'][a],sum(v>=0 for v in predicted[t][a])/n)
        aggregate=statistics.geometric_mean(ratios)
        near(summary['equal_task_owned_refine_ratio' if 'owned_refine' in arms else 'equal_task_refine_end_ratio'],aggregate)
        results[dirname]={'observations':len(obs),'full_coverage_over_native':aggregate}
    result={'status':'PASS','underlying_rows':total,'certificate_counts':counts,'timing':results,'auditor_sha256':sha(__file__),
            'scope':'recorded source/data/decision/timing identities and complete summary arithmetic; not authentic clocks, a hostile-evidence signature, or new quality confirmation'}
    if recompile:result['recompiled']=reconstruct(source,root)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--recompile',action='store_true');a=p.parse_args()
    r=audit(a.root,a.source,a.recompile)
    with a.out.open('x') as f:json.dump(r,f,indent=2)
    print(json.dumps(r,indent=2))
