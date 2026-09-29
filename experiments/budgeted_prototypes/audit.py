"""Independent standard-library audit of model selection, outputs and timing.

No learner/runtime import, numerical Python dependency or pickle execution. Verifies
recorded identities and arithmetic, not clock truth, provenance authenticity or
unrecorded human adaptation. Whole dependency environments are outside the packet.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,math,random,statistics,struct,zipfile
from pathlib import Path
TASKS=('letter','pendigits','satellite','optdigits')
FAMILIES=('fixed','centers','local','svc','mlp','linear')
PROTO=FAMILIES[:3]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unique(items):
    out={}
    for k,v in items:
        if k in out:raise ValueError('duplicate JSON key')
        out[k]=v
    return out

def decode(text):
    def fail(x):raise ValueError('nonfinite JSON')
    def finite_float(value):
        number=float(value)
        if not math.isfinite(number):raise ValueError('nonfinite JSON number')
        return number
    return json.loads(text,object_pairs_hook=unique,parse_constant=fail,parse_float=finite_float)
def read(p):return decode(Path(p).read_text(encoding='utf-8'))
def check(ok,message):
    if not ok:raise ValueError(message)
def same(a,b):
    if isinstance(b,dict):
        for k,v in b.items():check(k in a,'missing derived field');same(a[k],v)
    elif type(b) is float:check(type(a) in (int,float) and math.isfinite(a) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),'derived number mismatch')
    else:check(type(a) is type(b) and a==b,'derived value mismatch')
def member(root,name):
    check(type(name) is str and '\\' not in name,'invalid inventory path')
    p=Path(name);check(not p.is_absolute() and '..' not in p.parts,'escaping inventory')
    target=Path(root)
    for part in p.parts:
        target/=part;check(not target.is_symlink(),'symlink inventory member')
    check(target.is_file(),'missing inventory member '+name)
    return target

def npy(raw):
    check(type(raw) is bytes and len(raw)>=10 and raw[:6]==b'\x93NUMPY'
          and tuple(raw[6:8]) in ((1,0),(2,0),(3,0)),'invalid NPY signature')
    size=2 if raw[6]==1 else 4;offset=8+size
    check(len(raw)>=offset,'truncated NPY header length')
    n=int.from_bytes(raw[8:offset],'little')
    check(0<n<65536 and offset+n<=len(raw),'invalid NPY header')
    try:h=ast.literal_eval(raw[offset:offset+n].decode('utf-8' if raw[6]==3 else 'latin1'))
    except (SyntaxError,ValueError,TypeError,UnicodeError,RecursionError) as e:
        raise ValueError('invalid NPY header syntax') from e
    check(type(h) is dict and set(h)=={'descr','fortran_order','shape'}
          and h['fortran_order'] is False,'unsupported array header/order')
    shape=h['shape'];check(type(shape) is tuple and len(shape)<=8 and all(type(n) is int and n>=0 for n in shape),'array shape')
    formats={'|u1':'B','|b1':'?','<i8':'q','<u8':'Q','<i4':'i','<u2':'H','<i2':'h','<f8':'d','<f4':'f'}
    check(type(h['descr']) is str and h['descr'] in formats,'unsupported NPY dtype')
    fmt=formats[h['descr']];body=raw[offset+n:];count=math.prod(shape)
    check(count<=16_000_000 and len(body)==struct.calcsize('<'+fmt)*count,'array inventory')
    return {'shape':shape,'values':list(struct.unpack('<'+str(count)+fmt,body)),'bytes':body}

def arrays(path):
    with zipfile.ZipFile(path) as z:
        names=z.namelist();check(len(names)==len(set(names)) and len(names)<=128,'NPZ inventory')
        check(sum(z.getinfo(n).file_size for n in names)<=256*1024**2,'NPZ aggregate cap')
        out={}
        for name in names:
            check(name.endswith('.npy') and '/' not in name and '\\' not in name and z.getinfo(name).file_size<=128*1024**2,'NPZ member cap')
            out[name[:-4]]=npy(z.read(name))
        return out

def data_path(root,t,test=False):
    if t=='optdigits':return root/('evaluation/optdigits-test.npz' if test else 'new_data/optdigits-train.npz')
    if t=='satellite':return root/'prior/satellite/data'/('test.npz' if test else 'train.npz')
    return root/'prior/datasets'/(t+('-test.npz' if test else '-train.npz'))
def feature_rows(a):
    n,d=a['shape'];check(len(a['bytes'])==n*d,'expected uint8 feature matrix')
    return [a['bytes'][i*d:(i+1)*d] for i in range(n)]
def quality(p,y):
    check(len(p)==len(y),'prediction length');labels=sorted(set(y));index={v:i for i,v in enumerate(labels)};m=[[0]*len(labels) for _ in labels]
    for a,b in zip(y,p):check(b in index,'unknown predicted label');m[index[a]][index[b]]+=1
    f1=[]
    for i in range(len(labels)):
        denom=sum(m[i])+sum(row[i] for row in m);f1.append(2*m[i][i]/denom if denom else 0.)
    correct=sum(a==b for a,b in zip(p,y))
    return {'correct':correct,'rows':len(y),'accuracy':correct/len(y),'macro_f1':statistics.mean(f1),'labels':labels,'confusion':m}
def hashes(record, required=()):
    check(type(record) is dict and bool(record),'empty or invalid identity binding')
    check(set(required)<=set(record),'incomplete required identity binding')
    for name,h in record.items():
        check(type(name) is str and type(h) is str and len(h)==64
              and all(c in '0123456789abcdef' for c in h),'invalid digest binding')

def sources(record,folder,required=()):
    hashes(record,required)
    for name,h in record.items():check(sha(member(folder,name))==h,'source snapshot changed '+name)

def audit(root):
    root=Path(root);initial=read(root/'selection/LOCK.json');amend=read(root/'selection_optical/LOCK.json')
    check(initial['split_seed_inventory']==[611,977] and len(initial['jobs'])==248,'initial selection inventory')
    check(len(amend['jobs'])==36 and amend['original_selection_sha256']==sha(root/'selection/LOCK.json'),'amendment inventory')
    sources(initial['source'],root/'source_snapshots/selection',('learning.py','study.py'));sources(amend['source'],root/'source_snapshots/optical',('learning.py','study.py'))
    training={t:arrays(data_path(root,t)) for t in TASKS};splits={};allrows=[]
    for t in TASKS:
        check(sha(data_path(root,t))==initial['inputs'][t],'training identity changed')
        q=feature_rows(training[t]['q']);y=training[t]['y']['values']
        for seed in (611,977):
            s=arrays(root/f'selection/{t}-{seed}-split.npz');f=s['fit']['values'];v=s['validation']['values']
            check(len(f)==len(set(f)) and len(v)==len(set(v)) and not set(f)&set(v),'repeated/overlapping split index')
            check(all(type(i) is int and 0<=i<len(q) for i in f+v),'split index bounds')
            check(not {q[i] for i in f}&{q[i] for i in v},'feature-group leakage')
            splits[t,seed]=(f,v)
    original_keys=set();amend_keys=set()
    for role,lock in [('selection',initial),('selection_optical',amend)]:
        check(all(type(j.get('job')) is int for j in lock['jobs']) and [j['job'] for j in lock['jobs']]==list(range(len(lock['jobs']))),'invalid job index inventory')
        for job in lock['jobs']:
            folder=root/role/f'job-{job["job"]:03d}';r=read(folder/'record.json');same(r,job)
            task=r['task'];arm=r['arm'];seed=r['seed'];key=(task,arm,seed,tuple(sorted(r['choice'].items())))
            seen=original_keys if role=='selection' else amend_keys
            check(key not in seen,'duplicate selected candidate');seen.add(key)
            hashes(r['files'],('validation.npz','training.json','model.npz' if arm in PROTO else 'model.pkl'))
            for file,h in r['files'].items():check(sha(member(folder,file))==h,'candidate artifact changed')
            p=arrays(folder/'validation.npz');f,v=splits[task,seed];wanted=[training[task]['y']['values'][i] for i in v]
            check(p['expected']['values']==wanted,'validation label mismatch')
            same(r,{'correct':sum(a==b for a,b in zip(p['prediction']['values'],wanted)),'fit_rows':len(f),'validation_rows':len(v)})
            check(len(p['prediction']['values'])==len(wanted),'missing validation prediction');allrows.append(r)
    expected_initial=set()
    for t in TASKS:
        for seed in (611,977):
            for arm in (*PROTO,'svc','mlp'):
                if arm in PROTO:choices=[{'prototypes':p,'gamma':g} for p in (256,512,1024) for g in ((8.,32.) if t=='satellite' else (2.,8.))]
                elif arm=='svc':choices=[{'C':c,'gamma':g} for c in (1.,10.,100.) for g in ((.125,.5,2.) if t=='optdigits' else (.5,2.,8.))]
                else:choices=[{'width':w,'alpha':a} for w in (128,256) for a in (1e-5,.001)]
                expected_initial|={(t,arm,seed,tuple(sorted(c.items()))) for c in choices}
    expected_extra={('optdigits',a,s,tuple(sorted({'prototypes':p,'gamma':g}.items()))) for a in PROTO for s in (611,977) for p in (256,512,1024) for g in (.125,.5)}
    check(original_keys==expected_initial and amend_keys==expected_extra,'incomplete declared grid')
    final_choices=read(root/'selection_final/selected.json')
    for t in TASKS:
        for arm in (*PROTO,'svc','mlp'):
            rows=[r for r in allrows if r['task']==t and r['arm']==arm]
            choices=[]
            for r in rows:
                if r['choice'] not in choices:choices.append(r['choice'])
            if t=='optdigits' and arm in PROTO:
                choices=sorted(choices,key=lambda c:(c['prototypes'],(2.,8.,.125,.5).index(c['gamma'])))
            else:
                # Restore declared grid order, not randomized execution order.
                choices=sorted(choices,key=lambda c:next(r['choice_index'] for r in rows if r['choice']==c))
            ranked=[]
            for i,c in enumerate(choices):
                rr=[r for r in rows if r['choice']==c];check(len(rr)==2,'wrong split count')
                ranked.append((sum(r['correct'] for r in rr)/sum(r['validation_rows'] for r in rr),-i,c))
            best=max(ranked,key=lambda x:x[:2]);same(final_choices[t][arm],{'choice':best[2],'validation_accuracy':best[0]})
    lock=read(root/'models/FINAL_LOCK.json');opening=read(root/'evaluation/TEST_OPENING.json')
    check(type(lock['models']) is int and lock['models']==24 and type(lock['fits']) is list and len(lock['fits'])==24,'final model inventory')
    expected_models={(t,a) for t in TASKS for a in FAMILIES}
    check(all(type(f) is dict and type(f.get('task')) is str and type(f.get('arm')) is str for f in lock['fits']),'invalid final fit entry')
    check({(f['task'],f['arm']) for f in lock['fits']}==expected_models,'duplicate or missing final fit')
    required_files={f'{t}-{a}/model.'+('spp' if a in PROTO else 'srt' if a=='svc' else 'snn') for t,a in expected_models}
    required_files|={f'{t}-{a}/fit.json' for t,a in expected_models}
    hashes(lock['files'],required_files)
    check(opening['final_lock_sha256']==sha(root/'models/FINAL_LOCK.json'),'wrong evaluation lock')
    check(read(root/'models/FIT_LOCK.json')['selection_sha256']==sha(root/'selection_final/selected.json'),'wrong selected-model lock')
    sources(lock['source'],root/'source_snapshots/final_fit',('learning.py','study.py','export.py'));sources(opening['source'],root/'source_snapshots/evaluation',('evaluate.py',))
    for name,h in lock['files'].items():check(sha(member(root/'models',name))==h,'frozen model changed '+name)
    for fit in lock['fits']:
        same(fit,{'training_rows':training[fit['task']]['q']['shape'][0],'features':training[fit['task']]['q']['shape'][1],'maximum':training[fit['task']]['maximum']['values'][0]})
        if fit['arm']!='linear':same(fit['choice'],final_choices[fit['task']][fit['arm']]['choice'])
    qreport=read(root/'evaluation/quality.json');predictions={};truth={}
    for t in TASKS:
        original=arrays(data_path(root,t,True));q=feature_rows(original['q']);y=original['y']['values'];truth[t]=y
        check((root/f'evaluation/{t}/input.u8').read_bytes()==original['q']['bytes'],'timed features changed')
        check(npy((root/f'evaluation/{t}/truth.npy').read_bytes())['values']==y,'evaluation truth changed')
        known=set(feature_rows(training[t]['q']));mask=[r not in known for r in q]
        same(qreport[t],{'overlap_rows':sum(not x for x in mask),'nonoverlap_rows':sum(mask)})
        for arm in FAMILIES:
            result=arrays(root/f'evaluation/{t}/{arm}-predictions.npz');p=result['prediction']['values'];predictions[t,arm]=p
            check(result['expected']['values']==y and result['nonoverlap']['values']==mask,'prediction truth/overlap changed')
            same(qreport[t]['arms'][arm],quality(p,y))
            same(qreport[t]['arms'][arm]['nonoverlap'],quality([v for v,m in zip(p,mask) if m],[v for v,m in zip(y,mask) if m]))
        for baseline in ('fixed','centers','svc','mlp'):
            a=predictions[t,'local'];b=predictions[t,baseline];wins=sum(x==z and y!=z for x,y,z in zip(a,b,y));losses=sum(x!=z and y==z for x,y,z in zip(a,b,y))
            n=wins+losses;k=min(wins,losses);p=min(1.,2*sum(math.comb(n,j) for j in range(k+1))/2**n) if n else 1.
            same(qreport[t]['local_vs_'+baseline],{'gain_points':100.*(wins-losses)/len(y),'candidate_only_correct':wins,'baseline_only_correct':losses,'paired_binomial_p':p})
    # Independently decode the new official files and bind both train/test partitions.
    with zipfile.ZipFile(root/'new_data/optdigits.zip') as z:
        for part,t in [('optdigits.tra',training['optdigits']),('optdigits.tes',arrays(root/'evaluation/optdigits-test.npz'))]:
            raw=z.read(part);rows=[[int(v) for v in s.split(',')] for s in raw.decode().splitlines()]
            check([v for r in rows for v in r[:-1]]==t['q']['values'] and [r[-1] for r in rows]==t['y']['values'],'official optical data mismatch')
    check(sha(root/'new_data/optdigits.zip')==opening['optical_archive_sha256'],'official archive identity')
    fp32=read(root/'models_fp32/LOCK.json');sources(fp32['source'],root/'source_snapshots/float_conversion',('float_control.py',))
    fp32_report=read(root/'models_fp32/evaluation.json')
    for t in TASKS:
        a=arrays(root/f'models/{t}-mlp/weights.npz');entry=fp32['models'][t]
        check(sha(root/f'models/{t}-mlp/weights.npz')==entry['original_weights_sha256'],'FP32 source weights changed')
        dims=[a['w0']['shape'][0]]+[a[f'w{i}']['shape'][1] for i in range(3)]
        meta=json.dumps({'labels':a['classes']['values']},separators=(',',':'),allow_nan=False).encode()
        payload=struct.pack('<4I',*dims)
        for name in ('mean','scale','w0','b0','w1','b1','w2','b2'):
            values=a[name]['values'];payload+=struct.pack('<'+str(len(values))+'f',*values)
        payload+=meta
        expected=struct.pack('<8sIIIIIII',b'SPNF0001',dims[0],dims[-1],a['maximum']['values'][0],3,len(meta),len(payload),__import__('zlib').crc32(payload))+payload
        file=root/f'models_fp32/{t}.sfn'
        check(file.read_bytes()==expected and sha(file)==entry['sha256'] and file.stat().st_size==entry['bytes'],'mechanical FP32 conversion mismatch')
        pred=arrays(root/f'models_fp32/{t}-predictions.npz')
        check(pred['expected']['values']==truth[t] and pred['prediction']['values']==predictions[t,'mlp'],'FP32 retained labels changed')
        same(fp32_report[t],{'correct':sum(a==b for a,b in zip(pred['prediction']['values'],truth[t])),'rows':len(truth[t]),'disagreements_with_original':0})
    reports={}
    for folder,seed,extra,snapshot in [('benchmark',2026092909,False,'first_benchmark'),('benchmark_strong',2026092912,True,'strong_benchmark'),('benchmark_final',2026092917,True,'final_benchmark')]:
        run=root/folder;p=read(run/'protocol.json');arms=tuple(f'{f}_{l}' for l in ('original','packet','register') for f in PROTO)+('local_scalar','local_direct_exp','svc','mlp','linear')+(('mlp_blas','svc_finite') if extra else ())
        if folder=='benchmark_final':
            arms+=('mlp_float32',);check(p['fp32_lock_sha256']==sha(root/'models_fp32/LOCK.json'),'timing FP32 model lock')
        check(p['arms']==list(arms) and p['tasks']==list(TASKS) and p['chunks']==[1,32,256] and p['repeats']==7 and p['seed']==seed,'timing protocol inventory')
        sources(p['source'],root/'source_snapshots'/snapshot,('bench.py','runtime.cpp'))
        check(p['final_lock_sha256']==sha(root/'models/FINAL_LOCK.json') and p['quality_sha256']==sha(root/'evaluation/quality.json'),'timing model/quality lock')
        hashes(p['libraries'])
        library_paths=[(Path(name).parent.name,Path(name).name) for name in p['libraries']]
        expected_library_count=7 if folder=='benchmark_final' else 6 if extra else 4
        check(len(library_paths)==expected_library_count and len(set(library_paths))==len(library_paths),'missing or aliased native library binding')
        for name,h in p['libraries'].items():
            path=Path(name);check(sha(root/path.parent.name/path.name)==h,'native library bytes differ')
        wanted=[];rng=random.Random(seed)
        for t in TASKS:
            for repeat in range(7):
                jobs=[(c,a) for c in (1,32,256) for a in arms];rng.shuffle(jobs);wanted.extend((t,repeat,c,a) for c,a in jobs)
        observed=[decode(s) for s in (run/'timings.jsonl').read_text().splitlines()]
        check(len(observed)==len(wanted),'missing timing cell')
        for r,key in zip(observed,wanted):
            check(tuple(r[k] for k in ('task','repeat','chunk','arm'))==key,'timing order differs')
            check(all(type(r[k]) is int for k in ('repeat','chunk','rows','ns')) and r['ns']>0,'invalid time/shape type')
            pred=predictions[r['task'],r['arm'].split('_')[0]]
            check(r['rows']==len(pred) and r['prediction_sha256']==hashlib.sha256(struct.pack('<'+str(len(pred))+'q',*pred)).hexdigest(),'timed output mismatch')
        summary=read(run/'summary.json');ratios=[];passes=0
        for t in TASKS:
            med={str(c):{a:statistics.median(r['ns'] for r in observed if r['task']==t and r['chunk']==c and r['arm']==a)/len(truth[t])/1000 for a in arms} for c in (1,32,256)}
            same(summary['tasks'][t]['us_per_row'],med);v=med['32'];ratio=v['local_register']/v['fixed_register'];gain=qreport[t]['local_vs_fixed']['gain_points'];gate=gain>=.5 and ratio<=1.25;passes+=gate
            layout={f:v[f+'_register']/v[f+'_original'] for f in PROTO};ratios.extend(layout.values())
            same(summary['tasks'][t],{'local_over_fixed':ratio,'local_gain_points':gain,'joint_gate':gate,'layout_ratios':layout,
                 'local_over_mlp':v['local_register']/v['mlp'],'local_over_svc':v['local_register']/v['svc'],'direct_exp_prediction_disagreements':0})
            if extra:same(summary['tasks'][t],{'local_over_mlp_blas':v['local_register']/v['mlp_blas'],'local_over_svc_finite':v['local_register']/v['svc_finite']})
            if folder=='benchmark_final':
                same(summary['tasks'][t],{'local_over_mlp_float32':v['local_register']/v['mlp_float32']})
        g=statistics.geometric_mean(ratios);same(summary,{'cells':len(observed),'checked_predictions':sum(r['rows'] for r in observed),'layout_ratio':g,'layout_gate':g<=1/1.10 and max(ratios)<=1.10,'primary_two_task_gate':passes>=2})
        reports[folder]={'cells':len(observed),'primary_two_task_gate':passes>=2,'layout_ratio':g,'layout_gate':g<=1/1.10 and max(ratios)<=1.10}
    return {'status':'PASS','auditor_sha256':sha(__file__),'selection_cells':len(allrows),'final_models':24,'underlying_evaluation_rows':sum(map(len,truth.values())),
            'timing':reports,'scope':'recorded splits, identities, outcomes and arithmetic; no clock authentication or independent researcher replication'}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();result=audit(a.root)
    with a.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
