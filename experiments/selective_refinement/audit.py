"""Independent recorded-evidence audit; standard library, no model execution.

Reconstructs selection, independent data roles, calibration bounds, routing,
mechanical exports and all timing arithmetic. It does not authenticate clocks,
prove IID sampling, reproduce training, or certify correctness of every label.
"""
from __future__ import annotations
import ast, functools, hashlib, io, json, math, random, statistics, struct, zipfile, zlib
from pathlib import Path
TASKS=('letter','pendigits','satellite','optdigits')
FAMILIES=('fast','strong','mlp')
ARMS=('fast','strong','primary','strict','loose','blind','mlp32')
THRESHOLDS=(0.,.25,.5,1.,2.,3.,4.,6.,8.,12.,16.,24.,32.)
BUDGETS=(.005,.01,.02)

def require(condition,message):
    if not condition: raise ValueError(message)
def unique(pairs):
    out={}
    for k,v in pairs:
        require(k not in out,'duplicate JSON key');out[k]=v
    return out
def decode(text):
    def bad(value): raise ValueError('nonfinite JSON')
    return json.loads(text,object_pairs_hook=unique,parse_constant=bad)
def read(path): return decode(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()
def member(root,name):
    require(type(name) is str and '\\' not in name,'invalid relative path')
    p=Path(name);require(not p.is_absolute() and '..' not in p.parts,'escaping inventory path')
    target=Path(root)
    for part in p.parts:
        target/=part;require(not target.is_symlink(),'symlink in evidence')
    require(target.is_file(),'missing evidence: '+name)
    return target
def verify_files(root,files,exact=False):
    for name,h in files.items():require(sha(member(root,name))==h,'file/source changed: '+name)
    if exact:
        require(set(files)=={p.relative_to(root).as_posix() for p in Path(root).rglob('*') if p.is_file()},'source closure mismatch')
def same(actual,expected):
    if isinstance(expected,dict):
        require(type(actual) is dict,'expected mapping')
        for key,v in expected.items():require(key in actual,'missing derived field');same(actual[key],v)
    elif type(expected) is float:
        require(type(actual) in (int,float) and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-11),'derived number differs')
    else:require(type(actual) is type(expected) and actual==expected,'derived value differs')
def npy(raw):
    require(len(raw)>=10 and raw[:6]==b'\x93NUMPY' and raw[6] in (1,2,3),'invalid NPY')
    nsize=2 if raw[6]==1 else 4;offset=8+nsize;n=int.from_bytes(raw[8:offset],'little')
    require(n<=65536 and offset+n<=len(raw),'array header cap')
    h=ast.literal_eval(raw[offset:offset+n].decode('latin1'))
    require(type(h) is dict and h.get('fortran_order') is False,'array order')
    shape=h.get('shape');require(type(shape) is tuple and all(type(v) is int and v>=0 for v in shape),'array shape')
    formats={'|u1':'B','|b1':'?','<u2':'H','<i2':'h','<i4':'i','<i8':'q','<u8':'Q','<f4':'f','<f8':'d'}
    require(h.get('descr') in formats,'unsupported inert array dtype')
    fmt=formats[h['descr']];payload=raw[offset+n:];count=math.prod(shape)
    require(count<=16_000_000 and len(payload)==count*struct.calcsize('<'+fmt),'array byte inventory')
    return {'shape':shape,'data':list(struct.unpack('<'+str(count)+fmt,payload)),'bytes':payload,'format':fmt}
def arrays(path):
    with zipfile.ZipFile(path) as z:
        names=z.namelist();require(len(names)==len(set(names)) and len(names)<=64,'NPZ inventory')
        result={}
        for name in names:
            require('/' not in name and name.endswith('.npy') and z.getinfo(name).file_size<=128*1024**2,'NPZ member bound')
            result[name[:-4]]=npy(z.read(name))
        return result
def rowbytes(a):
    require(len(a['shape'])==2 and a['format']=='B','expected uint8 features')
    n,d=a['shape'];return [a['bytes'][i*d:(i+1)*d] for i in range(n)]
def dataset(root,task,test=False):
    if task!='optdigits':
        a=arrays(root/'inputs'/f'{task}-{"test" if test else "train"}.npz')
        return rowbytes(a['q']),a['y']['data'],a['maximum']['data'][0]
    with zipfile.ZipFile(root/'inputs/optdigits.zip') as z:
        rows=[[int(v) for v in line.split(',')] for line in z.read('optdigits.tes' if test else 'optdigits.tra').decode().splitlines()]
    require(len(rows)==(1797 if test else 3823) and all(len(r)==65 and all(0<=v<=16 for v in r[:64]) for r in rows),'official optical shape/domain')
    return [bytes(r[:64]) for r in rows],[r[-1] for r in rows],16

def quality(pred,truth):
    require(len(pred)==len(truth)>0,'prediction row inventory')
    labels=sorted(set(truth));index={v:i for i,v in enumerate(labels)};m=[[0]*len(labels) for _ in labels]
    for t,p in zip(truth,pred):require(p in index,'unknown predicted class');m[index[t]][index[p]]+=1
    correct=sum(a==b for a,b in zip(pred,truth));fs=[]
    for i in range(len(labels)):
        total=sum(m[i])+sum(r[i] for r in m);fs.append(2*m[i][i]/total if total else 0.)
    return {'rows':len(truth),'correct':correct,'accuracy':correct/len(truth),'macro_f1':statistics.mean(fs),'confusion':m}

def grids(task,family):
    if family=='fast':return [{'gamma':g} for g in ((8.,32.) if task=='satellite' else (.125,.5,2.) if task=='optdigits' else (2.,8.))]
    if family=='strong':return [{'C':c,'gamma':g} for c in (1.,10.,100.) for g in ((.125,.5,2.) if task=='optdigits' else (.5,2.,8.))]
    return [{'width':w,'alpha':a} for w in (128,256) for a in (1e-5,.001)]

@functools.lru_cache(None)
def cp_upper(k,n,delta):
    """Invert binomial CDF independently, without scipy/beta-quantile code."""
    require(type(k) is int and type(n) is int and 0<=k<=n and n>0,'binomial counts')
    if k==n:return 1.
    if not k:return -math.expm1(math.log(delta)/n)
    logs=[math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1) for j in range(k+1)]
    lo,hi=k/n,1.
    for _ in range(70):
        p=(lo+hi)/2;lp=math.log(p);lq=math.log1p(-p)
        values=[a+j*lp+(n-j)*lq for j,a in enumerate(logs)];maximum=max(values)
        logcdf=maximum+math.log(math.fsum(math.exp(v-maximum) for v in values))
        if logcdf>math.log(delta):lo=p
        else:hi=p
    return (lo+hi)/2

def blind_value(row):
    mask=(1<<64)-1;h=1469598103934665603
    for q in row:h=((h^q)*1099511628211)&mask
    h^=0x9e3779b97f4a7c15;h=((h^(h>>30))*0xbf58476d1ce4e5b9)&mask
    h=((h^(h>>27))*0x94d049bb133111eb)&mask;h^=h>>31
    return (h>>11)*2.**-53

def score_gap(scores,labels):
    n,c=scores['shape'];require(c==len(labels) and scores['format']=='d','score shape/dtype')
    gaps=[];pred=[]
    for r in range(n):
        row=scores['data'][r*c:(r+1)*c];require(all(math.isfinite(v) for v in row),'invalid class score')
        winner=max(range(c),key=lambda j:row[j]);ordered=sorted(row)
        gaps.append(ordered[-1]-ordered[-2]);pred.append(labels[winner])
    return pred,gaps

def verify_exports(folder,D):
    a=arrays(folder/'fast/model.npz');s=read(folder/'fast/settings.json');p,d=a['centers']['shape'];classes=a['classes']['data'];c=len(classes)
    require(p==256 and s['prototypes']==256 and s['quarter']==4 and s['units']==4 and s['arm']=='local','changed fixed-budget classifier')
    bound=4*d*(D*4)**2;bits=(bound.bit_length()+1)//2;alpha=s['gamma']/(4*(D*4)**2)
    meta=json.dumps({'labels':classes},ensure_ascii=True,separators=(',',':'),allow_nan=False).encode()
    payload=struct.pack('<d',alpha)
    for name,fmt in (('centers','H'),('weights','H'),('head','d'),('bias','d')):
        values=a[name]['data'];payload+=struct.pack('<'+str(len(values))+fmt,*values)
    payload+=meta
    expected=struct.pack('<8sIIIIIIIIII',b'SPPRO001',d,p,c,D,4,4,bits,len(meta),len(payload),zlib.crc32(payload))+payload
    require(expected==(folder/'fast/model.spp').read_bytes(),'prototype export differs from frozen arrays')
    neural=arrays(folder/'mlp/weights.npz');dims=[neural['w0']['shape'][0]]+[neural[f'w{i}']['shape'][1] for i in range(3)]
    meta=json.dumps({'labels':neural['classes']['data']},separators=(',',':'),allow_nan=False).encode();payload=struct.pack('<4I',*dims)
    for name in ('mean','scale','w0','b0','w1','b1','w2','b2'):
        values=neural[name]['data'];payload+=struct.pack('<'+str(len(values))+'f',*values)
    payload+=meta
    expected=struct.pack('<8sIIIIIII',b'SPNF0001',dims[0],dims[-1],D,3,len(meta),len(payload),zlib.crc32(payload))+payload
    require(expected==(folder/'mlp/model.sfn').read_bytes(),'FP32 model differs from mechanical conversion')
    return classes

def audit(root):
    root=Path(root);fit=root/'refinement_fit';cal=root/'refinement_calibration';ev=root/'refinement_evaluation';bench=root/'refinement_benchmark'
    initial=read(fit/'LOCK.json');models=read(fit/'FINAL_MODELS.json');cl=read(cal/'CALIBRATION_LOCK.json');opening=read(ev/'OPENING.json');bl=read(bench/'LOCK.json')
    require(initial['parent']=='c5af3032e1f9dd8908e922836dd7c06d4222b876','wrong experiment parent')
    same(initial,{'tasks':list(TASKS),'calibration_split_seed':20260930,'selection_split_seed':611})
    same(initial['grids'],{t:{f:grids(t,f) for f in FAMILIES} for t in TASKS})
    require(models['selection_lock_sha256']==sha(fit/'LOCK.json'),'selection source binding')
    require(models['model_count']==12 and len(models['selection'])==61 and len(models['fits'])==12,'model selection/final count')
    verify_files(root/'inputs',initial['training_inputs']);verify_files(fit/'source',initial['source'],True);verify_files(fit,models['files'])
    verify_files(cal,cl['files']);co=read(cal/'OPENING.json');verify_files(cal/'source',co['source'],True)
    require(cl['models_sha256']==sha(fit/'FINAL_MODELS.json')==co['models_sha256']==opening['model_lock_sha256']==bl['models_sha256'],'model lock chain broken')
    require(opening['calibration_lock_sha256']==sha(cal/'CALIBRATION_LOCK.json'),'calibration lock chain broken')
    require(cl['policies_sha256']==sha(cal/'POLICIES.json')==bl['policies_sha256'],'policy bytes changed')
    verify_files(ev/'source',opening['source'],True);verify_files(bench/'source',bl['source'],True)
    require(bl['evaluation_sha256']==sha(ev/'RESULTS.json'),'evaluation result changed after timing')
    native=sha(root/'refinement-native-final/refinement.so');mlp=sha(root/'mlp-native/control.so')
    require(native==opening['native_sha256']==bl['library_sha256']==co['library_sha256'],'native artifact changed')
    require(mlp==opening['mlp_sha256']==bl['mlp_sha256'],'MLP artifact changed')
    policies=read(cal/'POLICIES.json');outcomes=read(ev/'RESULTS.json');truths={};predictions={};roles={};calculated={};score_count=0
    selection_seen=[];fit_seen=[]
    for task in TASKS:
        q,y,D=dataset(root,task);r=arrays(fit/task/'roles.npz');ids={k:r[k]['data'] for k in ('development','calibration','fit','validation')};roles[task]=ids
        for key,v in ids.items():require(len(v)==len(set(v)) and all(type(i) is int and 0<=i<len(q) for i in v),'invalid role index')
        dev,calid,train,validation=(ids[k] for k in ('development','calibration','fit','validation'))
        require(set(dev)|set(calid)==set(range(len(q))) and not set(dev)&set(calid),'development/calibration roles overlap or omit rows')
        require(set(train)<=set(dev) and set(validation)<=set(dev) and not set(train)&set(validation),'selection leaks calibration')
        require(not {q[i] for i in dev}&{q[i] for i in calid} and not {q[i] for i in train}&{q[i] for i in validation},'duplicate-feature group leakage')
        choice=read(fit/task/'SELECTION.json')
        for family in FAMILIES:
            records=[]
            for i,setting in enumerate(grids(task,family)):
                folder=fit/task/f'{family}-choice{i}';record=read(folder/'record.json');a=arrays(folder/'validation.npz')
                true=[y[j] for j in validation];require(a['expected']['data']==true and len(a['prediction']['data'])==len(true),'selection label identity')
                same(record,{'task':task,'family':family,'index':i,'choice':setting,'correct':sum(x==t for x,t in zip(a['prediction']['data'],true)),
                             'fit_rows':len(train),'validation_rows':len(validation)})
                records.append(record);selection_seen.append(record)
            best=max(records,key=lambda x:(x['correct'],-x['index']));require(choice[family]==best,'selection differs from fixed tie rule')
            final=read(fit/task/family/'fit.json');same(final,{'task':task,'family':family,'choice':best['choice'],'training_rows':len(dev),'calibration_rows':len(calid),'features':len(q[0]),'maximum':D});fit_seen.append(final)
        labels=verify_exports(fit/task,D)
        obs=arrays(cal/task/'observations.npz');require(obs['indices']['data']==calid and obs['truth']['data']==[y[i] for i in calid],'calibration role/label mismatch')
        fp,gaps=score_gap(obs['scores'],labels);require(fp==obs['fast']['data'] and gaps==obs['gap']['data'],'calibration confidence differs from saved scores')
        sp=obs['strong']['data'];true=obs['truth']['data'];report=read(cal/task/'calibration.json');same(report,{'delta':.05,'finite_threshold_count':13})
        rows=[]
        for t in THRESHOLDS:
            accepted=[g>=t for g in gaps];harm=sum(a and f!=y and s==y for a,f,s,y in zip(accepted,fp,sp,true))
            rows.append({'threshold':t,'accepted':sum(accepted),'n':len(true),'harmful':harm,'upper_added_harm':cp_upper(harm,len(true),.05/13)})
        require(len(report['rows'])==13,'calibration threshold count')
        for a,b in zip(report['rows'],rows):same(a,b)
        for budget in BUDGETS:
            selected=next((row for row in rows if row['upper_added_harm']<=budget),{'threshold':None,'accepted':0,'n':len(true),'harmful':0,'upper_added_harm':0.})
            desired={**selected,'risk_budget':budget,'accept_fraction':selected['accepted']/len(true),'always_strong':selected['threshold'] is None}
            same(report['policies'][str(budget)],desired);same(policies[task][str(budget)],desired)
        testq,testy,testD=dataset(root,task,True);inp=arrays(ev/task/'input.npz');pred=arrays(ev/task/'predictions.npz')
        require(rowbytes(inp['q'])==testq and inp['truth']['data']==testy and inp['maximum']['data']==[testD],'official evaluation data changed')
        require((ev/task/'input.u8').read_bytes()==b''.join(testq),'native input bytes mismatch')
        require(pred['truth']['data']==testy,'prediction truth changed');truths[task]=testy
        fp,gaps=score_gap(pred['fast_scores'],labels);sp=pred['strong']['data'];require(pred['fast']['data']==fp and pred['gap']['data']==gaps,'test confidence mismatch')
        fidelity=read(ev/task/'score-fidelity.json');same(fidelity,{'score_values':len(testy)*len(labels),'sha256':hashlib.sha256(pred['fast_scores']['bytes']).hexdigest(),'all_equal':True});score_count+=fidelity['score_values']
        rec=outcomes['tasks'][task];require(rec['policies']==policies[task],'outcome policy changed')
        counts={}
        for name,key in (('primary','0.01'),('strict','0.005'),('loose','0.02')):
            policy=policies[task][key];t=policy['threshold'];accept=[False if t is None else g>=t for g in gaps]
            result=[f if a else s for f,s,a in zip(fp,sp,accept)];require(result==pred[name]['data'],'threshold routing disagrees')
            harm=sum(a and f!=y and s==y for a,f,s,y in zip(accept,fp,sp,testy));benefit=sum(a and f==y and s!=y for a,f,s,y in zip(accept,fp,sp,testy))
            fixes=sum(x==y and f!=y for x,f,y in zip(result,fp,testy));broken=sum(x!=y and f==y for x,f,y in zip(result,fp,testy))
            cp=sum(x==y for x,y in zip(result,testy));cs=sum(x==y for x,y in zip(sp,testy));cf=sum(x==y for x,y in zip(fp,testy));n=len(testy)
            details={'accepted_fast':sum(accept),'escalated':n-sum(accept),'observed_added_harm':harm/n,'harmful_accepted':harm,'beneficial_accepted':benefit,
                     'fixed_fast_errors':fixes,'introduced_fast_errors':broken,'gain_over_fast_points':100*(cp-cf)/n,'loss_against_strong_points':100*(cs-cp)/n}
            same(rec[name],details);counts[name]=details
            same(rec['work'][name],{'fast_evaluations':0 if t is None else n,'strong_evaluations':n-sum(accept),'fast_accepted':sum(accept)})
        fraction=policies[task]['0.01']['accept_fraction'];accept=[blind_value(q)<fraction for q in testq]
        require(pred['blind']['data']==[f if a else s for f,s,a in zip(fp,sp,accept)],'confidence-blind routing changed')
        for name in ARMS:
            values=pred[name]['data'];same(rec['quality'][name],quality(values,testy));predictions[task,name]=values
        same(rec['work']['fast'],{'fast_evaluations':len(testy),'strong_evaluations':0,'fast_accepted':len(testy)})
        same(rec['work']['strong'],{'fast_evaluations':0,'strong_evaluations':len(testy),'fast_accepted':0})
        same(rec['work']['blind'],{'fast_evaluations':len(testy),'strong_evaluations':len(testy)-sum(accept),'fast_accepted':sum(accept)})
        devset={q[i] for i in dev};calset={q[i] for i in calid}
        same(rec['overlap'],{'test_rows_seen_in_development':sum(q in devset for q in testq),'test_rows_seen_in_calibration':sum(q in calset for q in testq)})
        calculated[task]=counts
    require(selection_seen==models['selection'] and fit_seen==models['fits'],'summary selection/fitting inventories changed')
    same(outcomes['independent_prototype_scores'],score_count)
    same(bl,{'tasks':list(TASKS),'arms':list(ARMS),'chunks':[1,32,256],'repeats':7,'seed':2026093037})
    rng=random.Random(2026093037);schedule=[]
    for t in TASKS:
        for r in range(7):
            cells=[(c,a) for c in (1,32,256) for a in ARMS];rng.shuffle(cells);schedule.extend((t,r,c,a) for c,a in cells)
    times=[decode(l) for l in (bench/'timings.jsonl').read_text().splitlines()];require(len(times)==len(schedule)==588,'timing grid incomplete')
    for row,key in zip(times,schedule):
        require(tuple(row[k] for k in ('task','repeat','chunk','arm'))==key,'timing schedule changed')
        require(all(type(row[k]) is int for k in ('repeat','chunk','rows','ns')) and row['ns']>0,'invalid timing type/value')
        values=predictions[row['task'],row['arm']];require(row['rows']==len(values),'timing row count')
        digest=hashlib.sha256(struct.pack('<'+str(len(values))+'q',*values)).hexdigest();require(row['prediction_sha256']==digest,'timed prediction digest mismatch')
    summary=read(bench/'RESULTS.json');results={}
    for task in TASKS:
        n=len(truths[task]);med={str(c):{a:statistics.median(r['ns'] for r in times if r['task']==task and r['chunk']==c and r['arm']==a)/n/1000 for a in ARMS} for c in (1,32,256)}
        ratio=med['32']['primary']/med['32']['strong'];quality_record=outcomes['tasks'][task]['quality'];cf,cp,cs=(quality_record[a]['correct'] for a in ('fast','primary','strong'))
        gate=200*(cp-cf)>=n and ratio<=.5 and 100*(cs-cp)<=n
        same(summary['tasks'][task],{'us_per_row':med,'primary_over_strong':ratio,'gain_over_fast_points':100*(cp-cf)/n,'loss_against_strong_points':100*(cs-cp)/n,'gate':gate})
        results[task]={'fast_correct':cf,'refined_correct':cp,'strong_correct':cs,'test_rows':n,'primary_over_strong':ratio,'gate':gate,
                       'observed_added_harm':calculated[task]['primary']['observed_added_harm'],'above_nominal_harm_budget':calculated[task]['primary']['observed_added_harm']>.01}
    passed=sum(r['gate'] for r in results.values())>=2
    same(summary,{'timing_cells':588,'repeated_predictions':sum(r['rows'] for r in times),'two_task_gate':passed})
    return {'status':'PASS','selection_cells':61,'frozen_models':12,'calibration_thresholds':52,'timing_cells':588,
            'underlying_test_rows':sum(map(len,truths.values())),'independent_recorded_prototype_scores':score_count,
            'tasks':results,'two_task_gate':passed,
            'scope':'recorded model/source/role identities, independent bound inversion, routed labels and arithmetic; not IID/shift assurance or authenticated clocks'}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();print(json.dumps(audit(a.root),indent=2))
