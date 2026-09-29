"""Synthetic receipt tests, never model/clock evidence or reported benchmark results.

The fixture exercises the entire 284-choice, 24-model and three-timing-run schema
with tiny invented data. Every synthetic payload is generated here, not passed off
as a recovered classifier, real measurement or reproduction of a scientific run.
"""
from __future__ import annotations
import copy, hashlib, importlib.util, json, math, random, shutil, struct, zipfile, zlib
from pathlib import Path
import pytest

SPEC = importlib.util.spec_from_file_location('prototype_audit_tested', Path(__file__).with_name('audit.py'))
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def digest(raw): return hashlib.sha256(raw).hexdigest()
def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')
def write_bytes(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
def npy(shape, values, dtype='<i8'):
    fmts={'<i8':'q','|u1':'B','|b1':'?','<f8':'d','<f4':'f'}
    assert math.prod(shape) == len(values)
    header=repr({'descr':dtype,'fortran_order':False,'shape':shape}).encode('ascii')
    header += b' ' * ((64-(10+len(header)+1)%64)%64) + b'\n'
    return b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header+struct.pack('<'+str(len(values))+fmts[dtype],*values)
def npz(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,(shape,arr,*dtype) in values.items():z.writestr(name+'.npy',npy(shape,arr,*dtype))
def hashfile(path):return digest(path.read_bytes())
def snapshot(root, role):
    files={n:b'# SYNTHETIC SOURCE TEST FIXTURE\n' for n in ('learning.py','study.py','runtime.cpp','export.py','session.py','bench.py','evaluate.py','float_control.py')}
    for n,v in files.items():write_bytes(root/'source_snapshots'/role/n,v)
    return {n:digest(v) for n,v in files.items()}
def choices(task,arm,extra=False):
    if arm in audit.PROTO:
        gs=(.125,.5) if extra else ((8.,32.) if task=='satellite' else (2.,8.))
        return [{'prototypes':p,'gamma':g} for p in (256,512,1024) for g in gs]
    if arm=='svc':return [{'C':c,'gamma':g} for c in (1.,10.,100.) for g in ((.125,.5,2.) if task=='optdigits' else (.5,2.,8.))]
    return [{'width':w,'alpha':a} for w in (128,256) for a in (1e-5,.001)]


def make_fixture(root):
    write_json(root/'SYNTHETIC_TEST_FIXTURE.json',{'scope':'invented input/model bytes and clocks for auditor tests; not scientific evidence'})
    train_q=list(range(4));train_y=[0,1,0,1];test_q=[0,4,5,6];test_y=[0,1,0,1]
    inp={}
    for task in audit.TASKS:
        tr=audit.data_path(root,task);te=audit.data_path(root,task,True)
        npz(tr,q=((4,1),train_q,'|u1'),y=((4,),train_y),maximum=((),[15]))
        npz(te,q=((4,1),test_q,'|u1'),y=((4,),test_y),maximum=((),[15]));inp[task]=hashfile(tr)
        for seed in (611,977):npz(root/f'selection/{task}-{seed}-split.npz',fit=((2,),[0,1]),validation=((2,),[2,3]))
    archive=root/'new_data/optdigits.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for fn,q,y in [('optdigits.tra',train_q,train_y),('optdigits.tes',test_q,test_y)]:
            z.writestr(fn,''.join(f'{x},{label}\n' for x,label in zip(q,y)))
    jobs=[]
    for task in audit.TASKS:
        for seed in (611,977):
            for arm in (*audit.PROTO,'svc','mlp'):
                jobs.extend({'task':task,'seed':seed,'arm':arm,'choice_index':i,'choice':c} for i,c in enumerate(choices(task,arm)))
    random.Random(20260929).shuffle(jobs)
    for i,j in enumerate(jobs):j['job']=i
    extra=[{'seed':s,'arm':a,'choice':c} for s in (611,977) for a in audit.PROTO for c in choices('optdigits',a,True)]
    random.Random(2026092910).shuffle(extra)
    for i,j in enumerate(extra):j['job']=i
    initial={'source':snapshot(root,'selection'),'inputs':inp,'jobs':jobs,'split_seed_inventory':[611,977]}
    write_json(root/'selection/LOCK.json',initial)
    amend={'source':snapshot(root,'optical'),'jobs':extra,'original_selection_sha256':hashfile(root/'selection/LOCK.json'),'training_sha256':inp['optdigits']}
    write_json(root/'selection_optical/LOCK.json',amend)
    for role,inventory in [('selection',jobs),('selection_optical',extra)]:
        for j in inventory:
            d=root/role/f'job-{j["job"]:03d}';proto=j['arm'] in audit.PROTO
            write_bytes(d/('model.npz' if proto else 'model.pkl'),b'SYNTHETIC MODEL PLACEHOLDER, NOT EXECUTABLE\n')
            write_json(d/'training.json',{'scope':'synthetic auditor fixture'})
            npz(d/'validation.npz',prediction=((2,),[0,1]),expected=((2,),[0,1]))
            record={**j,'task':j.get('task','optdigits'),'correct':2,'fit_rows':2,'validation_rows':2,'cpu_seconds':1.,'files':{p.name:hashfile(p) for p in d.iterdir() if p.is_file()}}
            write_json(d/'record.json',record)
    selected={t:{arm:{'choice':choices(t,arm)[0],'choice_index':0,'validation_accuracy':1.} for arm in (*audit.PROTO,'svc','mlp')} for t in audit.TASKS}
    write_json(root/'selection_final/selected.json',selected)
    write_json(root/'selection_final/LOCK.json',{'original_lock_sha256':hashfile(root/'selection/LOCK.json'),'amendment_lock_sha256':hashfile(root/'selection_optical/LOCK.json'),'selected_sha256':hashfile(root/'selection_final/selected.json'),'all_selection_cells':284,'selection_only':True})
    fits=[]
    for task in audit.TASKS:
        for arm in audit.FAMILIES:
            d=root/f'models/{task}-{arm}';suffix='spp' if arm in audit.PROTO else 'srt' if arm=='svc' else 'snn'
            write_bytes(d/f'model.{suffix}',b'SYNTHETIC MODEL PLACEHOLDER\n')
            fit={'task':task,'arm':arm,'choice':selected[task][arm]['choice'] if arm!='linear' else {},'training_rows':4,'features':1,'maximum':15}
            write_json(d/'fit.json',fit);fits.append(fit)
        npz(root/f'models/{task}-mlp/weights.npz',
            w0=((1,2),[.1,.2],'<f8'),b0=((2,),[.01,.02],'<f8'),
            w1=((2,2),[.3,.4,.5,.6],'<f8'),b1=((2,),[.03,.04],'<f8'),
            w2=((2,2),[.7,.8,.9,1.],'<f8'),b2=((2,),[.05,.06],'<f8'),
            mean=((1,),[.25],'<f8'),scale=((1,),[.5],'<f8'),maximum=((),[15]),classes=((2,),[0,1]))
    write_json(root/'models/FIT_LOCK.json',{'source':snapshot(root,'final_fit'),'selection_sha256':hashfile(root/'selection_final/selected.json'),'inputs':inp})
    lock={'models':24,'fits':fits,'source':snapshot(root,'final_fit'),'files':{p.relative_to(root/'models').as_posix():hashfile(p) for p in (root/'models').rglob('*') if p.is_file()}}
    write_json(root/'models/FINAL_LOCK.json',lock)
    pred=[0,1,1,1];mask=[False,True,True,True];qreport={}
    for task in audit.TASKS:
        write_bytes(root/f'evaluation/{task}/input.u8',bytes(test_q));write_bytes(root/f'evaluation/{task}/truth.npy',npy((4,),test_y))
        report={'overlap_rows':1,'nonoverlap_rows':3,'arms':{}}
        for arm in audit.FAMILIES:
            npz(root/f'evaluation/{task}/{arm}-predictions.npz',prediction=((4,),pred),expected=((4,),test_y),nonoverlap=((4,),mask,'|b1'))
            report['arms'][arm]={**audit.quality(pred,test_y),'nonoverlap':audit.quality(pred[1:],test_y[1:])}
        for arm in ('fixed','centers','svc','mlp'):report['local_vs_'+arm]={'gain_points':0.,'candidate_only_correct':0,'baseline_only_correct':0,'paired_binomial_p':1.}
        qreport[task]=report
    write_json(root/'evaluation/quality.json',qreport)
    write_json(root/'evaluation/TEST_OPENING.json',{'source':snapshot(root,'evaluation'),'final_lock_sha256':hashfile(root/'models/FINAL_LOCK.json'),'optical_archive_sha256':hashfile(archive)})
    fpmodels={};fpeval={}
    for task in audit.TASKS:
        a=audit.arrays(root/f'models/{task}-mlp/weights.npz');payload=struct.pack('<4I',1,2,2,2)
        for n in ('mean','scale','w0','b0','w1','b1','w2','b2'):v=a[n]['values'];payload+=struct.pack('<'+str(len(v))+'f',*v)
        meta=b'{"labels":[0,1]}';payload+=meta
        raw=struct.pack('<8sIIIIIII',b'SPNF0001',1,2,15,3,len(meta),len(payload),zlib.crc32(payload))+payload
        p=root/f'models_fp32/{task}.sfn';write_bytes(p,raw)
        fpmodels[task]={'sha256':hashfile(p),'bytes':len(raw),'original_weights_sha256':hashfile(root/f'models/{task}-mlp/weights.npz')}
        npz(root/f'models_fp32/{task}-predictions.npz',prediction=((4,),pred),expected=((4,),test_y))
        fpeval[task]={'correct':3,'rows':4,'disagreements_with_original':0}
    write_json(root/'models_fp32/LOCK.json',{'source':snapshot(root,'float_conversion'),'models':fpmodels})
    write_json(root/'models_fp32/evaluation.json',fpeval)
    for run,seed,extra,snap in [('benchmark',2026092909,False,'first_benchmark'),('benchmark_strong',2026092912,True,'strong_benchmark'),('benchmark_final',2026092917,True,'final_benchmark')]:
        arms=[f'{f}_{l}' for l in ('original','packet','register') for f in audit.PROTO]+['local_scalar','local_direct_exp','svc','mlp','linear']
        if extra:arms+=['mlp_blas','svc_finite']
        if run=='benchmark_final':arms+=['mlp_float32']
        libraries={}
        names=['native/prototypes.so','native-packet/prototypes.so','native-register/prototypes.so','controls-native/controls.so']
        if extra:names+=['blas-native/control.so','finite-native/control.so']
        if run=='benchmark_final':names+=['float-native/control.so']
        for rel in names:
            lib=root/rel;write_bytes(lib,b'SYNTHETIC LIBRARY PLACEHOLDER\n');libraries['/irrelevant/'+rel]=hashfile(lib)
        protocol={'source':snapshot(root,snap),'tasks':list(audit.TASKS),'arms':arms,'chunks':[1,32,256],'repeats':7,'seed':seed,'libraries':libraries,'final_lock_sha256':hashfile(root/'models/FINAL_LOCK.json'),'quality_sha256':hashfile(root/'evaluation/quality.json')}
        if run=='benchmark_final':protocol['fp32_lock_sha256']=hashfile(root/'models_fp32/LOCK.json')
        write_json(root/run/'protocol.json',protocol)
        records=[];rng=random.Random(seed)
        for task in audit.TASKS:
            for repeat in range(7):
                jobs=[(c,a) for c in (1,32,256) for a in arms];rng.shuffle(jobs)
                for c,arm in jobs:records.append({'task':task,'repeat':repeat,'chunk':c,'arm':arm,'rows':4,'ns':1000000,'prediction_sha256':digest(struct.pack('<4q',*pred))})
        write_bytes(root/run/'timings.jsonl',''.join(json.dumps(r)+'\n' for r in records).encode())
        tasks={}
        for task in audit.TASKS:
            report={'us_per_row':{str(c):{a:250. for a in arms} for c in (1,32,256)},'local_over_fixed':1.,'local_gain_points':0.,'joint_gate':False,'layout_ratios':{f:1. for f in audit.PROTO},'local_over_mlp':1.,'local_over_svc':1.,'direct_exp_prediction_disagreements':0}
            if extra:report.update(local_over_mlp_blas=1.,local_over_svc_finite=1.)
            if run=='benchmark_final':report['local_over_mlp_float32']=1.
            tasks[task]=report
        write_json(root/run/'summary.json',{'tasks':tasks,'cells':len(records),'checked_predictions':4*len(records),'layout_ratio':1.,'layout_gate':False,'primary_two_task_gate':False})
    return root


@pytest.fixture(scope='module')
def clean(tmp_path_factory):return make_fixture(tmp_path_factory.mktemp('synthetic-audit'))


def test_complete_synthetic_schema(clean):
    r=audit.audit(clean)
    assert r['selection_cells']==284 and r['final_models']==24 and r['underlying_evaluation_rows']==16
    assert [r['timing'][n]['cells'] for n in ('benchmark','benchmark_strong','benchmark_final')]==[1176,1344,1428]


def mutate_json(path,fn):
    original=path.read_bytes();value=json.loads(original);fn(value);path.write_text(json.dumps(value));return original


@pytest.mark.parametrize('role,path,key',[
    ('initial source','selection/LOCK.json','source'),
    ('amendment source','selection_optical/LOCK.json','source'),
    ('candidate file','selection/job-000/record.json','files'),
    ('timed library','benchmark_final/protocol.json','libraries'),
])
def test_empty_binding_rejected(clean,role,path,key):
    p=clean/path;old=mutate_json(p,lambda v:v.update({key:{}}))
    # Keep the outer identity honest where another record intentionally binds it.
    changed=[]
    if path=='selection/LOCK.json':
        ap=clean/'selection_optical/LOCK.json';changed.append((ap,mutate_json(ap,lambda v:v.update(original_selection_sha256=hashfile(p)))))
    try:
        with pytest.raises(ValueError):audit.audit(clean)
    finally:
        p.write_bytes(old)
        for ap,raw in changed:ap.write_bytes(raw)


def test_duplicate_final_fit_rejected(clean):
    p=clean/'models/FINAL_LOCK.json';old=mutate_json(p,lambda v:v['fits'].__setitem__(1,copy.deepcopy(v['fits'][0])))
    updates=[]
    for rel in ['evaluation/TEST_OPENING.json','benchmark/protocol.json','benchmark_strong/protocol.json','benchmark_final/protocol.json']:
        pp=clean/rel;updates.append((pp,mutate_json(pp,lambda v:v.update(final_lock_sha256=hashfile(p)))))
    try:
        with pytest.raises(ValueError):audit.audit(clean)
    finally:
        p.write_bytes(old)
        for pp,raw in updates:pp.write_bytes(raw)


@pytest.mark.parametrize('text',['{"x":1e999}','{"x":NaN}','{"x":1,"x":2}'])
def test_invalid_json_numbers_and_keys(text):
    with pytest.raises(ValueError):audit.decode(text)


@pytest.mark.parametrize('raw',[b'',b'\x93NUMPY',b'\x93NUMPY\x01',b'\x93NUMPY\x01\x00',b'\x93NUMPY\x01\x00\x00\x00'])
def test_truncated_npy_is_clean_failure(raw):
    with pytest.raises(ValueError):audit.npy(raw)


def test_summary_only_is_not_full_evidence(tmp_path):
    write_json(tmp_path/'quality.json',{'status':'PASS'})
    with pytest.raises((ValueError,FileNotFoundError)):audit.audit(tmp_path)

PSPEC=importlib.util.spec_from_file_location('prototype_package_tested',Path(__file__).with_name('evidence_package.py'))
package=importlib.util.module_from_spec(PSPEC);PSPEC.loader.exec_module(package)


def test_source_bound_full_synthetic_acceptance(clean,tmp_path):
    manifest=tmp_path/'manifest.json';out=tmp_path/'acceptance.json'
    sealed=package.seal(clean,manifest)
    report=package.checked_audit(clean,manifest,out)
    assert report['status']=='PASS' and report['checked_before_and_after']
    assert report['evidence_manifest_sha256']==sealed['manifest_sha256']
    assert report['auditor_sha256']==hashfile(Path(audit.__file__))
    assert report['verification_source']['audit.py']['sha256']==report['auditor_sha256']
    with pytest.raises(FileExistsError):package.checked_audit(clean,manifest,out)


@pytest.mark.parametrize('change',['added','deleted','changed'])
def test_manifest_detects_tree_changes(tmp_path,change):
    root=tmp_path/'evidence';write_bytes(root/'file.bin',b'actual input')
    manifest=tmp_path/'manifest.json';package.seal(root,manifest)
    if change=='added':write_bytes(root/'new.bin',b'other input')
    elif change=='deleted':(root/'file.bin').unlink()
    else:(root/'file.bin').write_bytes(b'changed input')
    with pytest.raises(ValueError):package.verify(root,manifest)


@pytest.mark.parametrize('destination',['manifest','receipt'])
def test_outputs_cannot_live_inside_evidence(clean,destination):
    if destination=='manifest':
        with pytest.raises(ValueError):package.seal(clean,clean/'manifest.json')
    else:
        with pytest.raises(ValueError):package.checked_audit(clean,clean/'missing.json',clean/'receipt.json')


def test_symlink_not_followed(tmp_path):
    evidence=tmp_path/'evidence';evidence.mkdir();target=tmp_path/'outside';target.write_text('secret')
    (evidence/'link').symlink_to(target)
    with pytest.raises(ValueError):package.inventory(evidence)


def test_empty_manifest_rejected(tmp_path):
    root=tmp_path/'evidence';write_bytes(root/'example',b'example')
    manifest=tmp_path/'manifest.json';write_json(manifest,{'format':package.FORMAT,'files':{}})
    with pytest.raises(ValueError):package.verify(root,manifest)


def test_manipulation_during_audit_cannot_publish(clean,tmp_path,monkeypatch):
    manifest=tmp_path/'manifest.json';package.seal(clean,manifest)
    actual_load=package.load_auditor
    def altered(path):
        module=actual_load(path);original=module.audit
        def changed(root):
            result=original(root);write_bytes(Path(root)/'unrecorded-file',b'after audit');return result
        module.audit=changed;return module
    monkeypatch.setattr(package,'load_auditor',altered)
    out=tmp_path/'acceptance.json'
    try:
        with pytest.raises(ValueError):package.checked_audit(clean,manifest,out)
        assert not out.exists()
    finally:(clean/'unrecorded-file').unlink(missing_ok=True)


def test_unbound_legacy_receipt_cannot_publish(clean,tmp_path,monkeypatch):
    manifest=tmp_path/'manifest.json';package.seal(clean,manifest)
    class Legacy:
        def audit(self,root):return {'status':'PASS'}
    monkeypatch.setattr(package,'load_auditor',lambda path:Legacy())
    out=tmp_path/'receipt.json'
    with pytest.raises(ValueError,match='bound'):package.checked_audit(clean,manifest,out)
    assert not out.exists()


def test_isolated_standard_library_audit(clean,tmp_path):
    import subprocess,sys
    manifest=tmp_path/'manifest.json';package.seal(clean,manifest)
    out=tmp_path/'receipt.json'
    p=subprocess.run([sys.executable,'-I','-S',str(Path(package.__file__)),'audit','--root',str(clean),'--manifest',str(manifest),'--out',str(out)],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert json.loads(out.read_text())['status']=='PASS'


def test_auditor_loader_ignores_stale_bytecode(tmp_path):
    import os, py_compile
    p=tmp_path/'auditor.py'
    p.write_text("MARKER = 'old-value'\n")
    previous=p.stat()
    py_compile.compile(str(p),doraise=True)
    p.write_text("MARKER = 'new-value'\n")
    os.utime(p,ns=(previous.st_atime_ns,previous.st_mtime_ns))
    module=package.load_auditor(p)
    assert module.MARKER=='new-value'
    assert module._executed_source_sha256==hashfile(p)
