"""Fixed complete-call matrix: compact abstention is not full-coverage prediction.

All original models and implementations are pinned before timing. Source
verification, loading and compilation are outside warm calls. Input validation,
uint8 conversion, routing, certification, refinement, fallback and new lists count.
"""
from __future__ import annotations
from array import array
from contextlib import ExitStack
import argparse,hashlib,json,os,platform,random,statistics,struct,sys,time
from pathlib import Path
from .session import TreeSession,VerifiedCompact
from .export_control import ExportSession
TASKS=('letter','pendigits','satellite','optdigits')
ARMS=('official_128','official_1210','export_cpp','int8_scalar_end','int16_scalar_end',
      'int8_tiled_end','int16_tiled_end','int8_scalar_early','int16_scalar_early',
      'int8_tiled_early','int16_tiled_early','refine_8_16','refine_8_16_official',
      'refine_16_official','refine_8_16_official_early')
CHUNKS=(1,32,256);REPEATS=7;SEED=2026092921

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(indices):return hashlib.sha256(struct.pack('<'+str(len(indices))+'i',*indices)).hexdigest()
def sources():
    root=Path(__file__).parent
    return {p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.suffix in ('.py','.cpp')}

def main(a):
    a.out.mkdir(parents=True,exist_ok=False)
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    lock=json.loads((a.models/'MODEL_LOCK.json').read_text());quality=json.loads((a.evaluation/'QUALITY.json').read_text())
    for name,h in lock['files'].items():
        if sha(a.models/name)!=h:raise ValueError('frozen model modified')
    protocol={'tasks':TASKS,'arms':ARMS,'chunks':CHUNKS,'repeats':REPEATS,'seed':SEED,'source':sources(),
      'model_lock_sha256':sha(a.models/'MODEL_LOCK.json'),'quality_sha256':sha(a.evaluation/'QUALITY.json'),
      'library_sha256':sha(a.library),'official':{v:sha(a.upstream/f'libcatboostmodel-linux-x86_64-{v}.so') for v in ('1.2.8','1.2.10')},
      'compact':{p.name:sha(p) for p in a.compiled.glob('*.sct')},
      'cpu':next(s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),
      'python':sys.version,'platform':platform.platform(),'affinity':sorted(os.sched_getaffinity(0)),
      'scope':'whole input-to-new-index calls; all fallback work included; creation/verification/compilation excluded; four exposed fixed compatibility models'}
    (a.out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
    randomizer=random.Random(SEED);records=[];validations={}
    with (a.out/'rows.jsonl').open('x') as log:
      for task in TASKS:
        info=quality[task];d=info['features'];q=array('B',(a.evaluation/task/'input.u8').read_bytes());n=info['rows']
        source_indices=list(struct.unpack('<'+str(n)+'i',(a.evaluation/task/'indices.i32').read_bytes()))
        proofs={b:VerifiedCompact.from_files(a.models/task/'model.json',a.compiled/f'{task}-{b}.sct') for b in (8,16)}
        with ExitStack() as stack:
            engines={}
            for v,name in (('1.2.8','official_128'),('1.2.10','official_1210')):
                engines[name]=stack.enter_context(TreeSession(a.library,official_model=a.models/task/'model.cbm',official_library=a.upstream/f'libcatboostmodel-linux-x86_64-{v}.so',features=d,maximum=info['maximum'],classes=len(info['classes'])))
            eng={b:stack.enter_context(TreeSession(a.library,first=proofs[b])) for b in (8,16)}
            mixed=stack.enter_context(TreeSession(a.library,first=proofs[8],second=proofs[16],official_model=a.models/task/'model.cbm',official_library=a.upstream/'libcatboostmodel-linux-x86_64-1.2.10.so'))
            high=stack.enter_context(TreeSession(a.library,first=proofs[16],official_model=a.models/task/'model.cbm',official_library=a.upstream/'libcatboostmodel-linux-x86_64-1.2.10.so'))
            export=ExportSession(a.replay/task/'cpp')
            def invoke(arm,chunk):
                out=[]
                for start in range(0,n,chunk):
                    view=memoryview(q)[start*d:min(n,start+chunk)*d]
                    try:
                        if arm in engines:out.extend(engines[arm].predict_buffer(view))
                        elif arm=='export_cpp':out.extend(export.predict_buffer(view))
                        elif arm.startswith('int'):
                            precision,mode,stop=arm.split('_');out.extend(eng[int(precision[3:])].predict_buffer(view,mode=mode,checkpoint=16 if stop=='early' else 0))
                        elif arm=='refine_16_official':out.extend(high.predict_buffer(view,fallback=True))
                        else:out.extend(mixed.predict_buffer(view,refine=True,fallback='official' in arm,checkpoint=16 if arm.endswith('early') else 0))
                    finally:view.release()
                return out
            expected={arm:invoke(arm,256) for arm in ARMS}
            for arm,values in expected.items():
                if any(p!=-1 and p!=s for p,s in zip(values,source_indices)):raise ValueError('false certificate or wrong fallback')
                if ('official' in arm or arm=='export_cpp') and values!=source_indices:raise ValueError('incomplete fallback')
                (a.out/f'{task}-{arm}.indices').write_bytes(struct.pack('<'+str(n)+'i',*values))
            validations[task]={arm:{'unresolved':values.count(-1),'digest':digest(values),'rows':n} for arm,values in expected.items()}
            for repeat in range(REPEATS):
                jobs=[(c,arm) for c in CHUNKS for arm in ARMS];randomizer.shuffle(jobs)
                for chunk,arm in jobs:
                    begin=time.perf_counter_ns();values=invoke(arm,chunk);elapsed=time.perf_counter_ns()-begin
                    if values!=expected[arm]:raise ValueError('timed output changed')
                    record={'task':task,'repeat':repeat,'chunk':chunk,'arm':arm,'ns':elapsed,'rows':n,
                      'prediction_sha256':digest(values),'unresolved':values.count(-1)}
                    log.write(json.dumps(record)+'\n');log.flush();records.append(record)
                print(task,repeat,'complete',flush=True)
    summaries={}
    for task in TASKS:
        medians={str(c):{arm:statistics.median(r['ns'] for r in records if r['task']==task and r['chunk']==c and r['arm']==arm)/quality[task]['rows']/1000 for arm in ARMS} for c in CHUNKS}
        summaries[task]={'us_per_row':medians,'outputs':validations[task],
          'refine8_over_official1210_batch32':medians['32']['refine_8_16_official']/medians['32']['official_1210'],
          'refine16_over_official1210_batch32':medians['32']['refine_16_official']/medians['32']['official_1210']}
    report={'status':'COMPLETE','cells':len(records),'underlying_rows':sum(x['rows'] for x in quality.values()),'tasks':summaries,
       'claims':'ratios against actual native full-coverage control; compact-only missing outputs remain explicit; no changed classifier or independent accuracy study'}
    (a.out/'SUMMARY.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('models','compiled','evaluation','replay','upstream','library','out'):p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
