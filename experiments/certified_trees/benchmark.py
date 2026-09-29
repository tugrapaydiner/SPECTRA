"""Complete frozen-panel raw-code classification versus native CatBoost.

Compact-only arms may return -1. Full-coverage arms include every fallback cost;
compact-only latency is not advertised as equally complete classification.
"""
from __future__ import annotations
import argparse,hashlib,json,os,platform,random,statistics,struct,sys,time
from contextlib import ExitStack
from pathlib import Path
import numpy as np
from .session import TreeSession
from .controls import CatBoostSession
from .pipeline import RefinementSession
TASKS=('letter','pendigits','satellite','optdigits')
ARMS=('q8_end','q8_early','q16_end','q16_early','q16_scalar','q16_full_end','q16_full_early',
      'q8_q16','catboost_refine_end','catboost_refine_early','full_fp64','catboost_native','catboost_export','q16_row_major','refine_row_major','owned_refine')
BATCHES=(1,32,256);REPEATS=7;SEED=2026092920

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(values):return hashlib.sha256(struct.pack('<'+str(len(values))+'i',*values)).hexdigest()
def summarize(records,expected):
    result={}
    for t in TASKS:
        rows=len(expected[t]['catboost_native']);costs={str(b):{a:statistics.median(r['ns'] for r in records if r['task']==t and r['batch']==b and r['arm']==a)/rows/1000 for a in ARMS} for b in BATCHES}
        costs32=costs['32'];ratios={a:costs32[a]/costs32['catboost_native'] for a in ('q16_end','q16_early','catboost_refine_end','catboost_refine_early','full_fp64','catboost_export','owned_refine')}
        result[t]={'rows':rows,'us_per_row':costs,'batch32_over_native':ratios,
                   'coverage':{a:sum(v>=0 for v in expected[t][a])/rows for a in ARMS},
                   'paired_refine_end_ratios':[next(r['ns'] for r in records if (r['task'],r['repeat'],r['batch'],r['arm'])==(t,rep,32,'catboost_refine_end'))/next(r['ns'] for r in records if (r['task'],r['repeat'],r['batch'],r['arm'])==(t,rep,32,'catboost_native')) for rep in range(REPEATS)]}
    return {'status':'COMPLETE','cells':len(records),'tasks':result,'equal_task_refine_end_ratio':statistics.geometric_mean(result[t]['batch32_over_native']['catboost_refine_end'] for t in TASKS),
            'equal_task_owned_refine_ratio':statistics.geometric_mean(result[t]['batch32_over_native']['owned_refine'] for t in TASKS),
            'scope':'same frozen models, warm raw-code-to-fresh-class-index calls including required upstream fallbacks; no source-model quality increase'}

def run(root,out):
    root=Path(root).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    libs=[root/'native/trees.so',root/'catboost_refine/control.so',root/'catboost_native/control.so',root/'tiled_bench/trees.so',root/'catboost_tiled_bench/control.so',root/'catboost_owned_fixed/control.so',root/'upstream/libcatboostmodel.so']+[root/'exports'/t/'control.so' for t in TASKS]
    locks=['models/FINAL_LOCK.json','compiled/COMPILED_LOCK.json','fidelity/report.json']
    files={str(p.relative_to(root)):sha(p) for p in libs}
    for sub in ('compiled','models','source_evaluation','fidelity'):
        files.update({p.relative_to(root).as_posix():sha(p) for p in (root/sub).rglob('*') if p.is_file()})
    protocol={'tasks':TASKS,'arms':ARMS,'batches':BATCHES,'repeats':REPEATS,'seed':SEED,
              'files':files,'sources':{p.name:sha(p) for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.cpp')},
              'cpu':next(l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),
              'affinity':sorted(os.sched_getaffinity(0)),'python':sys.version,'platform':platform.platform(),
              'scope':'input validation/conversion, traversal, certificates, fallback and fresh Python integer lists included; loading, source verification, compilation and training excluded'}
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2));records=[];expected={};rng=random.Random(SEED)
    with (out/'observations.jsonl').open('x') as log:
        for t in TASKS:
            x=np.load(root/'source_evaluation'/f'{t}.npz')['q'];reference=np.load(root/'source_evaluation'/f'{t}.npz')['indices'].tolist();fit=json.loads((root/'models'/t/'fit.json').read_text())
            saved=np.load(root/'fidelity'/f'{t}.npz');e={}
            for bits in (8,16):
                for label,cp in (('end',0),('early',16)):e[f'q{bits}_{label}']=saved[f'q{bits}_cp{cp}_indices'].tolist()
            e['q16_scalar']=e['q16_end'];e['q8_q16']=saved['cascade_indices'].tolist()
            for a in ARMS:
                if a not in e:e[a]=e['q16_end'] if a=='q16_row_major' else reference
            expected[t]=e
            with ExitStack() as stack:
                models={k:stack.enter_context(TreeSession(root/'compiled'/t/f'{k}.sct',root/'tiled_bench/trees.so',expected_sha256=sha(root/'compiled'/t/f'{k}.sct'))) for k in ('q8','q16','full')}
                cbm=root/'models'/t/'source.cbm';kwargs={'expected_sha256':sha(cbm),'source_json_sha256':sha(root/'models'/t/'source.json')}
                native=stack.enter_context(CatBoostSession(cbm,root/'catboost_tiled_bench/control.so',fit['maximum'],**kwargs))
                legacy=stack.enter_context(TreeSession(root/'compiled'/t/'q16.sct',root/'native/trees.so',expected_sha256=sha(root/'compiled'/t/'q16.sct')))
                old_native=stack.enter_context(CatBoostSession(cbm,root/'catboost_refine/control.so',fit['maximum'],**kwargs))
                exported=stack.enter_context(CatBoostSession(cbm,root/'exports'/t/'control.so',fit['maximum'],**kwargs))
                owned=stack.enter_context(RefinementSession(root/'compiled'/t/'q16.sct',cbm,root/'catboost_owned_fixed/control.so',compact_sha256=sha(root/'compiled'/t/'q16.sct'),cbm_sha256=sha(cbm),source_json_sha256=sha(root/'models'/t/'source.json')))
                def call(a,batch):
                    result=[]
                    for offset in range(0,len(x),batch):
                        q=x[offset:offset+batch]
                        if a=='owned_refine':p=owned.predict_buffer(q)
                        elif a=='q16_row_major':p=legacy.predict_buffer(q)
                        elif a=='refine_row_major':p=old_native.refine(legacy,q)
                        elif a=='catboost_native':p=native.predict_buffer(q)
                        elif a=='catboost_export':p=exported.predict_buffer(q)
                        elif a=='full_fp64':p=models['full'].predict_buffer(q)
                        elif a=='catboost_refine_end':p=native.refine(models['q16'],q,checkpoint=0)
                        elif a=='catboost_refine_early':p=native.refine(models['q16'],q,checkpoint=16)
                        elif a=='q8_q16':p=models['q8'].hybrid(models['q16'],q,checkpoint=0)
                        elif a.startswith('q16_full'):p=models['q16'].hybrid(models['full'],q,checkpoint=16 if a.endswith('early') else 0)
                        else:
                            p=models[a.split('_')[0]].predict_buffer(q,checkpoint=16 if a.endswith('early') else 0,scalar=a.endswith('scalar'))
                        result.extend(p)
                    return result
                for a in ARMS:
                    if call(a,256)!=e[a]:raise ValueError('warmup fidelity '+t+'/'+a)
                for rep in range(REPEATS):
                    jobs=[(b,a) for b in BATCHES for a in ARMS];rng.shuffle(jobs)
                    for b,a in jobs:
                        start=time.perf_counter_ns();pred=call(a,b);elapsed=time.perf_counter_ns()-start
                        if pred!=e[a]:raise ValueError('timed fidelity '+t+'/'+a)
                        record={'task':t,'repeat':rep,'batch':b,'arm':a,'rows':len(pred),'ns':elapsed,'output_sha256':digest(pred)}
                        records.append(record);log.write(json.dumps(record)+'\n');log.flush()
                    print(t,rep,'complete',flush=True)
    (out/'expected.json').write_text(json.dumps(expected,separators=(',',':')))
    report=summarize(records,expected);(out/'summary.json').write_text(json.dumps(report,indent=2));return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.root,a.out)
