"""Complete frozen-model replay with independent oracle and two native controls.

The four source models are newly fitted fixed compatibility workloads, not recovered
old weights or a claim of superior learning. Every unresolved compact row is retained.
"""
from __future__ import annotations
import argparse,hashlib,json,random,struct,time
from contextlib import ExitStack
from pathlib import Path
import numpy as np
from . import certificate_oracle as co
from .session import TreeSession
from .controls import CatBoostSession

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(root,library,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    cm=json.loads((root/'compiled/COMPILED_LOCK.json').read_text())
    for n,h in cm['files'].items():
        if sha(root/'compiled'/n)!=h:raise ValueError('compiled artifact changed')
    report={};total_oracle=0
    for task in ('letter','pendigits','satellite','optdigits'):
        data=np.load(root/'source_evaluation'/f'{task}.npz');x=data['q'];idx=data['indices'];orig=data['scores'];raw=(root/'models'/task/'source.json').read_bytes()
        fit=json.loads((root/'models'/task/'fit.json').read_text());D=fit['maximum'];n=len(x);results={};saved={}
        with ExitStack() as stack:
            engines={name:stack.enter_context(TreeSession(root/'compiled'/task/f'{name}.sct',library,expected_sha256=sha(root/'compiled'/task/f'{name}.sct'))) for name in ('q8','q16','full')}
            controls={name:stack.enter_context(CatBoostSession(root/'models'/task/'source.cbm',path,D,expected_sha256=sha(root/'models'/task/'source.cbm'))) for name,path in (('catboost_native',root/'catboost_native/control.so'),('catboost_export',root/'exports'/task/'control.so'))}
            # Original export may round decimal leaf literals differently from JSON.
            # We require source predictions here and separately expose score differences.
            full=np.frombuffer(engines['full'].scores(x),dtype='<f8').reshape(orig.shape).copy();saved['full_scores']=full
            if engines['full'].predict_buffer(x)!=idx.tolist():raise ValueError('source JSON evaluator disagrees')
            for name,w in controls.items():
                p=w.predict_buffer(x);scores=np.frombuffer(w.scores(x),dtype='<f8').reshape(orig.shape).copy();saved[name+'_scores']=scores
                if p!=idx.tolist():raise ValueError(name+' class mismatch')
                results[name]={'matched_rows':n,'score_max_abs_from_python':float(np.max(np.abs(scores-orig))),
                               'score_max_abs_from_source_json':float(np.max(np.abs(scores-full))),
                               'bit_equal_source_scores':int(np.sum(scores.view('u8')==full.view('u8'))),'score_values':scores.size}
            # A fixed deterministic subset, not chosen by confidence/label correctness.
            subset=sorted(random.Random(20260929).sample(range(n),min(48,n)))
            for bits in (8,16):
                name='q'+str(bits);model=co.loads((root/'compiled'/task/(name+'.json')).read_bytes());oracle=co.CertifiedOracle(raw,model)
                for cp in (0,16):
                    z=engines[name].inspect_buffer(x,checkpoint=cp);p=np.asarray(z['indices']);steps=np.asarray(z['trees_evaluated']);approx=np.asarray(z['approximate_indices']);key=f'{name}_cp{cp}'
                    if np.any((p>=0)&(p!=idx)):raise ValueError('false accepted certificate')
                    for i in subset:
                        check=oracle.predict(x[i].tolist(),checkpoint=cp)
                        if p[i]!=(-1 if check['class_index'] is None else check['class_index']) or steps[i]!=check['trees_evaluated']:raise ValueError('exact oracle decision/work mismatch')
                        total_oracle+=1
                    saved[key+'_indices']=p;saved[key+'_steps']=steps;saved[key+'_approx']=approx
                    results[key]={'certified':int(np.sum(p>=0)),'unresolved':int(np.sum(p<0)),'accepted_disagreements':0,
                        'mean_trees':float(np.mean(steps)),'min_trees':int(steps.min()),'max_trees':int(steps.max()),
                        'naive_wrong_on_completed':int(np.sum((approx>=0)&(approx!=idx))),
                        'source_info':engines[name].info,'oracle_checked_rows':len(subset)}
                for cp in (0,16):
                    z=engines[name].hybrid(engines['full'],x,checkpoint=cp,inspect=True)
                    if z['indices']!=idx.tolist():raise ValueError('hybrid full fidelity')
                    key=f'{name}_hybrid{cp}';saved[key+'_indices']=z['indices'];saved[key+'_fallback']=z['fallback']
                    results[key]={'matched_rows':n,'fallback_rows':sum(z['fallback']),'mean_total_trees':float(np.mean(z['trees_evaluated']))}
            z=engines['q8'].hybrid(engines['q16'],x,inspect=True);p=np.asarray(z['indices'])
            if np.any((p>=0)&(p!=idx)):raise ValueError('compact cascade false acceptance')
            saved['cascade_indices']=p;results['q8_q16_cascade']={'certified':int(np.sum(p>=0)),'unresolved':int(np.sum(p<0)),'second_stage_rows':sum(z['fallback'])}
            # Exact-rational per-row enclosures for actual native-source scores on
            # the same fixed rows: compare actual doubles with exact rational sums.
            source=oracle.source;bounds=co._roundoff_bounds(source);maximum_fraction=0.
            for i in subset:
                exact=co.source_scores(source,x[i].tolist(),exact=True)
                for scores in (orig,full,saved['catboost_native_scores'],saved['catboost_export_scores']):
                    for c in range(source.classes):
                        error=abs(co.F.from_float(float(scores[i,c]))-exact[c]);bound=bounds[c]*source.scale
                        if error>bound:raise ValueError('source arithmetic contract contradicted by observed scores')
                        maximum_fraction=max(maximum_fraction,float(error/bound) if bound else 0.)
            results['source_roundoff_sample']={'rows':len(subset),'backends':4,'max_fraction_of_bound':maximum_fraction}
        saved.update(q=x,source_indices=idx,truth=data['y'],source_scores=orig)
        np.savez_compressed(out/(task+'.npz'),**saved)
        report[task]={'rows':n,'source_accuracy_correct':int(np.sum(data['prediction']==data['y'])),'source_info':fit,'results':results,
                      'source_json_sha256':co.sha(raw),'output_sha256':sha(out/(task+'.npz'))}
        print(task,results['q16_cp0'],flush=True)
    summary={'status':'PASS','tasks':report,'exact_oracle_queries_checked':total_oracle,
             'underlying_rows':sum(r['rows'] for r in report.values()),'native_library_sha256':sha(library),
             'script_sha256':sha(__file__),'compiled_lock_sha256':sha(root/'compiled/COMPILED_LOCK.json'),
             'scope':'same-source class agreement with unresolved rows retained; sampled rational roundoff checks do not prove an arbitrary backend implementation'}
    (out/'report.json').write_text(json.dumps(summary,indent=2));return summary
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.root,a.library,a.out)
