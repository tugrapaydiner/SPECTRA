"""Independent witness/inventory checks plus optional exact native replay/refits.

Integrity is separate from scientific success. The native reconstruction shares
its solver/model implementation; original-formula answer checks do not.
"""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path
import tempfile
import numpy as np
from data.cnf import CNF
from .data import cases,assignment
from .runtime import State,WORK,F,W
from . import learn,rank
from .experiment import sha,verify as verify_original,model_bank
from .followup import VARIANTS,CONTROL,own_sources,check_binding,banks,run_variant,models_hash,aggregate,paired


def records(path):
    def pairs(items):
        out={}
        for k,v in items:
            if k in out:raise ValueError('duplicate JSON key')
            out[k]=v
        return out
    def bad(x):raise ValueError('nonfinite JSON value')
    with gzip.open(path,'rt') as f:
        for line in f:
            row=json.loads(line,object_pairs_hook=pairs,parse_constant=bad)
            if line!=json.dumps(row,sort_keys=True,allow_nan=False)+'\n':raise ValueError('noncanonical JSONL')
            yield row


def same_work(expected,actual):
    left={k:v for k,v in expected.items() if k not in ('native_ns','witness')}
    right={k:v for k,v in actual.items() if k not in ('native_ns','witness')}
    if left!=right or tuple(expected['witness'])!=tuple(actual['witness']):raise ValueError('native inference/work reconstruction differs')


def original_truth(formula,result):
    witness=result['witness']
    if type(witness) not in (tuple,list) or any(type(v) is not bool for v in witness):raise ValueError('nonboolean witness')
    witness=tuple(witness);actual=formula.satisfied(witness)
    if type(result['native_status']) is not int or actual!=(result['native_status']==1):raise ValueError('false native status')
    if result['unsatisfied']!=len(formula.violated(witness)):raise ValueError('wrong residual count')
    for k in WORK:
        if type(result[k]) is not int or result[k]<0:raise ValueError('invalid work/time counter')
    return actual


def hash_inventory(root):
    hashes=json.loads((root/'SHA256.json').read_text())
    files={str(p.relative_to(root)):p for p in root.rglob('*') if p.is_file() and p.name!='SHA256.json'}
    if set(files)!=set(hashes):raise ValueError('missing/extra evidence files')
    for n,p in files.items():
        if p.is_symlink() or sha(p.read_bytes())!=hashes[n]:raise ValueError('corrupt/nonregular evidence file')


def training(original,cfg,replay):
    train=original/'training';items=json.loads((train/'cases.json').read_text())
    if items!=cases(cfg,'train'):raise ValueError('training formulas do not regenerate')
    with np.load(train/'labels.npz',allow_pickle=False) as data:
        x=data['features'];y=data['targets'];ci=data['case_index']
    if x.shape!=(len(y),F) or ci.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('invalid label arrays')
    snapshots=list(records(train/'snapshots.jsonl.gz'));cf=list(records(train/'counterfactuals.jsonl.gz'))
    expected=[(i,d) for i in range(len(items)) for d in cfg['snapshot_moves']]
    if [(s['case_index'],s['depth']) for s in snapshots]!=expected:raise ValueError('snapshot inventory mismatch')
    candidate=0;offset=0;checked=0;native_replays=0;features_reconstructed=0
    for s in snapshots:
        c=items[s['case_index']];p=CNF.from_record(c['formula']);source=s['source']
        solved=original_truth(p,source);checked+=1
        if s['candidate_start']!=candidate:raise ValueError('candidate offset mismatch')
        seed=c['seed']^0x6e624eb7
        if replay:
            with State(p,assignment(p.nvars,seed)) as state:
                actual=state.run(seed+1,moves=s['depth'],cb=cfg['break_exponent']);same_work(source,actual);native_replays+=1
                if not solved:
                    ps,fs=state.pool(seed+s['depth']+2)
                    if ps.tolist()!=s['pairs'] or fs.shape!=(s['candidate_count'],F) or not np.allclose(fs,x[candidate:candidate+len(ps)],atol=1e-12,rtol=1e-12):raise ValueError('candidate or feature replay differs')
                    features_reconstructed+=len(ps)
        if solved:
            if s['candidate_count']!=0:raise ValueError('solved snapshot must not invent candidate labels')
            continue
        if s['candidate_count']!=len(s['pairs']) or len(set(map(tuple,s['pairs'])))!=len(s['pairs']):raise ValueError('invalid candidate pool inventory')
        for pair in s['pairs']:
            patch=tuple(int(v) for v in pair if v>=0)
            if not 1<=len(patch)<=2 or len(set(patch))!=len(patch) or any(not 0<=v<p.nvars for v in patch):raise ValueError('invalid recorded patch')
            a=tuple(source['witness']);b=tuple(not v if i in patch else v for i,v in enumerate(a))
            before=[any(a[abs(l)-1]==(l>0) for l in row) for row in p.clauses]
            after=[any(b[abs(l)-1]==(l>0) for l in row) for row in p.clauses]
            make=sum(not aa and bb for aa,bb in zip(before,after));br=sum(aa and not bb for aa,bb in zip(before,after))
            if tuple(x[candidate,[5,6]]*8)!=(make,br) or ci[candidate]!=s['case_index']:raise ValueError('false exact feature or case label')
            successes=0
            for rep in range(cfg['teacher_repeats']):
                r=cf[offset];offset+=1
                expected_seed=seed+s['depth']+100+rep
                if r['candidate']!=candidate or r['repeat']!=rep or r['seed']!=expected_seed:raise ValueError('counterfactual inventory/seed mismatch')
                successes+=original_truth(p,r);checked+=1
                if replay:
                    with State(p,a) as state:
                        state.patch(patch);actual=state.run(expected_seed,moves=cfg['teacher_moves'],cb=cfg['break_exponent'])
                        record={k:v for k,v in r.items() if k not in ('candidate','repeat','seed')}
                        same_work(record,actual);native_replays+=1
            if y[candidate]!=successes/cfg['teacher_repeats']:raise ValueError('teacher labels differ from witnessed outcomes')
            candidate+=1
    if candidate!=len(y) or offset!=len(cf):raise ValueError('extra/missing training observations')
    return {'answer_checks':checked,'candidate_labels':candidate,'native_replays':native_replays,'feature_rows_reconstructed':features_reconstructed}


def validate_row(c,r,cfg,*,work):
    p=CNF.from_record(c['formula']);actual=original_truth(p,dict(r['work'],witness=r['witness']))
    if type(r['valid']) is not bool or type(r['within_budget']) is not bool or type(r['work_only']) is not bool or r['work_only']!=work:raise ValueError('nonboolean outcome/type mismatch')
    if actual!=r['valid'] or r['within_budget']!=(actual and (work or r['complete_ns']<=r['budget_ms']*1e6)):raise ValueError('false witness/budget credit')
    for k in ('complete_ns','setup_ns','native_and_bridge_ns','independent_check_ns','overrun_ns','round','model_seed','search_seed'):
        if type(r[k]) is not int or r[k]<0:raise ValueError('invalid timing/index')
    if r['complete_ns']<=0 or r['work']['native_ns']>r['complete_ns'] or r['setup_ns']+r['native_and_bridge_ns']+r['independent_check_ns']>r['complete_ns']:raise ValueError('inconsistent timing scopes')
    if r['ordered_sha256']!=c['ordered_sha256']:raise ValueError('input hash mismatch')
    if type(r['budget_ms']) is not int or r['budget_ms']!=(0 if work else r['budget_ms']) or (not work and r['budget_ms'] not in cfg['wall_budget_ms']):raise ValueError('undeclared budget')
    if r['overrun_ns']!=(0 if work else max(0,r['complete_ns']-r['budget_ms']*1000000)):raise ValueError('incorrect deadline overrun')
    arm,interval,model=VARIANTS[r['variant']]
    if (r['arm'],r['patch_interval'],r['model_family'])!=(arm,interval,model):raise ValueError('variant does not match runtime configuration')


def followup(original,out,cfg,replay):
    hash_inventory(out);bank=banks(original,out,cfg);checks=0;replays=0
    all_items=cases(cfg,'train')+cases(cfg,'validation')+cases(cfg,'development')
    if len({c['normalized_sha256'] for c in all_items})!=len(all_items):raise ValueError('duplicate normalized formula across splits')
    for split in ('validation','development'):
        dest=out/split;items=json.loads((dest/'cases.json').read_text())
        if items!=cases(cfg,split):raise ValueError('follow-up inputs do not regenerate')
        lookup={c['id']:c for c in items};rows=list(records(dest/'rows.jsonl.gz'))
        expected={(c['id'],v,m,s,b,r) for c in items for v in VARIANTS for m in cfg['model_seeds'] for s in cfg['search_seeds'] for b in cfg['wall_budget_ms'] for r in range(cfg['timing_rounds'])}
        found=[(r['case_id'],r['variant'],r['model_seed'],r['search_seed'],r['budget_ms'],r['round']) for r in rows]
        if len(set(found))!=len(found) or set(found)!=expected:raise ValueError('incomplete/extra/duplicated follow-up timings')
        schedule=json.loads((dest/'schedule.json').read_text())
        observed=[(r['round'],next(i for i,c in enumerate(items) if c['id']==r['case_id']),r['model_seed'],r['search_seed'],r['budget_ms'],r['variant']) for r in rows]
        if observed!=list(map(tuple,schedule)):raise ValueError('timing execution order mismatch')
        for r in rows:validate_row(lookup[r['case_id']],r,cfg,work=False);checks+=1
        report=aggregate(cfg,items,rows);stored=json.loads((dest/'summary.json').read_text());stored.pop('execution_ns')
        if report!=stored:raise ValueError('follow-up summary differs')
        if split=='validation':
            selection={}
            for family in cfg['families']:
                for n in cfg['eval_sizes']:
                    eligible=[s for s in report['strata'] if s['family']==family and s['nvars']==n and s['budget_ms']==cfg['primary_wall_budget_ms'] and s['variant'] in CONTROL]
                    best=max(eligible,key=lambda s:(s['success_rate'],-s['mean_complete_ns'],-CONTROL.index(s['variant'])))
                    selection[f'{family}:{n}']=best['variant']
            if selection!=json.loads((dest/'selection.json').read_text()):raise ValueError('validation selection differs')
        else:
            lock=json.loads((dest/'pre_execution_lock.json').read_text())
            if lock['sources']!=own_sources() or lock['models']!=models_hash(original,out) or lock['selection']!=selection:raise ValueError('source/model/selection lock mismatch')
            if paired(cfg,items,rows,selection)!=json.loads((dest/'paired_analysis.json').read_text()):raise ValueError('paired follow-up analysis differs')
        work_rows=list(records(dest/'work_rows.jsonl.gz'))
        keys=[(r['case_id'],r['variant'],r['model_seed'],r['search_seed']) for r in work_rows]
        wanted={(c['id'],v,m,s) for c in items for v in VARIANTS for m in cfg['model_seeds'] for s in cfg['search_seeds']}
        if set(keys)!=wanted or len(keys)!=len(set(keys)):raise ValueError('fixed-work inventory differs')
        for r in work_rows:
            validate_row(lookup[r['case_id']],r,cfg,work=True);checks+=1
            if replay:
                actual=run_variant(lookup[r['case_id']],r['variant'],r['model_seed'],r['search_seed'],0,cfg,bank,True)
                same_work(dict(r['work'],witness=r['witness']),dict(actual['work'],witness=actual['witness']));replays+=1
    return {'answer_checks':checks,'exact_native_work_replays':replays}


def refit(original,out,cfg):
    errors={}
    with tempfile.TemporaryDirectory(prefix='spectra-rank-refit-') as temp:
        temp=Path(temp)
        for name,trainer,stored in [('bce',learn.train,original/'models'),('rank',rank.train,out/'models')]:
            trainer(cfg,original/'training',temp/name)
            for file in sorted((temp/name).glob('*_*.json')):
                actual=json.loads(file.read_text());expected=json.loads((stored/file.name).read_text())
                error=float(np.max(np.abs(np.array(actual['weights'])-np.array(expected['weights']))))
                if error>1e-10:raise ValueError('refit coefficient difference exceeds 1e-10')
                errors[name+'/'+file.name]=error
    return errors


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--original',type=Path,required=True);p.add_argument('--followup',type=Path,required=True);p.add_argument('--replay',action='store_true');p.add_argument('--refit',action='store_true');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('refusing to overwrite verification receipt')
    cfg=check_binding(a.original,a.followup)
    original=verify_original(a.original,a.replay)
    train=training(a.original,cfg,a.replay)
    second=followup(a.original,a.followup,cfg,a.replay)
    report={'integrity':'PASS','original_evaluation':original,'training':train,'followup':second,
       'refit_errors':refit(a.original,a.followup,cfg) if a.refit else None,
       'scientific_quality_gate':json.loads((a.followup/'development/paired_analysis.json').read_text())['quality_gate'],
       'scope':'answer semantics independently checked; native reconstructions share the solver; fixed-work replays do not refresh timing claims; no new confirmation'}
    a.out.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
