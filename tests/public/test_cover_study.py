"""Synthetic study-auditor contracts; these are not performance observations."""
import hashlib
import json
from pathlib import Path

import pytest
from experiments.cover_search import analyse,study


def test_quantiles_and_paired_intervals():
    assert analyse.quantile([1,2,3,4],.95)==pytest.approx(3.85)
    assert analyse.quantile([8],.95)==8
    cases=[{'id':str(i),'stratum':'one','kind':'colouring'} for i in range(4)]
    rows=[{'case':c['id'],'arm':a,'round':r,'status':'SAT_VERIFIED','elapsed_ns':t}
          for c in cases for a,t in [('cover',100),('minicard',400)] for r in range(study.ROUNDS)]
    result=analyse.comparison(cases,rows,'cover','minicard',resamples=32)
    assert result['speed_gate'] and result['problems']==4
    assert result['descriptive_95_intervals']['mean_ratio']==[.25,.25]
    rows[0]['status']='UNKNOWN';rows[0]['elapsed_ns']=1
    assert not analyse.comparison(cases,rows,'cover','minicard',resamples=32)['speed_gate']


def test_fast_failure_cannot_pass_speed_gate():
    cases=[{'id':'one','stratum':'one','kind':'colouring'}]
    rows=[{'case':'one','arm':a,'round':r,'status':'UNKNOWN','elapsed_ns':t}
          for a,t in [('cover',1),('minicard',400)] for r in range(study.ROUNDS)]
    assert not analyse.comparison(cases,rows,'cover','minicard',resamples=16)['speed_gate']


def fixture(tmp_path,monkeypatch):
    case={'id':'synthetic-fixture','stratum':'fixture','kind':'colouring','n':1,'colours':3,'edges':[]}
    cases=[case];jobs=study.schedule(cases)
    freeze={'sources':{'spectra/_native/cover_search.cpp':hashlib.sha256(b'synthetic source').hexdigest()}}
    monkeypatch.setattr(study,'check_freeze',lambda:freeze)
    monkeypatch.setattr(study,'make_cases',lambda:cases)
    (tmp_path/'native').mkdir()
    binary=b'SYNTHETIC, NOT EXECUTABLE OR PERFORMANCE EVIDENCE'
    (tmp_path/'native/spectra_cover.so').write_bytes(binary)
    (tmp_path/'native/build.json').write_text(json.dumps({'source_sha256':freeze['sources']['spectra/_native/cover_search.cpp'],
                 'library_sha256':hashlib.sha256(binary).hexdigest()}))
    for name,value in [('controls.json',{}),('environment.json',{}),('cases.json',cases),('freeze.json',freeze),('schedule.json',jobs)]:
        (tmp_path/name).write_text(json.dumps(value))
    rows=[]
    for i,(identity,arm,r) in enumerate(jobs):
        rows.append({'job':i,'case':identity,'arm':arm,'round':r,'elapsed_ns':100,'status':'SAT_VERIFIED',
                     'witness':[True,False,False],'deadline_overrun':False,
                     'details':{'nodes':1,'max_nodes':study.MAX_NODES,'max_state_bytes':study.MAX_STATE_BYTES,
                                'state_word_bytes_peak':32,'index_payload_bytes':32}})
    resources=[{'case':case['id'],'arm':arm,'cold_process_wall_ns':100,'VmHWM_KiB':100,
                'numerical_frameworks_loaded':[],'result':{'status':'SAT_VERIFIED','witness':[True,False,False]}}
               for arm in study.arms(case)]
    (tmp_path/'timings.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (tmp_path/'resources.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in resources))
    reseal(tmp_path)
    return rows


def reseal(path):
    (path/'MANIFEST.json').write_text(json.dumps({str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.rglob('*') if p.is_file() and p.name!='MANIFEST.json'}))


def test_complete_synthetic_fixture_passes(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch)
    assert len(analyse.validate(tmp_path)[1])==len(study.ARMS)*study.ROUNDS

@pytest.mark.parametrize('mutation',['missing_row','false_witness','float_time','boolean_round','false_unsat',
                                    'boolean_counter','exceeded_budget','missing_resource','wrong_hash','missing_role'])
def test_inconsistent_fixtures_refused(tmp_path,monkeypatch,mutation):
    rows=fixture(tmp_path,monkeypatch)
    target=next(r for r in rows if r['arm']=='cover')
    if mutation=='missing_row':rows.pop()
    elif mutation=='false_witness':target['witness']=[False]*3
    elif mutation=='float_time':target['elapsed_ns']=1.5
    elif mutation=='boolean_round':next(r for r in rows if r['round']==1)['round']=True
    elif mutation=='false_unsat':target['status']='UNSAT_REPORTED'
    elif mutation=='boolean_counter':target['details']['nodes']=True
    elif mutation=='exceeded_budget':target['details']['nodes']=study.MAX_NODES+1
    elif mutation=='missing_resource':(tmp_path/'resources.jsonl').write_text('')
    elif mutation=='missing_role':(tmp_path/'controls.json').unlink()
    (tmp_path/'timings.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    if mutation!='wrong_hash':reseal(tmp_path)
    else:(tmp_path/'environment.json').write_text('altered')
    with pytest.raises(ValueError):analyse.validate(tmp_path)


def test_graph_invariant_ignores_vertex_names():
    from experiments.cover_search.tasks import colouring
    c=colouring(12,3,90,planted=True)
    renamed={**c,'edges':[(11-b,11-a) for a,b in c['edges']]}
    assert study.graph_signature(c)==study.graph_signature(renamed)
    # This does not claim that equal invariants prove isomorphism.
