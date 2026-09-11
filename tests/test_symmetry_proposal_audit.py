"""The reporting layer may not convert incomplete or inconsistent runs into wins."""
from copy import deepcopy

import pytest

from scripts.audit_symmetry_proposals import ARMS, summarize


def fixture_rows():
    rows = []
    for eid in ('a', 'b'):
        for arm in ARMS:
            valid = eid == 'a' or arm in ('dihedral_32','symbolic')
            answer = [1,2] if valid else [0,0]
            for r in range(3):
                rows.append({'family':'maze','core_seed':1701,'example_id':eid,
                    'arm':arm,'round':r,'latency_ms':float(r+1),'valid':valid,
                    'answer':answer.copy(),'work':{'valid':valid,'transitions':0 if arm=='symbolic' else 4}})
    return rows


def test_complete_paired_summary_counts_unique_examples_not_rounds():
    report = summarize(fixture_rows())['maze']
    assert report['unique_examples'] == 2 and report['rounds'] == 3
    assert report['arms']['dihedral_32']['model_example_pairs'] == 2
    assert report['arms']['identity_4']['valid_answers'] == 1
    assert report['arms']['dihedral_32']['valid_answers'] == 2
    assert report['arms']['dihedral_32']['new_solves_vs_identity_32'] == 1
    assert report['arms']['dihedral_32']['mean_ms_after_round_medians'] == 2.0
    assert report['arms']['dihedral_32']['by_model_seed'] == {'1701':{'examples':2,'valid':2}}


@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf'), True, '2'])
def test_nonpositive_nonfinite_or_invalid_time_cannot_pass(value):
    rows = fixture_rows(); rows[0]['latency_ms'] = value
    with pytest.raises(ValueError): summarize(rows)


def test_missing_duplicate_and_unpaired_observations_are_rejected():
    rows = fixture_rows()
    for changed in (rows[:-1], rows+[deepcopy(rows[0])], [r for r in rows if r['arm']!='symbolic']):
        with pytest.raises(ValueError): summarize(changed)


@pytest.mark.parametrize('field,value', [('answer',[4,3]),('valid',False),('work',{'valid':True,'transitions':7})])
def test_repeated_outcomes_and_work_must_match(field,value):
    rows = fixture_rows(); rows[0][field] = value
    with pytest.raises(ValueError): summarize(rows)


def test_restart_control_cannot_hide_changed_numerical_execution():
    rows = fixture_rows()
    for r in rows:
        if r['example_id']=='b' and r['arm']=='repeat_identity_32':
            r['answer']=[1,2]; r['valid']=r['work']['valid']=True
    with pytest.raises(ValueError, match='restarts'): summarize(rows)


def test_losing_previously_verified_answer_is_failure():
    rows = fixture_rows()
    for r in rows:
        if r['example_id']=='a' and r['arm']=='dihedral_32':
            r['answer']=[0,0]; r['valid']=r['work']['valid']=False
    with pytest.raises(ValueError,match='lost'): summarize(rows)


@pytest.mark.parametrize('rounds', [0,-1,True,1.5])
def test_bad_round_count_is_not_zero_work_success(rounds):
    with pytest.raises(ValueError): summarize(fixture_rows(),rounds=rounds)


def test_empty_run_is_not_success():
    with pytest.raises(ValueError): summarize([])


def test_complete_raw_log_is_sealed_to_a_separate_exact_compressed_mirror(tmp_path):
    import gzip,hashlib,json
    from scripts.audit_symmetry_proposals import seal_rows
    rows=fixture_rows()
    raw=''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows).encode()
    (tmp_path/'rows.jsonl').write_bytes(raw)
    assert seal_rows(tmp_path,rows)==hashlib.sha256(raw).hexdigest()
    assert gzip.decompress((tmp_path/'rows.jsonl.gz').read_bytes())==raw
    with pytest.raises(FileExistsError):seal_rows(tmp_path,rows)


def test_missing_or_changed_raw_observation_cannot_be_declared_complete(tmp_path):
    import json
    from scripts.audit_symmetry_proposals import seal_rows
    rows=fixture_rows()
    raw=''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows[:-1]).encode()
    (tmp_path/'rows.jsonl').write_bytes(raw)
    with pytest.raises(ValueError,match='incomplete'):seal_rows(tmp_path,rows)
    assert not (tmp_path/'rows.jsonl.gz').exists()
    with pytest.raises(ValueError,match='empty'):seal_rows(tmp_path,[])
