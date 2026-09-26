"""Independent-stdlib matrix analysis must fail on incomplete or altered evidence."""
import copy
import random
import pytest
from spectra.fp_evidence import ARMS, ROUNDS, ORDER_SEED, analyze, correct

BOARD=[1,2,3,4,3,4,1,2,2,1,4,3,4,3,2,1]


def fixture():
    cases=[{'case_id':f'ordinary:{s}:0','family':'ordinary','seed':s,'problem_id':'ordinary:0',
            'input':BOARD.copy(),'blocks':1} for s in (1401,2402)]
    rows=[];rng=random.Random(ORDER_SEED)
    for c in cases:
        for r in range(ROUNDS):
            arms=list(ARMS);rng.shuffle(arms)
            for pos,arm in enumerate(arms):
                step=0 if arm=='symbolic' else 1
                work={'executed_steps':step,'block_applications':2*step,'semantic_checks':1,
                      'checker_constructions':1,'final_semantic':True,'target_used':False,
                      'stop_reason':'symbolic_complete' if arm=='symbolic' else 'semantic_valid'}
                rows.append({'case_id':c['case_id'],'round':r,'order':pos,'arm':arm,
                             'elapsed_ns':{'python':200,'native':100,'prepared':50,'symbolic':10}[arm],
                             'answer':BOARD.copy(),'valid':True,'steps':step,'work':work})
    return cases,rows


def test_complete_analysis_and_fixed_model_problem_clusters():
    cases,rows=fixture();report=analyze(cases,rows)
    assert report['observations']==56 and report['cases']==2 and report['gate']=='PASS'
    assert report['primary_ratio']==.5 and report['problem_bootstrap_ci95']==[.5,.5]
    assert report['arms']['prepared']['distinct_valid']==2
    assert report['arms']['prepared']['charged_ns_per_verified_answer']==50
    assert report['arms']['symbolic']['summed_case_median_ns']<report['arms']['prepared']['summed_case_median_ns']


@pytest.mark.parametrize('change',['missing','duplicate','order','case','round','arm','valid','answer',
                                   'steps','target','construction','work','stop','negative_time','bool_time',
                                   'fractional_time','duplicate_case','clue'])
def test_corruption_rejected(change):
    cases,rows=fixture();index=next(i for i,r in enumerate(rows) if r['arm']=='prepared');row=rows[index]
    if change=='missing':rows.pop()
    elif change=='duplicate':rows[-1]=copy.deepcopy(rows[0])
    elif change=='order':row['order']=4
    elif change=='case':row['case_id']='missing'
    elif change=='round':row['round']=17
    elif change=='arm':row['arm']='other'
    elif change=='valid':row['valid']=False
    elif change=='answer':row['answer'][0]=0
    elif change=='steps':row['steps']=0
    elif change=='target':row['work']['target_used']=True
    elif change=='construction':row['work']['checker_constructions']=0
    elif change=='work':row['work']['block_applications']=0
    elif change=='stop':row['work']['stop_reason']='budget_exhausted'
    elif change=='negative_time':row['elapsed_ns']=-1
    elif change=='bool_time':row['elapsed_ns']=True
    elif change=='fractional_time':row['elapsed_ns']=1.5
    elif change=='duplicate_case':cases.append(copy.deepcopy(cases[0]))
    elif change=='clue':cases[0]['input'][0]=5
    with pytest.raises(ValueError):analyze(cases,rows)


def test_failing_performance_gate_is_a_retained_outcome():
    cases,rows=fixture()
    for r in rows:
        if r['arm']=='prepared':r['elapsed_ns']=110
    report=analyze(cases,rows)
    assert report['gate']=='FAIL' and report['primary_ratio']==1.1
    assert report['case_median_regressions']==2


def test_charged_time_includes_failed_neural_attempts():
    cases,rows=fixture()
    for case in cases:case['input']=[0]*16
    for r in rows:
        if r['arm']!='symbolic':
            r['answer']=[0]*16;r['valid']=False;r['steps']=4
            r['work'].update(executed_steps=4,block_applications=8,semantic_checks=4,
                             final_semantic=False,stop_reason='budget_exhausted')
    report=analyze(cases,rows)
    assert report['gate']=='FAIL' and report['arms']['prepared']['charged_ns_per_verified_answer'] is None
    assert report['arms']['symbolic']['distinct_valid']==2


@pytest.mark.parametrize('answer',[BOARD,[0]*16,[1]*16,list(reversed(BOARD))])
def test_independent_latin_and_clue_checks(answer):
    assert correct([0]*16,answer)==(answer in (BOARD,list(reversed(BOARD))))


def test_respects_original_clues():
    changed=BOARD.copy();changed[0]=2
    assert not correct(changed,BOARD)


def ablation_fixture():
    from scripts.bench_fp_controls import ARMS as arms,ORDER_SEED as seed
    cases,_=fixture();rng=random.Random(seed);rows=[]
    for c in cases:
        for r in range(7):
            order=list(arms);rng.shuffle(order)
            for pos,arm in enumerate(order):
                work={'executed_steps':1,'semantic_checks':1,'checker_constructions':1,
                      'block_applications':2,'target_used':False,'stop_reason':'semantic_valid','final_semantic':True}
                rows.append({'case_id':c['case_id'],'round':r,'order':pos,'arm':arm,
                             'elapsed_ns':100 if arm==arms[0] else 75,'valid':True,'answer':BOARD.copy(),'work':work})
    return cases,rows


def test_ablation_matrix_analysis():
    from scripts.bench_fp_controls import analyze as analyze_controls
    cases,rows=ablation_fixture();r=analyze_controls(cases,rows)
    assert r['native_over_eager_ratio']==.75 and r['problem_bootstrap_ci95']==[.75,.75]
    assert r['observations']==28 and r['distinct_valid']==2


@pytest.mark.parametrize('change',['missing','order','answer','valid','steps','checks','target','stop','time','case'])
def test_ablation_corruption_rejection(change):
    from scripts.bench_fp_controls import analyze as analyze_controls
    cases,rows=ablation_fixture();r=rows[0]
    if change=='missing':rows.pop()
    elif change=='order':r['order']=6
    elif change=='answer':r['answer'][0]=0
    elif change=='valid':r['valid']=False
    elif change=='steps':r['work']['executed_steps']=5
    elif change=='checks':r['work']['semantic_checks']=0
    elif change=='target':r['work']['target_used']=True
    elif change=='stop':r['work']['stop_reason']='budget_exhausted'
    elif change=='time':r['elapsed_ns']=True
    elif change=='case':r['case_id']='missing'
    with pytest.raises(ValueError):analyze_controls(cases,rows)


@pytest.mark.parametrize('arm',ARMS)
def test_later_round_full_work_record_must_match(arm):
    cases,rows=fixture()
    row=next(r for r in rows if r['arm']==arm and r['round']==5)
    row['work']['extra_corrupted_field']='not present in original execution'
    with pytest.raises(ValueError):analyze(cases,rows)


@pytest.mark.parametrize('field',['executed_steps','semantic_checks','checker_constructions','block_applications'])
def test_work_counter_boolean_is_not_an_integer_receipt(field):
    cases,rows=fixture()
    row=next(r for r in rows if r['arm']=='prepared' and r['round']==5)
    row['work'][field]=True
    with pytest.raises(ValueError):analyze(cases,rows)


@pytest.mark.parametrize('field',['executed_steps','semantic_checks','checker_constructions','block_applications'])
def test_ablation_rejects_boolean_counters(field):
    from scripts.bench_fp_controls import analyze as analyze_controls
    cases,rows=ablation_fixture()
    for row in rows:row['work'][field]=True
    with pytest.raises(ValueError):analyze_controls(cases,rows)
