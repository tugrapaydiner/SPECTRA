"""Corruption and negative-result handling without executing a compiler."""
import copy
import pytest
from spectra.compiler_evidence import ARMS,analyze,schedule

BOARD=[1,2,3,4,3,4,1,2,2,1,4,3,4,3,2,1]


def fixture():
    cases=[{'case_id':f'ordinary:{s}:0','family':'ordinary','seed':s,'problem_id':'ordinary:0',
            'input':BOARD.copy(),'blocks':1} for s in (1401,2402)]
    rows=[]
    work={'executed_steps':1,'block_applications':2,'semantic_checks':1,'checker_constructions':1,
          'final_semantic':True,'target_used':False,'checker':'native_exact_sudoku_v1','stop_reason':'semantic_valid'}
    for c,r,o,a in schedule(cases):
        rows.append({'case_id':c['case_id'],'round':r,'order':o,'arm':a,
            'elapsed_ns':{'eager':200,'prepared':100,'inductor_default':80,'inductor_frozen':60}[a],
            'valid':True,'answer':BOARD.copy(),'work':copy.deepcopy(work)})
    traces=[]
    for c in cases:
        for a in ARMS:
            t={'case_id':c['case_id'],'arm':a,'embedding_sha256':'1'*64,'embedding_bitwise':True,'steps':[]}
            for _ in range(4):t['steps'].append({'y_sha256':'2'*64,'z_sha256':'3'*64,'logits_sha256':'4'*64,
                    'bitwise':True,'finite':True,'max_abs_y':0.,'max_abs_z':0.,'max_abs_logits':0.})
            traces.append(t)
    return cases,rows,traces


def test_known_matrix_recomputed():
    s=analyze(*fixture());assert s['cases']==2 and s['unique_problems']==1 and s['observations']==56
    assert s['arms']['inductor_frozen']['over_prepared_ratio']==.6
    assert s['arms']['inductor_default']['charged_ns_per_valid']==80
    assert s['arms']['inductor_default']['exact_track_pass'] is True


@pytest.mark.parametrize('corrupt',['missing_row','duplicate_row','order','timebool','timenan','roundbool','answerbool',
    'wrong_valid','workbool','later_work','extra_work','missing_trace','duplicate_trace','short_trace',
    'false_bits','false_embedding','nan_error','null_finite_error','duplicate_case','different_problem'])
def test_corruption_rejected(corrupt):
    c,r,t=fixture()
    if corrupt=='missing_row':r.pop()
    elif corrupt=='duplicate_row':r[-1]=copy.deepcopy(r[0])
    elif corrupt=='order':r[0]['order']=2
    elif corrupt=='timebool':r[0]['elapsed_ns']=True
    elif corrupt=='timenan':r[0]['elapsed_ns']=float('nan')
    elif corrupt=='roundbool':r[0]['round']=False
    elif corrupt=='answerbool':r[0]['answer'][0]=True
    elif corrupt=='wrong_valid':r[0]['valid']=False
    elif corrupt=='workbool':r[0]['work']['semantic_checks']=True
    elif corrupt=='later_work':
        rr=next(x for x in r if x['round']==6);rr['work']['executed_steps']=2;rr['work']['semantic_checks']=2;rr['work']['block_applications']=4
    elif corrupt=='extra_work':r[0]['work']['hidden']=1
    elif corrupt=='missing_trace':t.pop()
    elif corrupt=='duplicate_trace':t[-1]=copy.deepcopy(t[0])
    elif corrupt=='short_trace':t[0]['steps'].pop()
    elif corrupt=='false_bits':t[-1]['steps'][0]['bitwise']=False
    elif corrupt=='false_embedding':t[-1]['embedding_bitwise']=False
    elif corrupt=='nan_error':t[-1]['steps'][0]['max_abs_logits']=float('nan')
    elif corrupt=='null_finite_error':t[-1]['steps'][0]['max_abs_logits']=None
    elif corrupt=='duplicate_case':c[1]=copy.deepcopy(c[0])
    elif corrupt=='different_problem':c[1]['input'][0]=0
    with pytest.raises((ValueError,KeyError,TypeError)):analyze(c,r,t)


def test_nonbitwise_compiler_can_preserve_bounded_task_outputs():
    c,r,t=fixture()
    for tr in t:
        if tr['arm']=='inductor_frozen':
            for s in tr['steps']:s['bitwise']=False;s['logits_sha256']='a'*64;s['max_abs_logits']=1e-5
    s=analyze(c,r,t)['arms']['inductor_frozen']
    assert not s['exact_track_pass'] and s['bounded_task_track_pass'] and s['nonbitwise_cases']==2


def test_lost_valid_answer_retained_as_failure_not_filtered():
    c,r,t=fixture()
    for row in r:
        if row['arm']=='inductor_frozen':
            row['answer'][0]=0;row['valid']=False
            row['work'].update(executed_steps=4,semantic_checks=4,block_applications=8,
                               final_semantic=False,stop_reason='budget_exhausted')
    s=analyze(c,r,t)['arms']['inductor_frozen']
    assert s['lost_valid']==2 and s['valid']==0 and s['charged_ns_per_valid'] is None
    assert not s['bounded_task_track_pass'] and not s['exact_track_pass']
