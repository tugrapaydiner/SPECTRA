from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np
import pytest
import torch
from spectra_reliability.evaluation import TargetMCTS
from spectra_reliability.freeze import (create_freeze,verify_freeze,open_confirmation,scientific_source_inventory)
from spectra_reliability.identity import write_json


@torch.inference_mode()
def test_first_hit_mcts_separates_future_backup_from_current_selection():
    # A potential future solution must not outrank an already-good current answer.
    class Head:
        target='first_hit'
        def first_hit_probabilities(self,x,y,z,continuation_policy):
            return y
    s=object.__new__(TargetMCTS);s.target_evaluator=Head();s.max_depth=4
    s._current_selection={};s.best_value=None;s.best_node=None;s.last_search_stats={'horizon_queries':{}}
    a=SimpleNamespace(path=(0,),depth=1,y=torch.tensor([[.1,.8,.05,.03,.02]]),latent=lambda:None)
    b=SimpleNamespace(path=(1,),depth=1,y=torch.tensor([[.8,.01,.01,.01,.17]]),latent=lambda:None)
    va=s._value(None,a);vb=s._value(None,b)
    assert va>vb
    s._consider_best(a,va);s._consider_best(b,vb)
    assert s.best_node is b and s.best_value==pytest.approx(.8)
    assert s.last_search_stats['horizon_queries']=={'3':2}
    with pytest.raises(NotImplementedError):s.search_batched(None)


def fake_run(tmp_path):
    root=tmp_path/'source';out=tmp_path/'run';(root/'spectra_reliability').mkdir(parents=True)
    (root/'spectra_reliability'/'example.py').write_text('VALUE=1\n')
    for name in ('recipe.json','prepare_complete.json','fit_complete.json','development_complete.json'):
        write_json(out/name,{})
    for directory in ('sources','lineage','auxiliary'):
        write_json(out/directory/'one.json',{'identity':directory})
    for stage in ('fit','validation','development'):
        for suffix in ('.json','.npz','_audit.json'):
            p=out/'data'/f'{stage}{suffix}';p.parent.mkdir(exist_ok=True);p.write_bytes(b'fixture')
    for surface in ('validation','development'):
        write_json(out/'evaluation'/surface/'aggregate.json',{})
    return root,out


def test_freeze_rejects_changed_source_or_checkpoint(tmp_path):
    root,out=fake_run(tmp_path);create_freeze(root,out,'1'*40);verify_freeze(root,out)
    p=root/'spectra_reliability'/'example.py';p.write_text('VALUE=2\n')
    with pytest.raises(ValueError,match='implementation changed'):verify_freeze(root,out)
    p.write_text('VALUE=1\n');verify_freeze(root,out)
    (out/'sources'/'one.json').write_text('{}')
    with pytest.raises(ValueError,match='artifact changed'):verify_freeze(root,out)


def test_freeze_detects_added_code_and_requires_published_sha(tmp_path):
    root,out=fake_run(tmp_path)
    with pytest.raises(ValueError):create_freeze(root,out,'main')
    create_freeze(root,out,'1'*40)
    (root/'spectra_reliability'/'new.py').write_text('new=1')
    with pytest.raises(ValueError):verify_freeze(root,out)


def test_confirmation_exclusive_open_and_no_preexisting_holdout(tmp_path):
    root,out=fake_run(tmp_path);create_freeze(root,out,'1'*40)
    record=open_confirmation(root,out,'2'*40)
    assert record['freeze_commit']=='2'*40
    with pytest.raises(FileExistsError):open_confirmation(root,out,'2'*40)
    with pytest.raises(ValueError):create_freeze(root,out,'1'*40)


def test_preexisting_confirmation_cannot_be_called_unopened(tmp_path):
    root,out=fake_run(tmp_path);(out/'data'/'confirmation.npz').write_bytes(b'exposed')
    with pytest.raises(ValueError,match='holdout'):create_freeze(root,out,'1'*40)


def test_pool_export_keeps_labels_separate_from_same_named_predictions():
    from spectra_reliability.evaluation import pool_array_payload
    n=2;p=12
    dataset=SimpleNamespace(inputs=np.zeros((n,16),dtype=np.int64))
    pool=SimpleNamespace(answers=torch.ones(n*p,16,dtype=torch.int64),validity=torch.ones(n*p),
        quality=torch.ones(n*p),improvement=torch.zeros(n*p),first_hit=torch.zeros(n*p,dtype=torch.int64))
    predictions={'validity':np.full(n*p,.2),'quality':np.full(n*p,.3),'improvement':np.full(n*p,.9)}
    arrays=pool_array_payload(dataset,pool,predictions)
    assert arrays['label_validity'].dtype==np.bool_ and arrays['label_validity'].all()
    assert np.all(arrays['prediction_validity']==.2)
    assert np.all(arrays['label_improvement']==0) and np.all(arrays['prediction_improvement']==.9)
    assert set(arrays)=={'inputs','answers','label_validity','label_quality','label_improvement','label_first_hit',
                         'prediction_validity','prediction_quality','prediction_improvement'}
