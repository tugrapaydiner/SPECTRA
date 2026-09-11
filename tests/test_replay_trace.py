import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from eval.replay_trace import sha256,tensor_digest,load_probe,compare_probes
from scripts.m17_models import tensor_digest as production_digest


def write_probe(path, *, changed=False):
    path.mkdir()
    x=np.array([[1,2]],dtype=np.int64)
    e=np.array([[[1.,2.],[3.,4.]]],dtype=np.float32)
    z=e.copy()
    if changed: z[0,0,0]=np.nextafter(z[0,0,0],np.float32(2.))
    arrays={'embed_0':e.copy(),'linear_0':z.copy(),'input':x,'embedded':e.copy(),'y':z.copy(),'z':z.copy()}
    source={'files':{'model/a.py':sha256(b'source')}}
    source['source_sha256']=sha256(json.dumps(source['files'],sort_keys=True).encode())
    report={'status':'DIAGNOSTIC_COMPLETE_NOT_ACCEPTANCE','source':source,
        'pool':'maze/development/1701','manifest_sha256':sha256(b'manifest'),
        'expected_pool_hash':sha256(b'pool'),'observed_pool_hash':sha256(b'changed' if changed else b'pool'),
        'historical_hash_match':not changed,'trace_no_hook_equal':True,'independent_production_equal':True,
        'cpuinfo':'model name : fixture','torch':'fixture','numpy':np.__version__,'python':'fixture',
        'torch_build':'fixture','dispatch_environment':{'ATEN_CPU_CAPABILITY':'avx2'},
        'trace_sha256':tensor_digest(arrays),
        'fields':{k:{'shape':list(v.shape),'sha256':tensor_digest({k:v})} for k,v in arrays.items()}}
    np.savez_compressed(path/'first_cycle.npz',**arrays)
    (path/'diagnostic.json').write_text(json.dumps(report))
    return path


def test_numpy_tensor_hash_matches_independent_torch_implementation():
    arrays={'x':np.arange(6,dtype=np.int64).reshape(2,3),'f':np.array([-0.,1.,2.],dtype=np.float32)}
    assert tensor_digest(arrays)==production_digest({k:torch.from_numpy(v) for k,v in arrays.items()})


def test_first_difference_is_chronological_and_diagnostic_not_acceptance(tmp_path):
    a=load_probe(write_probe(tmp_path/'a')); b=load_probe(write_probe(tmp_path/'b',changed=True))
    r=compare_probes(a,b)
    assert r['status']=='DIAGNOSTIC_ONLY_NOT_REPLAY_ACCEPTANCE'
    assert r['inputs_equal'] and r['embedded_bytes_equal'] and r['declared_dispatch_equal']
    assert r['first_different_observed_module_output']['name']=='linear_0'
    assert r['first_different_observed_module_output']['numerically_different_elements']==1
    assert r['left']['historical_hash_match'] and not r['right']['historical_hash_match']
    assert compare_probes(a,a)['first_different_observed_module_output'] is None


@pytest.mark.parametrize('mutate', ['field_hash','shape','source','combined_hash','hook','match_status','missing_field'])
def test_corrupt_contracts_fail_closed(tmp_path,mutate):
    path=write_probe(tmp_path/'p'); p=path/'diagnostic.json';r=json.loads(p.read_text())
    if mutate=='field_hash': r['fields']['input']['sha256']=sha256(b'bad')
    if mutate=='shape': r['fields']['input']['shape']=[2,1]
    if mutate=='source': r['source']['files']['model/a.py']=sha256(b'different')
    if mutate=='combined_hash': r['trace_sha256']=sha256(b'wrong')
    if mutate=='hook': r['trace_no_hook_equal']=False
    if mutate=='match_status': r['historical_hash_match']=False
    if mutate=='missing_field': del r['fields']['input']
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError): load_probe(path)


@pytest.mark.parametrize('name',['pool','manifest_sha256','expected_pool_hash','torch','numpy'])
def test_incompatible_provenance_is_not_a_hardware_comparison(tmp_path,name):
    a=load_probe(write_probe(tmp_path/'a'));b=load_probe(write_probe(tmp_path/'b'))
    b.report[name]='different'
    with pytest.raises(ValueError,match='identity'):compare_probes(a,b)


def test_changed_source_and_changed_trace_order_are_rejected(tmp_path):
    a=load_probe(write_probe(tmp_path/'a'));b=load_probe(write_probe(tmp_path/'b'))
    b.report['source']['files']['model/a.py']=sha256(b'other source')
    with pytest.raises(ValueError,match='sources'):compare_probes(a,b)
    c=replace(a,arrays=dict(reversed(list(a.arrays.items()))))
    with pytest.raises(ValueError,match='order'):compare_probes(a,c)


def test_byte_identity_distinguishes_signed_zero():
    assert tensor_digest({'x':np.array([0.],dtype=np.float32)}) != tensor_digest({'x':np.array([-0.],dtype=np.float32)})
