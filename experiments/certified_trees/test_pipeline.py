"""Actual upstream-library fallback and one-owner lifetime tests on a tiny model."""
from array import array
from concurrent.futures import ThreadPoolExecutor
import ctypes as C,hashlib,os
from pathlib import Path
import pytest
from .packing import compile_binary
from .pipeline import RefinementSession

@pytest.fixture
def case(tmp_path):
    import numpy as np
    from catboost import CatBoostClassifier
    m=CatBoostClassifier(iterations=1,depth=1,loss_function='MultiClass',thread_count=1,
                         bootstrap_type='No',random_strength=0,random_seed=19,verbose=False,allow_writing_files=False)
    m.fit(np.array([[0],[0],[1],[1],[1],[1]],dtype=np.uint8),[0,0,1,1,2,2])
    assert len(m.get_leaf_values())==6
    # This is explicitly synthetic: create a source where coarse quantization
    # loses a positive margin. The upstream engine must repair it, not our FP64 code.
    m.set_leaf_values(np.array([0.,.125,0.,0.,100.,0.]))
    m.save_model(str(tmp_path/'source.cbm'));m.save_model(str(tmp_path/'source.json'),format='json')
    raw=(tmp_path/'source.json').read_bytes();blob,_=compile_binary(raw,1,bits=8,pairwise=True)
    (tmp_path/'q8.sct').write_bytes(blob)
    return tmp_path

@pytest.fixture
def library():
    path=os.environ.get('TREE_PIPELINE_LIBRARY')
    if path is None:pytest.skip('explicit official-CatBoost pipeline library required')
    return Path(path)

def open_case(folder,library,**changed):
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    args={'compact_sha256':sha(folder/'q8.sct'),'cbm_sha256':sha(folder/'source.cbm'),'source_json_sha256':sha(folder/'source.json')}
    args.update(changed)
    return RefinementSession(folder/'q8.sct',folder/'source.cbm',library,**args)

def test_real_upstream_repairs_wrong_quantization(case,library):
    with open_case(case,library) as p:
        r=p.inspect_buffer(bytearray([0,1]))
        assert r['indices']==[1,1] and r['fallback']==[1,0]
        assert p.predict_buffer(bytearray())==[]
        with ThreadPoolExecutor(3) as pool:assert list(pool.map(lambda _:p.predict_buffer(bytearray([0,1]*17)),range(6)))==[[1]*34]*6

@pytest.mark.parametrize('changed',[{'compact_sha256':'0'*64},{'cbm_sha256':'0'*64},{'source_json_sha256':'0'*64}])
def test_pairing_identity(case,library,changed):
    with pytest.raises(ValueError):open_case(case,library,**changed)

def test_pipeline_late_bad_input_leaves_outputs(case,library):
    with open_case(case,library) as p:
        data=(C.c_uint8*2)(0,2);out=(C.c_int*2)(117,118);steps=(C.c_uint32*2)(119,120);fallback=(C.c_uint8*2)(3,4)
        assert p._lib.cb_pipeline_run(p._handle,data,2,1,0,out,steps,fallback)!=0
        assert list(out)==[117,118] and list(steps)==[119,120] and list(fallback)==[3,4]

def test_pipeline_same_thread_close(case,library):
    p=open_case(case,library);original=p._lib.cb_pipeline_run;values=array('B',[0,1])
    def closing(*args):p.close();assert p._handle;return original(*args)
    p._lib.cb_pipeline_run=closing
    try:assert p.predict_buffer(values)==[1,1]
    finally:p._lib.cb_pipeline_run=original;p.close()
    values.extend([0]);assert p._handle is None
    with pytest.raises(ValueError):p.predict_buffer(values)
