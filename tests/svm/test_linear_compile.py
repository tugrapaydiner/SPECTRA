"""Restricted fixed-model export, strict ordered scores and framework-free inference."""
from array import array
from concurrent.futures import ThreadPoolExecutor
import ctypes as C
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import pytest
from spectra.linear import CompiledLinear, build_linear, export_linear_svc, _canonical, _identity, _load


def fitted(features=7, classes=3):
    import numpy as np
    from sklearn.svm import LinearSVC
    rng=np.random.default_rng(618+features+classes)
    x=rng.normal(size=(64,features));y=np.arange(64)%classes
    return LinearSVC(C=1,dual='auto',random_state=0,max_iter=20000).fit(x,y),x


@pytest.fixture(scope='module')
def compiled(tmp_path_factory):
    root=tmp_path_factory.mktemp('linear-model');model,x=fitted()
    export_linear_svc(model,root/'bundle')
    library=build_linear(root/'bundle',root/'native')
    return root,model,x,CompiledLinear(root/'bundle',library),library


def ordered(model,rows):
    result=[]
    for row in rows:
        for weights,bias in zip(model.coef_,model.intercept_):
            score=0.
            for x,w in zip(row,weights):score+=float(x)*float(w)
            result.append(score+float(bias))
    return result


@pytest.mark.parametrize('features,classes',[(1,2),(3,3),(16,4),(33,2)])
def test_fitted_model_scores_and_original_labels(tmp_path,features,classes):
    import numpy as np
    model,x=fitted(features,classes)
    model.classes_=np.array([f'classe-{i}é' for i in range(classes)])
    export_linear_svc(model,tmp_path/'model');lib=build_linear(tmp_path/'model',tmp_path/'native')
    w=CompiledLinear(tmp_path/'model',lib)
    assert w.predict_buffer(x)==model.predict(x).tolist()
    assert array('d',w.scores_buffer(x)).tobytes()==array('d',ordered(model,x)).tobytes()
    assert w.predict_many(x.tolist())==model.predict(x).tolist()
    assert w.predict_buffer(array('d'))==[] and w.scores_buffer(array('d'))==[]
    assert w.predict(x[0])==model.predict(x[:1])[0]


def test_shared_fixed_data_has_no_request_state(compiled):
    _,model,x,w,_=compiled
    expected=model.predict(x).tolist()
    with ThreadPoolExecutor(4) as pool:
        assert list(pool.map(lambda _:w.predict_buffer(x),range(40)))==[expected]*40
    one=w.predict_buffer(x);two=w.predict_buffer(x)
    assert one==two and one is not two
    for name in ('labels','features','sha256'):
        with pytest.raises(AttributeError):setattr(w,name,None)


@pytest.mark.parametrize('bad',[array('f',[0.]*7),[0.]*7,b'\0'*56,memoryview(array('d',[0.]*7)).toreadonly(),array('d',[0.]*8)])
def test_incompatible_buffers_rejected(compiled,bad):
    with pytest.raises(ValueError):compiled[3].predict_buffer(bad)


def test_buffer_noncontiguous_shape_alignment(compiled):
    import numpy as np
    w=compiled[3]
    for values in (np.zeros((3,8)),np.zeros((3,14))[:,::2],np.zeros((1,1,7)),np.zeros((3,7),dtype='>f8')):
        with pytest.raises(ValueError):w.predict_buffer(values)
    raw=bytearray(57)
    with pytest.raises(ValueError,match='unaligned'):w.predict_buffer(memoryview(raw)[1:].cast('d'))


@pytest.mark.parametrize('row',[[0.]*6,[0.]*8,[True]*7,['0']*7,[float('inf')]*7,[float('nan')]*7,[10**1000]*7],
                         ids=['short','wide','bool','string','inf','nan','overflow'])
def test_invalid_input_rejected(compiled,row):
    with pytest.raises(ValueError):compiled[3].predict(row)


def test_bounded_iterator_and_byte_array_released(compiled):
    w=compiled[3]
    with pytest.raises(ValueError):w.predict(itertools.repeat(0.))
    values=array('d',[0.]*7);w.predict_buffer(values);values.append(1.)
    assert len(values)==8


def test_batch_row_cap(compiled):
    w=compiled[3]
    with pytest.raises(ValueError):w.predict_buffer(array('d',[0.])*(7*65537))
    with pytest.raises(ValueError):w.predict_many(itertools.repeat([0.]*7,65537))


def test_native_checks_all_finite_inputs_before_writes(compiled):
    w=compiled[3]
    values=(C.c_double*14)(*([0.]*13+[float('nan')]))
    output=(C.c_int*2)(99,88)
    assert w._lib.sp_linear_run(values,2,7,output,2,None,0)==3
    assert list(output)==[99,88]


def test_rounding_mode_rejected(compiled,rounding_library):
    old=rounding_library.test_get_round()
    try:
        assert rounding_library.test_set_round(rounding_library.test_downward())==0
        with pytest.raises(ValueError):compiled[3].predict([0.]*7)
    finally:assert rounding_library.test_set_round(old)==0


@pytest.mark.parametrize('classes',[2,3])
def test_exact_zero_tie_rules(tmp_path,classes):
    model,x=fitted(3,classes);model.coef_[:]=0.;model.intercept_[:]=0.
    export_linear_svc(model,tmp_path/'model');lib=build_linear(tmp_path/'model',tmp_path/'native')
    w=CompiledLinear(tmp_path/'model',lib)
    assert w.predict_many([[0.,-0.,1.]]*7)==[model.classes_[0]]*7


def test_arithmetic_overflow_rejected(tmp_path):
    model,_=fitted(1,2);model.coef_[:]=sys.float_info.max;model.intercept_[:]=0.
    export_linear_svc(model,tmp_path/'model');lib=build_linear(tmp_path/'model',tmp_path/'native')
    with pytest.raises(ValueError,match='4'):CompiledLinear(tmp_path/'model',lib).predict([2.])


def test_wrong_model_library_binding(compiled,tmp_path):
    root,model,_,_,lib=compiled
    model=model.__class__(**model.get_params()).fit([[i]*7 for i in range(12)],[i%3 for i in range(12)])
    export_linear_svc(model,tmp_path/'other')
    with pytest.raises(ValueError,match='binding'):CompiledLinear(tmp_path/'other',lib)


@pytest.mark.parametrize('damage',['missing-field','extra-field','version','dimension','bool-dimension','labels','duplicate-labels','identity','invalid-digest','duplicate-json','parameters','source','nonfinite'])
def test_changed_bundle_rejected(compiled,tmp_path,damage):
    root,_,_,_,lib=compiled;bundle=tmp_path/'bundle';shutil.copytree(root/'bundle',bundle)
    path=bundle/'model.json';meta=json.loads(path.read_text())
    if damage=='missing-field':meta.pop('labels')
    elif damage=='extra-field':meta['x']=1
    elif damage=='version':meta['format']='other'
    elif damage=='dimension':meta['features']=4097
    elif damage=='bool-dimension':meta['classes']=True
    elif damage=='labels':meta['labels']=[1,'mixed',3]
    elif damage=='duplicate-labels':meta['labels']=[1,1,1]
    elif damage=='identity':meta['model_sha256']='1'*64
    elif damage=='invalid-digest':meta['source_sha256']='z'*64
    elif damage=='parameters':
        value=bytearray((bundle/'parameters.f64').read_bytes());value[0]^=1;(bundle/'parameters.f64').write_bytes(value)
    elif damage=='source':(bundle/'model.cpp').write_text('not the original source')
    elif damage=='nonfinite':
        value=bytearray((bundle/'parameters.f64').read_bytes());struct.pack_into('<d',value,0,float('nan'))
        (bundle/'parameters.f64').write_bytes(value);meta['parameters_sha256']=hashlib.sha256(value).hexdigest()
        meta['model_sha256']=_identity(meta['features'],meta['labels'],meta['parameters_sha256'])
    path.write_bytes(_canonical(meta))
    if damage=='duplicate-json':path.write_bytes(b'{"format":"other",'+path.read_bytes()[1:])
    with pytest.raises(ValueError):build_linear(bundle,tmp_path/'native')
    assert not (tmp_path/'native').exists()


def test_export_scope_and_no_overwrite(compiled,tmp_path):
    root,model,_,_,_=compiled
    with pytest.raises(FileExistsError):export_linear_svc(model,root/'bundle')
    with pytest.raises(ValueError):export_linear_svc(object(),tmp_path/'bad')
    from sklearn.svm import LinearSVC
    with pytest.raises(ValueError):export_linear_svc(LinearSVC(),tmp_path/'bad')
    bad,_=fitted();bad.multi_class='crammer_singer'
    with pytest.raises(ValueError):export_linear_svc(bad,tmp_path/'bad')
    bad,_=fitted();bad.coef_[0,0]=float('nan')
    with pytest.raises(ValueError):export_linear_svc(bad,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_import_and_inference_without_frameworks(compiled):
    root,_,_,_,lib=compiled;source=Path(__file__).resolve().parents[2]
    code='import sys;sys.path.insert(0,sys.argv[1]);from spectra.linear import CompiledLinear;w=CompiledLinear(sys.argv[2],sys.argv[3]);assert len(w.predict_many([[0.]*7]))==1;assert not ({"numpy","sklearn","torch","pandas"}&sys.modules.keys())'
    done=subprocess.run([sys.executable,'-I','-S','-c',code,str(source),str(root/'bundle'),str(lib)],capture_output=True,text=True)
    assert done.returncode==0,done.stderr
