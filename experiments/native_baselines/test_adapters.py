"""Explicit experiment-only adapter checks; not a general LIBSVM compatibility suite."""
from array import array
import ctypes as C
import json,math,os,pickle,sys
from pathlib import Path
import numpy as np
import pytest
from sklearn.svm import SVC
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.native_baselines.runtime import NativeSession
from experiments.native_baselines.prepare import export_libsvm

@pytest.fixture(scope='module')
def library():
    p=os.environ.get('NATIVE_BASELINE_LIBSVM')
    if not p:pytest.skip('explicit built upstream comparison library required')
    return Path(p)

@pytest.mark.parametrize('classes',[2,3,7])
@pytest.mark.parametrize('features',[1,16,73])
def test_exact_model_export_and_outputs(tmp_path,library,classes,features):
    rng=np.random.default_rng(1400+classes+features)
    x=rng.normal(size=(84,features));x[rng.random(x.shape)<.3]=0
    y=np.array(['label-'+str(i%classes) for i in range(len(x))])
    model=SVC(C=2,gamma=.1).fit(x,y);path=tmp_path/'m.txt';export_libsvm(model,path)
    probe=np.concatenate((x,rng.normal(size=(29,features))))
    with NativeSession(library,features,model.classes_,path) as w:
        actual,raw=w.predict_buffer(probe,return_scores=True)
        assert actual==model.predict(probe).tolist()
        margins=np.frombuffer(raw,dtype=np.float64).reshape(len(probe),-1)
        reference=model._decision_function(probe).reshape(len(probe),-1)
        assert np.max(np.abs(margins-reference))<1e-10
        assert w.predict_buffer(array('d'))==[]
    with pytest.raises(ValueError,match='closed'):w.predict_buffer(probe)
    with pytest.raises(FileExistsError):export_libsvm(model,path)

@pytest.fixture
def worker(tmp_path,library):
    x=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]],dtype=np.float64)
    model=SVC(C=2).fit(x,[0,0,1,1]);path=tmp_path/'m.txt';export_libsvm(model,path)
    with NativeSession(library,2,[0,1],path) as w:yield w

@pytest.mark.parametrize('bad',[array('f',[0,0]),array('d',[0]),memoryview(b'0000000000000000'),
                                np.zeros((2,3)),np.zeros((3,2))[:,::-1]])
def test_buffer_contract(worker,bad):
    with pytest.raises(ValueError):worker.predict_buffer(bad)

@pytest.mark.parametrize('bad',[math.nan,math.inf,-math.inf])
def test_late_nonfinite_no_output(worker,bad):
    x=(C.c_double*4)(0.,0.,bad,1.);out=(C.c_int*2)(55,66)
    assert worker.lib.nb_run(worker._ptr,x,2,2,out,None)!=0
    assert list(out)==[55,66]
    assert len(worker.predict_buffer(array('d',[0.,0.])))==1

def test_input_ownership_and_release(worker):
    a=array('d',[0.,0.,1.,1.]);saved=a.tobytes();first=worker.predict_buffer(a)
    assert a.tobytes()==saved
    a.extend([1.,1.]);second=worker.predict_buffer(a)
    assert len(first)==2 and len(second)==3 and first is not second

def test_null_and_bad_model(tmp_path,library):
    path=tmp_path/'bad';path.write_text('not a model\n')
    with pytest.raises(ValueError):NativeSession(library,2,[0,1],path)

@pytest.mark.parametrize('classes',[2,3])
@pytest.mark.parametrize('features',[1,7,17])
def test_generated_wrapper_conventions(tmp_path,classes,features):
    import m2cgen,subprocess
    from experiments.native_baselines.build import wrapper
    rng=np.random.default_rng(2000+classes+features)
    x=rng.normal(size=(36,features));y=np.arange(36)%classes
    model=SVC(C=2,gamma=.2).fit(x,y)
    (tmp_path/'model.c').write_text(m2cgen.export_to_c(model))
    (tmp_path/'wrapper.cpp').write_text(wrapper(classes,features))
    command=['g++','-std=c++17','-O3','-fno-fast-math','-ffp-contract=off','-fPIC','-shared',str(tmp_path/'wrapper.cpp'),'-o',str(tmp_path/'model.so')]
    p=subprocess.run(command,capture_output=True,text=True,timeout=60);assert p.returncode==0,p.stderr
    with NativeSession(tmp_path/'model.so',features,list(range(classes))) as w:
        actual,raw=w.predict_buffer(x,return_scores=True)
        assert actual==model.predict(x).tolist()
        margins=np.frombuffer(raw,dtype=np.float64).reshape(len(x),-1)
        assert np.max(np.abs(margins-model._decision_function(x).reshape(len(x),-1)))<1e-10
        assert w.predict_buffer(array('d'))==[]
        with pytest.raises(ValueError):w.predict_buffer(array('d',[math.nan]*features))
