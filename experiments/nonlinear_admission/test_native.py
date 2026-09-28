"""Native baselines reproduce source parameters; only small synthetic fit fixtures."""
import ctypes as C
from pathlib import Path
import subprocess,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.nonlinear_admission.native import DenseSession,export_dense,FLAGS

@pytest.fixture(scope='module')
def library(tmp_path_factory):
    p=tmp_path_factory.mktemp('admission-dense')/'dense.so'
    subprocess.run(['g++',*FLAGS,str(Path(__file__).with_name('dense.cpp')),'-o',str(p)],check=True)
    return p

@pytest.mark.parametrize('kind',['linear','mlp'])
@pytest.mark.parametrize('d,c',[(1,3),(13,6),(17,7),(73,26)])
def test_trained_control_roundtrip(tmp_path,library,kind,d,c):
    from sklearn.svm import LinearSVC
    from sklearn.neural_network import MLPClassifier
    from threadpoolctl import threadpool_limits
    rng=np.random.default_rng(431+d+c);x=rng.normal(size=(156,d));y=np.arange(len(x))%c+10
    with threadpool_limits(1):
        model=LinearSVC(C=.1,dual='auto',max_iter=10000) if kind=='linear' else MLPClassifier(hidden_layer_sizes=(9,),max_iter=10,random_state=13)
        model.fit(x,y)
        path=tmp_path/'weights';entry=export_dense(model,path)
        with DenseSession(library,path,d,model.classes_) as w:
            assert w.predict_buffer(x)==model.predict(x).tolist()
            for n in (0,1,3,4,17):assert w.predict_buffer(x[:n])==model.predict(x[:n]).tolist() if n else w.predict_buffer(x[:0])==[]
            labels,scorebytes=w.predict_buffer(x,return_scores=True)
            actual=np.frombuffer(scorebytes,dtype=np.float64).reshape(len(x),c)
            expected=x@model.coef_.T+model.intercept_ if kind=='linear' else np.maximum(0.,x@model.coefs_[0]+model.intercepts_[0])@model.coefs_[1]+model.intercepts_[1]
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
        with pytest.raises(ValueError,match='closed'):w.predict_buffer(x)
        with pytest.raises(FileExistsError):export_dense(model,path)

def test_bad_model_bytes(tmp_path,library):
    path=tmp_path/'bad';path.write_bytes(b'0'*30)
    with pytest.raises(ValueError):DenseSession(library,path,3,[0,1,2])

def test_invalid_input_does_not_corrupt_reusable_model(tmp_path,library):
    from types import SimpleNamespace
    model=SimpleNamespace(coef_=np.eye(3),intercept_=np.zeros(3),classes_=np.array([0,1,2]))
    path=tmp_path/'weights';export_dense(model,path)
    with DenseSession(library,path,3,model.classes_) as w:
        for x in (np.zeros((1,2)),np.zeros((3,3),dtype=np.float32),np.ones((3,6))[:,::2]):
            with pytest.raises(ValueError):w.predict_buffer(x)
        bad=np.zeros((2,3));bad[-1,-1]=np.nan
        with pytest.raises(ValueError,match='nonfinite'):w.predict_buffer(bad)
        assert w.predict_buffer(np.eye(3))==[0,1,2]
