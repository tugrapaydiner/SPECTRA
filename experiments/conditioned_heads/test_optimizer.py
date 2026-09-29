import numpy as np
import pytest
from .optimizer import Coordinates, fit_head, loss_gradient

@pytest.mark.parametrize('method',['raw','diagonal','cholesky'])
@pytest.mark.parametrize('rank_deficient',[False,True])
def test_loss_and_gradient_transform(method,rank_deficient):
    rng=np.random.default_rng(61)
    f=rng.normal(size=(30,6));f+=2
    if rank_deficient:f[:,1]=f[:,0];f[:,3]=1
    h=rng.normal(size=(6,3));b=rng.normal(size=3);y=np.arange(30)%3
    coord,z=Coordinates.prepare(f,1e-3,method)
    w,beta=coord.encode(h,b);hh,bb=coord.decode(w,beta)
    np.testing.assert_allclose(hh,h,atol=1e-10);np.testing.assert_allclose(bb,b,atol=1e-10)
    np.testing.assert_allclose(z@w+beta,f@h+b,atol=1e-10)
    def val(u):
        h0,b0=coord.decode(u.reshape(h.shape),beta)
        return loss_gradient(f,y,h0,b0,1e-3)[0]
    # The data gradient uses z; regularization uses original h, not transformed w.
    logits=z@w+beta;prob=np.exp(logits-logits.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
    prob[np.arange(30),y]-=1;prob/=30
    g=z.T@prob+1e-3*coord.penalty_gradient(h)
    for j in range(w.size):
        d=np.zeros(w.size);d[j]=1e-5
        numeric=(val(w.ravel()+d)-val(w.ravel()-d))/(2e-5)
        assert np.isclose(numeric,g.ravel()[j],rtol=1e-4,atol=1e-7)

@pytest.mark.parametrize('method',['raw','diagonal','cholesky'])
def test_converged_same_objective(method):
    rng=np.random.default_rng(71);f=rng.normal(size=(80,5));y=rng.integers(0,3,80)
    h=np.zeros((5,3));b=np.zeros(3)
    a,c,report=fit_head(f,y,h,b,method=method,ridge=.01,maxiter=300,ftol=1e-14,gtol=1e-9)
    control_h,control_b,ref=fit_head(f,y,h,b,method='raw',ridge=.01,maxiter=300,ftol=1e-14,gtol=1e-9)
    assert abs(report['objective']-ref['objective'])<1e-10
    np.testing.assert_allclose(f@a+c,f@control_h+control_b,atol=3e-5)
    assert a.shape==h.shape and c.shape==b.shape

@pytest.mark.parametrize('bad',[-1,0,True,float('nan'),float('inf')])
def test_invalid_ridge(bad):
    with pytest.raises(ValueError):fit_head(np.ones((10,3)),np.arange(10)%2,np.zeros((3,2)),np.zeros(2),ridge=bad)

@pytest.mark.parametrize('bad',[0,-1,True,5001,2.5])
def test_invalid_iteration_count(bad):
    with pytest.raises(ValueError):fit_head(np.ones((10,3)),np.arange(10)%2,np.zeros((3,2)),np.zeros(2),maxiter=bad)

def test_missing_class_is_finite_and_inputs_unmodified():
    f=np.zeros((20,4));h=np.zeros((4,3));b=np.zeros(3);y=np.zeros(20,dtype=int)
    hh,bb,r=fit_head(f,y,h,b,maxiter=30)
    assert np.isfinite(hh).all() and np.isfinite(bb).all()
    assert not h.any() and not b.any() and not f.any()
    assert r['objective']<=r['initial_objective']
