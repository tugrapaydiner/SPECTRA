"""Tests of the actual supervised objective and deployment-preserving integer budget."""
import math
import numpy as np
import pytest
from . import learning
from .satellite import weights as satellite_weights


def fixture():
 rng=np.random.default_rng(123);q=rng.integers(0,16,(96,4),dtype=np.uint8)
 y=((q[:,0].astype(int)+q[:,1].astype(int))>15).astype(int)
 return q,y


def test_nca_analytic_gradient_matches_finite_differences(monkeypatch):
 original=learning.minimize;calls=[]
 def checked(fun,x,*args,**kwargs):
  _,gradient=fun(x);epsilon=1e-5
  for j in range(len(x)):
   direction=np.eye(len(x))[j]*epsilon
   numerical=(fun(x+direction)[0]-fun(x-direction)[0])/(2*epsilon)
   assert math.isclose(numerical,float(gradient[j]),rel_tol=2e-5,abs_tol=1e-7)
  calls.append(True);return original(fun,x,*args,**kwargs)
 monkeypatch.setattr(learning,'minimize',checked)
 q,y=fixture();w,r=learning.fit_weights(q,y,15,method='nca',seed=611)
 assert calls==[True] and w.sum()==8 and w.min()>=1 and r['history'][0]['converged']


def test_nca_is_deterministic_and_label_sensitive():
 q,y=fixture();w,a=learning.fit_weights(q,y,15,method='nca',seed=611)
 repeat,b=learning.fit_weights(q,y,15,method='nca',seed=611)
 other,c=learning.fit_weights(q,1-y,15,method='nca',seed=611)
 assert np.array_equal(w,repeat) and np.array_equal(w,other) # Relabelling classes must not change geometry.
 shuffled=np.random.default_rng(92).permutation(y);_,d=learning.fit_weights(q,shuffled,15,method='nca',seed=611)
 assert not np.allclose(a['continuous_weights'],d['continuous_weights'])


@pytest.mark.parametrize('arm',['uniform','variance','nca'])
def test_satellite_projection_never_discards_features(arm):
 rng=np.random.default_rng(190);q=rng.integers(0,256,(96,36),dtype=np.uint8);y=np.arange(96)%3
 w,record=satellite_weights(q,y,arm,611)
 assert w.min()>=1 and w.sum()==(36 if arm=='uniform' else 64)
 assert int(w.sum())*255**2+1<=4194304
 assert record['integer_weights']==w.tolist()


@pytest.mark.parametrize('bad',[0,-1,True,1.5,9])
def test_bad_quantization_units(bad):
 with pytest.raises(ValueError):learning.quantize_weights([1.,2.],units=bad)
