"""Fixed-dual gradient, training-only selection and mixture constraints."""
import numpy as np
import pytest
from sklearn.svm import SVC
from experiments.learned_signature.learning import signatures,table_profile
from experiments.learned_signature.margin_mixture import project_simplex,fixed_dual_objective,dual_vectors,learn_margin_mixture

@pytest.mark.parametrize('classes',[2,3,5])
def test_fixed_dual_derivative(classes):
 rng=np.random.default_rng(classes);q=rng.integers(0,8,size=(45,6),dtype=np.uint8);y=np.arange(45)%classes
 sig=signatures(q,q,np.ones(6,dtype=np.uint32));kernels=[table_profile(7,[1]*6,g)[sig] for g in (.1,.3,1.)];weights=np.array([.2,.3,.5]);K=sum(w*b for w,b in zip(weights,kernels))
 svc=SVC(C=2,kernel='precomputed').fit(K,y);pairs=dual_vectors(svc);value,gradient=fixed_dual_objective(kernels,weights,pairs)
 assert value>0
 for i in range(3):
  delta=np.zeros(3);delta[i]=1e-5
  numerical=(fixed_dual_objective(kernels,weights+delta,pairs)[0]-fixed_dual_objective(kernels,weights-delta,pairs)[0])/(2e-5)
  assert abs(numerical-gradient[i])<1e-7
  assert all(abs(a.sum())<1e-6 for _,a in pairs)

@pytest.mark.parametrize('v',[[0,0,0],[10,-1,-5],[.2,.3,.5],[-30,0,30]])
def test_simplex(v):
 p=project_simplex(v);assert np.all(p>=0) and abs(p.sum()-1)<1e-14
 assert np.allclose(project_simplex(p),p)


def test_margin_learning_has_monotone_accepted_trace():
 rng=np.random.default_rng(8);q=rng.integers(0,8,size=(90,5),dtype=np.uint8);y=np.arange(90)%3
 w,result=learn_margin_mixture(q,y,7,.2,1.,2026)
 assert w.min()>=0 and abs(w.sum()-1)<1e-14
 accepted=[r['objective'] for r in result['trace'] if r['accepted']]
 assert all(b<=a+1e-8*max(1,abs(a)) for a,b in zip(accepted,accepted[1:]))
 assert result['fits']==len(result['trace'])
