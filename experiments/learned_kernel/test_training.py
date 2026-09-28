"""Train/export/execute tiny independent models; no saved evaluation examples."""
import os
from pathlib import Path
import numpy as np
import pytest
from sklearn.svm import SVC
from experiments.learned_kernel.explore import radial,align_weights,fit_ovo
from experiments.learned_kernel.model import export_pairs,stock_pairs
from experiments.learned_kernel.session import KernelSession

@pytest.mark.parametrize('power',[1,2])
@pytest.mark.parametrize('kind',['single','uniform','aligned'])
def test_learned_mixture_roundtrip(tmp_path,power,kind):
 rng=np.random.default_rng(82);X=rng.integers(0,5,(48,5)).astype(float)/4;y=np.arange(48)%3;V=rng.integers(0,5,(23,5)).astype(float)/4
 g=[.2,1.,5.];w=[0,1,0] if kind=='single' else np.ones(3)/3 if kind=='uniform' else align_weights(X,y,g,power)
 K=radial(X,X,g,w,power)
 assert np.linalg.eigvalsh(K).min()>-1e-10
 pred,pairs=fit_ovo(X,y,V,g,w,3.,power);path=tmp_path/'test.mkl'
 export_pairs(X,y,pairs,gammas=g,weights=w,power=power,denominator=4,destination=path)
 with KernelSession(path,os.environ['LEARNED_KERNEL_LIBRARY']) as s:
  assert s.predict_buffer(V)==s.predict_buffer(V,mode='direct')==pred.tolist()

@pytest.mark.parametrize('classes',[2,3,5])
@pytest.mark.parametrize('D',[1,4])
def test_trained_integer_metric_roundtrip(tmp_path,classes,D):
 rng=np.random.default_rng(840+classes);X=rng.integers(0,D+1,(60,5)).astype(float)/D;y=np.arange(60)%classes;V=rng.integers(0,D+1,(19,5)).astype(float)/D
 counts=[0,1,2,3,4];order=np.repeat(np.arange(5),counts);m=SVC(C=3.,gamma=.75).fit(X[:,order],y);expected=m.predict(V[:,order]);path=tmp_path/'m.mkl'
 export_pairs(X,y,stock_pairs(m),gammas=[.75],weights=[1.],power=2,denominator=D,feature_counts=counts,destination=path)
 with KernelSession(path,os.environ['LEARNED_KERNEL_LIBRARY']) as s:
  assert s.predict_buffer(V)==s.predict_buffer(V,mode='direct_exhaustive')==expected.tolist()
  assert s.features==5 and s.info['table_entries']==sum(counts)*D*D+1
