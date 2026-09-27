"""Restricted raw-input pipeline: fidelity, fail-closed export and inert loading."""
from array import array
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from spectra.svm_pipeline import Preprocessor, PreparedPipeline, SCHEMA
from spectra.svm_pipeline_export import export_pipeline, export_plan


def definition():
    return {'schema':SCHEMA,'columns':['amount','kind'],'features':3,'model_sha256':'0'*64,
            'operations':[{'kind':'numeric','column':0,'fill':(2.).hex(),'mean':(-0.).hex(),'scale':(3.).hex()},
                          {'kind':'onehot','column':1,'categories':['a','b'],'unknown':'ignore'}]}


def plan(doc=None):
    return Preprocessor(json.dumps(doc or definition(),allow_nan=False).encode())


@pytest.fixture
def fitted():
    np=pytest.importorskip('numpy');pd=pytest.importorskip('pandas')
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler,OneHotEncoder
    from sklearn.svm import SVC
    frame=pd.DataFrame({'amount':[1.,2.,np.nan,3.,6.,7.,5.,8.],
                        'kind':['red','blue','red','blue','red','blue','red','blue'],
                        'constant':[4.]*8})
    pp=ColumnTransformer([('n',Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]),['amount','constant']),
                          ('c',OneHotEncoder(sparse_output=False,handle_unknown='ignore'),['kind'])],sparse_threshold=0.)
    x=pp.fit_transform(frame);svc=SVC(C=2.,gamma='scale').fit(x,['low']*4+['high']*4)
    return frame,pp,svc


def test_arithmetic_and_unknowns():
    p=plan();out=p.transform([[None,'a'],[math.nan,'b'],[1.,'new'],[-0.,'a']])
    assert out.tolist()==[2/3,1.,0.,2/3,0.,1.,1/3,0.,0.,0.,1.,0.]
    assert p.columns==('amount','kind') and p.features==3
    assert p.transform([])==array('d')
    assert out.buffer_info()[1]==12


@pytest.mark.parametrize('value',[float('inf'),-float('inf'),True,'1',b'1',complex(1,0),10**999])
def test_bad_numeric_rejected(value):
    with pytest.raises(ValueError):plan().transform([[value,'a']])


@pytest.mark.parametrize('value',[None,math.nan,1,True,[],{},'x'*4097])
def test_bad_category_rejected(value):
    with pytest.raises(ValueError):plan().transform([[1,value]])


def test_unknown_error_preserves_known_zero_column():
    d=definition();d['operations'][1]['unknown']='error'
    p=plan(d)
    assert p.transform([[0,'a']])[1]==1
    with pytest.raises(ValueError,match='unknown'):p.transform([[0,'unknown']])


@pytest.mark.parametrize('change',['schema','extra','duplicate_columns','bad_column','wrong_features','unknown_kind',
                                   'bad_scale','nonfinite_hex','noncanonical_hex','bad_hash','duplicate_categories',
                                   'unknown_mode','empty_categories','extra_operation_key','too_many_features'])
def test_bad_plans_rejected(change):
    d=definition()
    if change=='schema':d['schema']='arbitrary.code'
    elif change=='extra':d['callable']='os.system'
    elif change=='duplicate_columns':d['columns']=['a','a']
    elif change=='bad_column':d['operations'][0]['column']=True
    elif change=='wrong_features':d['features']=2
    elif change=='unknown_kind':d['operations'][0]['kind']='python'
    elif change=='bad_scale':d['operations'][0]['scale']=(-1.).hex()
    elif change=='nonfinite_hex':d['operations'][0]['fill']='inf'
    elif change=='noncanonical_hex':d['operations'][0]['fill']='2'
    elif change=='bad_hash':d['model_sha256']='../model'
    elif change=='duplicate_categories':d['operations'][1]['categories']=['x','x']
    elif change=='unknown_mode':d['operations'][1]['unknown']='guess'
    elif change=='empty_categories':d['operations'][1]['categories']=[]
    elif change=='extra_operation_key':d['operations'][1]['callable']='x'
    else:d['features']=4097
    with pytest.raises(ValueError):plan(d)


@pytest.mark.parametrize('raw',[b'',b'\xff',b'{"schema":1,"schema":2}',b'{"x":NaN}', b'['*2000+b']'*2000])
def test_bad_json_rejected(raw):
    with pytest.raises(ValueError):Preprocessor(raw)


def test_caps_and_bounded_iterators(monkeypatch):
    import spectra.svm_pipeline as mod
    monkeypatch.setattr(mod,'MAX_PLAN_BYTES',5)
    with pytest.raises(ValueError):plan()
    monkeypatch.setattr(mod,'MAX_PLAN_BYTES',4*1024*1024)
    monkeypatch.setattr(mod,'MAX_ROWS',3)
    p=plan()
    assert len(p.transform([[1.,'a']]*3))==9
    with pytest.raises(ValueError,match='cap'):p.transform(itertools.repeat([1.,'a']))
    with pytest.raises(ValueError):p.transform([itertools.repeat(0)])
    with pytest.raises(ValueError):p.transform([[1.,'a']],columns=itertools.repeat('amount'))


@pytest.mark.parametrize('rows',[[[1.]],[[1.,'a',2.]],{'amount':1},['not-a-row'],[{'amount':1,'kind':'a'}]])
def test_shape_and_record_type_rejected(rows):
    with pytest.raises(ValueError):plan().transform(rows)


def test_no_mutation_concurrency_and_schema_order():
    p=plan();rows=[[1.,'a'],[3.,'b']];original=copy.deepcopy(rows)
    with ThreadPoolExecutor(4) as pool:
        results=list(pool.map(lambda _:p.transform(rows).tobytes(),range(64)))
    assert len(set(results))==1 and rows==original
    with pytest.raises(ValueError,match='schema'):p.transform(rows,columns=['kind','amount'])
    assert p.transform(rows,columns=['amount','kind']).tobytes()==results[0]


def test_numerical_overflow_rejected():
    d=definition();d['operations'][0]['scale']=(5e-324).hex()
    with pytest.raises(ValueError,match='nonfinite'):plan(d).transform([[1.,'a']])


def test_fitted_roundtrip_and_workers_survive_owner(tmp_path,library,fitted):
    import numpy as np
    frame,pp,svc=fitted
    out=tmp_path/'bundle';receipt=export_pipeline(pp,svc,out)
    assert receipt['model']['sha256']==hashlib.sha256((out/'model.srt').read_bytes()).hexdigest()
    p=PreparedPipeline(out,library);workers=[p.session() for _ in range(3)]
    rows=list(frame.itertuples(index=False,name=None));expected=svc.predict(pp.transform(frame)).tolist()
    assert np.asarray(p.preprocessor.transform(rows)).tobytes()==pp.transform(frame).tobytes()
    p.close();p.close()
    with pytest.raises(ValueError):p.session()
    with ThreadPoolExecutor(3) as pool:
        assert list(pool.map(lambda w:w.predict_many(rows),workers))==[expected]*3
    assert workers[0].predict_with_certificate(rows[0]).label==expected[0]
    for w in workers:w.close()
    with pytest.raises(ValueError):workers[0].predict(rows[0])
    with pytest.raises(FileExistsError):export_pipeline(pp,svc,out)


def test_missing_unknown_and_extreme_roundtrip(fitted):
    import numpy as np
    frame,pp,_=fitted
    p=Preprocessor(export_plan(pp,model_sha256='1'*64,features=4))
    probe=frame.copy();probe.loc[0,'kind']='unseen';probe.loc[1,'amount']=np.nan;probe.loc[2,'amount']=-0.
    probe.loc[3,'amount']=np.nextafter(1.,2.);probe.loc[4,'amount']=1e150
    rows=list(probe.itertuples(index=False,name=None))
    assert p.transform(rows).tobytes()==pp.transform(probe).tobytes()


@pytest.mark.parametrize('kind',['hash','features','missing_model'])
def test_bound_model_rejection(tmp_path,library,fitted,kind):
    _,pp,svc=fitted;out=tmp_path/'bundle';export_pipeline(pp,svc,out)
    file=out/'preprocessing.json';d=json.loads(file.read_text())
    if kind=='hash':d['model_sha256']='0'*64
    elif kind=='features':
        d['features']+=1;d['operations'].append(d['operations'][0])
    else:(out/'model.srt').unlink()
    file.write_text(json.dumps(d))
    with pytest.raises((ValueError,FileNotFoundError)):PreparedPipeline(out,library)


@pytest.mark.parametrize('kind',['weights','remainder','scale_off','indicator','other_strategy',
                                 'drop','sparse','frequency','numeric_categories','subclass','extra_step','unfitted'])
def test_unsupported_export_rejected(fitted,kind):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.impute import SimpleImputer
    frame,pp,_=fitted;pp=copy.deepcopy(pp)
    numeric=pp.named_transformers_['n'];enc=pp.named_transformers_['c']
    if kind=='weights':pp.transformer_weights={'n':2}
    elif kind=='remainder':pp.remainder='passthrough'
    elif kind=='scale_off':numeric.steps[1][1].with_std=False
    elif kind=='indicator':numeric.steps[0][1].add_indicator=True
    elif kind=='other_strategy':numeric.steps[0][1].strategy='most_frequent'
    elif kind=='drop':enc.drop='first'
    elif kind=='sparse':enc.sparse_output=True
    elif kind=='frequency':enc.min_frequency=2
    elif kind=='numeric_categories':
        import numpy as np
        enc.categories_=[np.array([1,2])]
    elif kind=='subclass':
        class Other(ColumnTransformer):pass
        pp.__class__=Other
    elif kind=='extra_step':numeric.steps.append(('extra',FunctionTransformer()))
    elif kind=='unfitted':pp=ColumnTransformer([])
    with pytest.raises(ValueError):export_plan(pp,model_sha256='0'*64,features=4)


def test_deployment_import_is_framework_free():
    root=Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable,'-S','-c',
        'import spectra.svm_pipeline,sys; assert not ({"numpy","sklearn","pandas","torch"}&sys.modules.keys())'],cwd=root,check=True)


def test_exact_float_fastpath_preserves_other_real_types():
    import numpy as np
    class FloatSubclass(float): pass
    p=plan()
    for value in (1.25,FloatSubclass(1.25),np.float64(1.25),np.float32(1.25),np.int64(1),1):
        assert p.transform([[value,'a']])[0] == float(value)/3.
    for value in (np.bool_(True),np.complex128(1),np.inf):
        with pytest.raises(ValueError):p.transform([[value,'a']])


@pytest.mark.parametrize('where',['root','numeric','imputer','scaler','categorical'])
def test_instance_transform_overrides_rejected(fitted,where):
    _,pp,_=fitted
    n=pp.named_transformers_['n']
    target={'root':pp,'numeric':n,'imputer':n.steps[0][1],'scaler':n.steps[1][1],
            'categorical':pp.named_transformers_['c']}[where]
    target.transform=lambda x:x
    with pytest.raises(ValueError,match='overridden'):
        export_plan(pp,model_sha256='0'*64,features=4)
