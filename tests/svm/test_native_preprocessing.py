"""Optional compiled preprocessing: exact features, errors, ownership and fallback."""
from array import array
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import copy
import gc
import itertools
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest
from spectra.svm_pipeline import Preprocessor, PreparedPipeline, SCHEMA
from spectra.svm_preprocess_native import NativePreprocessor, build_preprocessor


def definition():
    return {'schema': SCHEMA, 'columns': ['x', 'k', 'z'], 'features': 5,
            'model_sha256': '0'*64, 'operations': [
                {'kind': 'onehot', 'column': 1, 'categories': ['α', 'b', ''], 'unknown': 'ignore'},
                {'kind': 'numeric', 'column': 2, 'fill': (2.).hex(), 'mean': (-0.).hex(), 'scale': (3.).hex()},
                {'kind': 'numeric', 'column': 0, 'fill': (-1.).hex(), 'mean': (1.).hex(), 'scale': (.3).hex()}]}


def reference(doc=None):
    return Preprocessor(json.dumps(doc or definition(), allow_nan=False).encode())


@pytest.fixture(scope='module')
def prelib(tmp_path_factory):
    override = os.environ.get('SPECTRA_PREPROCESS_LIBRARY')
    return Path(override) if override else build_preprocessor(tmp_path_factory.mktemp('preprocess')/'build')


@pytest.fixture
def compiled(prelib):
    return NativePreprocessor(reference(), prelib)


def test_builtin_fidelity_and_metadata(compiled):
    rows = [[1., 'α', None], [None, 'b', math.nan], [7, 'unknown', -0.], [-1., '', 1e150]]
    before = copy.deepcopy(rows)
    assert compiled.transform(rows).tobytes() == reference().transform(rows).tobytes()
    assert compiled.columns == ('x','k','z') and compiled.features == 5
    assert compiled.model_sha256 == '0'*64 and compiled.sha256 == reference().sha256
    assert rows[:1] == before[:1]  # NaN intentionally not equality comparable.
    assert compiled.transform([]).format == 'd'


@pytest.mark.parametrize('bad', [float('inf'), -float('inf'), True, '1', b'1', 1j, 10**999, object()])
def test_numeric_errors(prelib, bad):
    p=reference();c=NativePreprocessor(p,prelib)
    for transform in (p.transform,c.transform):
        with pytest.raises(ValueError):transform([[bad,'b',0.]])
    assert c.transform([[1.,'b',2.]]).tobytes()==p.transform([[1.,'b',2.]]).tobytes()


@pytest.mark.parametrize('bad', [None, math.nan, 1, True, [], {}, 'x'*4097])
def test_category_errors(compiled,bad):
    with pytest.raises(ValueError):compiled.transform([[0.,bad,0.]])


def test_invalid_unicode_and_category_policy(prelib):
    with pytest.raises(UnicodeError):NativePreprocessor(reference(),prelib).transform([[0.,'\ud800',0.]])
    d=definition();d['operations'][0]['unknown']='error';c=NativePreprocessor(reference(d),prelib)
    with pytest.raises(ValueError,match='unknown'):c.transform([[0.,'no',0.]])
    assert c.transform([[0.,'α',0.]])[0]==1.


@pytest.mark.parametrize('rows', [[[1.,'b']],[[1.,'b',2.,3.]],['abc'],[{'x':1}],{'x':1},'abc'])
def test_shapes(compiled,rows):
    with pytest.raises(ValueError):compiled.transform(rows)


def test_caps_and_iterators(prelib,monkeypatch):
    import spectra.svm_pipeline as p
    monkeypatch.setattr(p,'MAX_ROWS',3)
    c=NativePreprocessor(reference(),prelib)
    with pytest.raises(ValueError,match='cap'):c.transform(itertools.repeat([0.,'α',0.]))
    with pytest.raises(ValueError):c.transform([itertools.repeat(0.)])
    assert c.transform(iter([[0.,'α',0.]]*3)).tobytes()==reference().transform([[0.,'α',0.]]*3).tobytes()
    with pytest.raises(ValueError,match='schema'):c.transform([],columns=['x','z','k'])


def test_fallback_preserves_custom_scalars(compiled):
    class CustomFloat(float):
        def __float__(self):return 7.
    class CustomStr(str):
        pass
    rows=[[Fraction(1,3),CustomStr('b'),CustomFloat(2.)]]
    assert isinstance(compiled.transform(rows),array)  # Explicit whole-batch reference fallback.
    assert compiled.transform(iter(rows)).tobytes()==reference().transform(rows).tobytes()


def test_numpy_scalars_use_reference(compiled):
    np=pytest.importorskip('numpy')
    rows=[[np.float32(.2),np.str_('α'),np.int32(3)]]
    assert compiled.transform(rows).tobytes()==reference().transform(rows).tobytes()


def test_outputs_own_storage_and_plan_is_shared(compiled):
    a=compiled.transform([[1.,'b',2.]])
    b=compiled.transform([[2.,'α',3.]])
    wanted=b.tobytes();a[0]=33.
    with ThreadPoolExecutor(4) as pool:
        results=list(pool.map(lambda _:compiled.transform([[2.,'α',3.]]).tobytes(),range(100)))
    assert all(r==wanted for r in results)
    del compiled;gc.collect()
    assert b.tobytes()==wanted


def test_overflow(prelib):
    d=definition();d['operations'][2]['scale']=(5e-324).hex()
    c=NativePreprocessor(reference(d),prelib)
    with pytest.raises(ValueError,match='nonfinite'):c.transform([[2.,'b',2.]])


def test_exact_boundary_grid(prelib):
    values=[0.,-0.,math.nextafter(0.,1.),-math.nextafter(0.,1.),math.nextafter(1.,2.),
            math.nextafter(1.,0.),1e-300,1e150,math.nan,None]
    d=definition();d['operations'][2]['scale']=(3.).hex()
    p=reference(d);c=NativePreprocessor(p,prelib)
    rows=[[x,'b',z] for x in values for z in values]
    assert c.transform(rows).tobytes()==p.transform(rows).tobytes()


def test_capsule_and_direct_input_rejections(compiled):
    mod=compiled._module
    with pytest.raises(ValueError):mod.transform(None,[])
    for bad in ((),[None],[[1.,'b',0.]],[tuple([0.]*4)]):
        with pytest.raises(ValueError):mod.transform(compiled._plan,bad)


@pytest.mark.parametrize('args', [
    (0,1,1,(),()),(1,4097,1,(),()),(1,1,0,(),()),(4096,4096,65536,(),()),
    (True,1,1,(),()),(1,1,1,[],()),(1,1,1,((0,0,0.,0.,0.),),()),
    (1,1,1,((1,0,0.,0.,1.),),()),(1,1,1,((0,0,math.nan,0.,1.),),()),
    (1,1,1,((0,0,0,0.,1.),),()),(1,2,1,((0,0,0.,0.,1.),),()),
    (1,2,1,(),((0,0,('a','a'),False),)),(1,1,1,(),((0,0,(1,),False),)),
    (1,1,1,(),((0,0,('a',),1),)),(1,1,1,((0,0,0.,0.,1.),),((0,0,('a',),False),))])
def test_bad_native_plans(compiled,args):
    with pytest.raises((ValueError,TypeError,OverflowError)):compiled._module.prepare(*args)


def test_build_refuses_overwrite(prelib):
    with pytest.raises(FileExistsError):build_preprocessor(prelib.parent)


def test_wrong_abi_and_reference_type(prelib,tmp_path):
    with pytest.raises(ValueError):NativePreprocessor(object(),prelib)
    p=tmp_path/'wrong.so';p.write_bytes(b'not executable')
    with pytest.raises(ValueError,match='ABI'):NativePreprocessor(reference(),p)


def test_non_nearest_rejected(compiled):
    import ctypes
    lib=ctypes.CDLL(None);lib.fegetround.restype=ctypes.c_int
    original=lib.fegetround()
    # Linux/glibc FE_DOWNWARD; scope is the already supported Linux platform.
    try:
        assert lib.fesetround(0x400)==0
        with pytest.raises(ValueError,match='round-to-nearest'):compiled.transform([[0.,'α',0.]])
    finally:assert lib.fesetround(original)==0


def test_optional_pipeline_end_to_end(prelib,library,tmp_path):
    from spectra.svm_shared import PreparedModel
    import hashlib,zlib
    folder=tmp_path/'bundle';folder.mkdir()
    # Legacy 2-class,16-feature finite fixture; no sklearn needed.
    payload=struct.pack('<dII',.5,1,1)+struct.pack('<35d',*([0.]*35))
    raw=struct.pack('<8sIIII',b'SPCSVM01',2,2,len(payload),zlib.crc32(payload))+payload
    (folder/'model.srt').write_bytes(raw)
    doc={'schema':SCHEMA,'columns':['x'],'features':16,'model_sha256':hashlib.sha256(raw).hexdigest(),
         'operations':[{'kind':'numeric','column':0,'fill':(0.).hex(),'mean':(0.).hex(),'scale':(1.).hex()}]*16}
    (folder/'preprocessing.json').write_text(json.dumps(doc))
    model=PreparedPipeline(folder,library,preprocessor_library=prelib);worker=model.session()
    model.close()
    assert worker.predict_many([[0.],[1.],[None]])==[1,1,1]
    assert worker.predict_with_certificate([1.]).label==1
    worker.close()
    with pytest.raises(ValueError):worker.predict([0.])


def test_fast_raw_containers_and_single_use_fallback(compiled):
    rows=[[1.,'b',2.],[None,'α',-0.]]
    expected=reference().transform(rows).tobytes()
    assert compiled.transform(rows).tobytes()==expected
    assert compiled.transform(tuple(map(tuple,rows))).tobytes()==expected
    assert compiled.transform(iter(map(iter,rows))).tobytes()==expected
    assert compiled._module.transform_raw(compiled._plan,iter(rows)) is NotImplemented


def test_raw_subclass_iteration_is_preserved(compiled):
    class Row(list):
        def __iter__(self):return iter([5.,'α',9.])
    class Batch(list):
        def __iter__(self):return iter([[3.,'b',8.]])
    for rows in ([Row([1.,'b',2.])],Batch([[0.,'',0.]])):
        assert compiled.transform(rows).tobytes()==reference().transform(rows).tobytes()


def test_native_late_error_returns_no_partial_result(compiled):
    valid=compiled.transform([[0.,'b',1.]]).tobytes()
    with pytest.raises(ValueError):compiled.transform([[0.,'b',1.],[math.inf,'b',0.]])
    assert compiled.transform([[0.,'b',1.]]).tobytes()==valid


def test_native_repeated_calls_do_not_reuse_outputs(compiled):
    a=compiled.transform([[1.,'b',2.]]);saved=a.tobytes()
    for i in range(100):compiled.transform([[float(i),'α',-float(i)]])
    assert a.tobytes()==saved
