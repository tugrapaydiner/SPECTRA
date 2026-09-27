"""Borrowed binary64 inputs: zero-copy lifetime and strict boundary tests."""
from array import array
from concurrent.futures import ThreadPoolExecutor
import ctypes as C
import math
import pytest
from spectra.svm_shared import PreparedModel, SHARED_SCHEDULES
from test_shared_model import encoded

@pytest.fixture
def worker(tmp_path, library):
    path=encoded(tmp_path/'model.srt',features=3,labels=['a','b','c'])
    with PreparedModel(path,library,input_dtype='float64') as model,model.session() as session:
        yield session

@pytest.mark.parametrize('schedule',tuple(SHARED_SCHEDULES))
def test_flat_buffer_and_input_preservation(worker,schedule):
    data=array('d',[0.,1.,2.,3.,4.,5.]);original=data.tobytes()
    assert worker.predict_buffer(data,schedule=schedule)==['c','c']
    assert data.tobytes()==original
    # No dangling export after method completion.
    data.append(9.)

@pytest.mark.parametrize('kind',['f32','integers','bytes','readonly','strided','incomplete','nonbuffer'])
def test_rejects_incompatible_buffers(worker,kind):
    data=array('d',[0.]*6)
    cases={'f32':array('f',[0.]*6),'integers':array('q',[0]*6),'bytes':bytearray(48),
           'readonly':memoryview(bytes(48)).cast('d'),'strided':memoryview(data)[::2],
           'incomplete':array('d',[0.]*5),'nonbuffer':[0.]*6}
    with pytest.raises(ValueError):worker.predict_buffer(cases[kind])

@pytest.mark.parametrize('kind',['wrong_width','fortran','big_endian','rank3','unaligned'])
def test_matrix_layout_rejections(worker,kind):
    np=pytest.importorskip('numpy')
    data={'wrong_width':np.zeros((3,2)), 'fortran':np.asfortranarray(np.zeros((2,3))),
          'big_endian':np.zeros((2,3),dtype='>f8'),'rank3':np.zeros((1,2,3)),
          'unaligned':np.ndarray((2,3),dtype='f8',buffer=bytearray(49),offset=1)}[kind]
    with pytest.raises(ValueError):worker.predict_buffer(data)

def test_contiguous_matrix_and_empty(worker):
    np=pytest.importorskip('numpy')
    x=np.arange(12,dtype=np.float64).reshape(4,3)
    assert worker.predict_buffer(x)==worker.predict_many(x)
    assert worker.predict_buffer(array('d'))==[]
    assert worker.predict_buffer(np.empty((0,3),dtype=np.float64))==[]

@pytest.mark.parametrize('value',[math.nan,math.inf,-math.inf])
def test_late_nonfinite_rejected_and_next_call_recovers(worker,value):
    worker.predict_with_certificate([0.,0.,0.])
    data=array('d',[0.]*5+[value])
    with pytest.raises(ValueError):worker.predict_buffer(data)
    trace=(C.c_int8*3)()
    assert worker._lib.sp_worker_certificate(worker._handle,trace,3)!=0
    assert worker.predict_buffer(array('d',[0.]*6))==['c','c']

def test_success_does_not_expose_last_row_certificate(worker):
    worker.predict_with_certificate([0.,0.,0.])
    worker.predict_buffer(array('d',[0.]*6))
    trace=(C.c_int8*3)()
    assert worker._lib.sp_worker_certificate(worker._handle,trace,3)!=0

def test_cap_closed_and_rounding_rejected(worker,tmp_path,library):
    with pytest.raises(ValueError):worker.predict_buffer(array('d',[0.])*(3*65537))
    path=encoded(tmp_path/'float.srt',features=3)
    with PreparedModel(path,library,input_dtype='float32') as m,m.session() as w:
        with pytest.raises(ValueError,match='float64'):w.predict_buffer(array('d',[0.]*3))
    worker.close()
    with pytest.raises(ValueError,match='closed'):worker.predict_buffer(array('d',[0.]*3))

@pytest.mark.parametrize('kwargs',[{'hint':True},{'hint':3},{'schedule':[]},{'schedule':'wrong'}])
def test_settings(worker,kwargs):
    with pytest.raises(ValueError):worker.predict_buffer(array('d',[0.]*3),**kwargs)

def test_workers_outlive_model_and_mixed_calls_are_serialized(tmp_path,library):
    path=encoded(tmp_path/'model.srt',features=3)
    owner=PreparedModel(path,library,input_dtype='float64');workers=[owner.session() for _ in range(4)]
    owner.close();path.unlink()
    data=array('d',[0.]*96)
    def call(i):
        w=workers[i%4]
        return w.predict_buffer(data) if i%2 else w.predict_many([[0.]*3]*32)
    try:
        with ThreadPoolExecutor(4) as pool:
            assert all(x==[2]*32 for x in pool.map(call,range(64)))
    finally:
        for w in workers:w.close()

def test_no_python_packing_and_export_lifetime(worker,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('copying pack path used')
    monkeypatch.setattr(worker,'_pack',forbidden)
    data=array('d',[0.]*3)
    assert worker.predict_buffer(data)==['c']
    data.extend([1.,2.,3.])
