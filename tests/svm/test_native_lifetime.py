"""Reentrant-close regressions without dereferencing freed native memory.

Dispatch hooks run at a boundary where a Python signal/audit callback can run.
The guard assertion stops the old implementation before it can use the freed
pointer. These are lifecycle tests, not benchmark or arbitrary-signal proofs.
"""
from array import array
import ctypes as C
import struct
import zlib

import pytest
from spectra.svm import Session
from spectra.svm_shared import PreparedModel


def model_file(path):
    classes = 3
    payload = struct.pack('<dIII', .5, 1, 1, 1)
    values = [0.] * (16 * 3 + 2 * 3 + 3)
    payload += struct.pack('<' + 'd' * len(values), *values)
    path.write_bytes(struct.pack('<8sIIII', b'SPCSVM01', classes, 3, len(payload), zlib.crc32(payload)) + payload)
    return path


@pytest.fixture(params=['legacy', 'shared'])
def owned(request, tmp_path, library):
    path = model_file(tmp_path / 'model.srt')
    if request.param == 'legacy':
        obj = Session(path, library)
        yield obj, 'sp_svm_single', 'et_destroy'
        obj.close()
    else:
        with PreparedModel(path, library, input_dtype='float64') as owner:
            obj = owner.session()
        yield obj, 'sp_worker_run', 'sp_worker_destroy'
        obj.close()


@pytest.mark.parametrize('certificate', [False, True])
def test_close_at_native_entry_keeps_borrowed_pointer_alive(owned, monkeypatch, certificate):
    obj, run_name, _ = owned
    original = getattr(obj._lib, run_name)
    saved = obj._handle
    seen = []

    def dispatch(*args):
        obj.close()
        seen.append(obj._handle == saved)
        # Deliberately prevent a use-after-free on the unpatched baseline.
        assert obj._handle == saved, 'close destroyed the handle already passed into dispatch'
        return original(*args)

    monkeypatch.setattr(obj._lib, run_name, dispatch)
    method = obj.predict_with_certificate if certificate else obj.predict
    answer = method([0.] * 16)
    assert (answer.class_index if certificate else answer) == 2
    assert seen == [True]
    assert obj._handle is None


@pytest.mark.parametrize('where', ['before', 'after'])
def test_nested_inference_is_rejected_through_result_validation(owned, monkeypatch, where):
    obj, run_name, _ = owned
    original = getattr(obj._lib, run_name)
    entered = False

    def dispatch(*args):
        nonlocal entered
        if entered:
            return original(*args)
        entered = True
        if where == 'after':
            result = original(*args)
        with pytest.raises(ValueError, match='reentrant'):
            obj.predict([1.] * 16)
        return result if where == 'after' else original(*args)

    monkeypatch.setattr(obj._lib, run_name, dispatch)
    assert obj.predict_with_certificate([0.] * 16).class_index == 2


def test_destruction_detaches_pointer_before_reentrant_close(owned, monkeypatch):
    obj, _, destroy_name = owned
    original = getattr(obj._lib, destroy_name)
    calls = []

    def destroy(handle):
        calls.append(handle)
        if len(calls) == 1:
            obj.close()
            original(handle)
        # The old implementation re-enters here. Do not double-free in the test.

    monkeypatch.setattr(obj._lib, destroy_name, destroy)
    obj.close()
    assert len(calls) == 1, 'reentrant close attempted to destroy the same pointer twice'
    assert obj._handle is None


def test_prepared_owner_close_during_worker_creation(tmp_path, library, monkeypatch):
    model = PreparedModel(model_file(tmp_path / 'model.srt'), library, input_dtype='float64')
    original = model._lib.sp_worker_create
    pointer = model._handle

    def create(handle):
        model.close()
        assert model._handle == pointer, 'owner destroyed while its pointer was being borrowed'
        return original(handle)

    monkeypatch.setattr(model._lib, 'sp_worker_create', create)
    with model.session() as worker:
        assert worker.predict([0.] * 16) == 2
    assert model._handle is None


def test_buffer_close_at_native_entry(tmp_path, library, monkeypatch):
    with PreparedModel(model_file(tmp_path / 'model.srt'), library, input_dtype='float64') as owner:
        worker = owner.session()
    original = worker._lib.sp_worker_run
    pointer = worker._handle

    def dispatch(*args):
        worker.close()
        assert worker._handle == pointer, 'buffer dispatch lost its native owner'
        return original(*args)

    monkeypatch.setattr(worker._lib, 'sp_worker_run', dispatch)
    assert worker.predict_buffer(array('d', [0.] * 80)) == [2] * 5
    assert worker._handle is None


def test_closed_resource_released_after_last_lease(tmp_path, library, monkeypatch):
    """A saved local lease, not the public integer handle, owns destruction."""
    owner = PreparedModel(model_file(tmp_path / 'model.srt'), library, input_dtype='float64')
    worker = owner.session()
    resource = worker._resource
    original = worker._lib.sp_worker_destroy
    destroyed = []
    def destroy(pointer):
        destroyed.append(pointer)
        original(pointer)
    monkeypatch.setattr(worker._lib, 'sp_worker_destroy', destroy)
    pointer = worker._handle
    worker.close()
    assert worker._handle is None and not destroyed
    with pytest.raises(ValueError, match='closed'):
        worker.predict([0.] * 16)
    del resource
    assert destroyed == [pointer]
    owner.close()


def test_exception_clears_active_request_and_applies_close(owned, monkeypatch):
    obj, run_name, _ = owned
    original = getattr(obj._lib, run_name)
    def dispatch(*args):
        obj.close()
        raise KeyboardInterrupt('controlled native boundary')
    monkeypatch.setattr(obj._lib, run_name, dispatch)
    with pytest.raises(KeyboardInterrupt):
        obj.predict([0.] * 16)
    assert not obj._native_active and obj._handle is None
    monkeypatch.setattr(obj._lib, run_name, original)
    with pytest.raises(ValueError, match='closed'):
        obj.predict([0.] * 16)


def test_failed_request_without_close_can_be_reused(owned, monkeypatch):
    obj, run_name, _ = owned
    original = getattr(obj._lib, run_name)
    def dispatch(*args):
        raise KeyboardInterrupt('controlled native boundary')
    monkeypatch.setattr(obj._lib, run_name, dispatch)
    with pytest.raises(KeyboardInterrupt):
        obj.predict([0.] * 16)
    assert not obj._native_active and obj._handle
    monkeypatch.setattr(obj._lib, run_name, original)
    assert obj.predict_with_certificate([0.] * 16).class_index == 2
