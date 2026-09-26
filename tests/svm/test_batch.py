from array import array
import concurrent.futures
import ctypes as C
import itertools
import math
import pytest
from spectra.svm import Session, SCHEDULES
from test_runtime import write_model, oracle


@pytest.mark.parametrize('schedule', list(SCHEDULES))
@pytest.mark.parametrize('tables', [False, True])
def test_batch_matches_single_and_oracle(tmp_path, library, schedule, tables):
    model = write_model(tmp_path / 'm.srt', 10)
    rows = [[((i * 5 + j) % 29) / 29 for j in range(16)] for i in range(37)]
    expected = [oracle(model, row) for row in rows]
    with Session(model, library, tables=tables) as s:
        assert s.predict_many(iter(rows), schedule=schedule, hint=9) == expected
        assert [s.predict(row, schedule=schedule, hint=9) for row in rows] == expected
        assert s.predict_many([]) == []
        assert s.predict_many(rows[::-1], schedule=schedule) == expected[::-1]


def test_native_late_invalid_input_is_atomic_and_invalidates_certificate(tmp_path, library):
    with Session(write_model(tmp_path / 'm.srt'), library) as s:
        s.predict_with_certificate([0.] * 16)
        inputs = (C.c_float * 32)(*([0.] * 31 + [float('nan')]))
        out = (C.c_int * 2)(-91, -92)
        assert s._lib.sp_svm_batch(s._handle, inputs, 2, 16, 5, -1, out, 2) != 0
        assert list(out) == [-91, -92]
        trace = (C.c_int8 * 3)()
        assert s._lib.et_certificate(s._handle, trace, 3) != 0
        assert s.predict_many([[0.] * 16])[0] == s.predict([0.] * 16)


@pytest.mark.parametrize('rows,features,capacity', [(-1, 16, -1), (65537, 16, 65537),
                                                   (1, 15, 1), (1, 16, 0)])
def test_native_shape_rejections_before_dereference(tmp_path, library, rows, features, capacity):
    with Session(write_model(tmp_path / 'm.srt'), library) as s:
        assert s._lib.sp_svm_batch(s._handle, None, rows, features, 5, -1, None, capacity) != 0


def test_batch_never_exposes_last_row_as_a_single_certificate(tmp_path, library):
    with Session(write_model(tmp_path / 'm.srt'), library) as s:
        s.predict_many([[0.] * 16, [1.] * 16])
        trace = (C.c_int8 * 3)()
        assert s._lib.et_certificate(s._handle, trace, 3) != 0
        s.predict_with_certificate([0.] * 16)
        assert s._lib.et_certificate(s._handle, trace, 3) == 0
        out, stats = C.c_int(), (C.c_uint64 * 10)()
        assert s._lib.sp_svm_single(s._handle, None, 16, 99, -1, C.byref(out), stats, 10) != 0
        assert s._lib.et_certificate(s._handle, trace, 3) != 0


def test_batch_errors_and_limit(tmp_path, library):
    with Session(write_model(tmp_path / 'm.srt'), library) as s:
        with pytest.raises(ValueError):s.predict_many([[0.] * 16, [0.] * 15])
        with pytest.raises(ValueError):s.predict_many(itertools.repeat([0.] * 16, 65537))
        with pytest.raises(ValueError):s.predict_many([], schedule='unsafe')
        s.close()
        with pytest.raises(ValueError):s.predict_many([])


def test_mixed_concurrent_batch_and_single_calls(tmp_path, library):
    with Session(write_model(tmp_path / 'm.srt', 10), library, tables=True) as s:
        rows = [[i / 17.] * 16 for i in range(17)]
        expected = [s.predict(row) for row in rows]
        def run(index):
            if index % 2:
                return s.predict_many(rows)
            return [s.predict_with_certificate(row).class_index for row in rows]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            assert list(pool.map(run, range(80))) == [expected] * 80
