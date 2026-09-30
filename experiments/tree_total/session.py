"""Total source-arithmetic inference. Diagnostic certificate_only can abstain."""
from __future__ import annotations
import ctypes as C
from pathlib import Path
from spectra.svm_lifetime import _NativeOwner
from .compiler import VerifiedTotal

POLICIES = {'total': 0, 'certificate_only': 1, 'exact': 2, 'audit': 3}
FIELDS = ('coarse_certified', 'exact_completed', 'unresolved',
          'routed_trees', 'coarse_leaf_vectors', 'exact_leaf_vectors')


class TotalSession(_NativeOwner):
    def __init__(self, model: VerifiedTotal, library):
        self._init_lifetime('verified total trees')
        if type(model) is not VerifiedTotal:
            raise ValueError('source-verified total model required')
        self.features = model.info['features']
        self.maximum = model.info['maximum']
        self.classes = model.info['classes']
        self.source_sha256 = model.info['source_sha256']
        self.model_sha256 = model.info['model_sha256']
        self._lib = lib = C.CDLL(str(Path(library).resolve(strict=True)))
        lib.tt_abi.restype = C.c_int
        if lib.tt_abi() != 1:
            raise ValueError('unsupported total ABI')
        lib.tc_error.restype = C.c_char_p
        lib.tt_create.argtypes = [C.c_void_p, C.c_uint64]
        lib.tt_create.restype = C.c_void_p
        lib.tt_destroy.argtypes = [C.c_void_p]
        lib.tt_destroy.restype = None
        lib.tt_info.argtypes = [C.c_void_p, C.POINTER(C.c_uint64), C.c_int]
        lib.tt_info.restype = C.c_int
        lib.tt_run.argtypes = [C.c_void_p, C.POINTER(C.c_uint8), C.c_int, C.c_int,
                              C.c_int, C.c_int, C.POINTER(C.c_int32),
                              C.POINTER(C.c_uint64), C.c_int]
        lib.tt_run.restype = C.c_int
        lib.tt_scores.argtypes = [C.c_void_p, C.POINTER(C.c_uint8), C.c_int, C.c_int,
                                  C.c_int, C.POINTER(C.c_double), C.c_uint64]
        lib.tt_scores.restype = C.c_int
        blob = C.create_string_buffer(model.raw, len(model.raw))
        handle = lib.tt_create(blob, len(model.raw))
        if not handle:
            raise ValueError(lib.tc_error().decode(errors='replace'))
        self._adopt(handle, lib, 'tt_destroy')
        with self._operation() as ptr:
            info = (C.c_uint64 * 9)()
            self._check(lib.tt_info(ptr, info, 9))
            self.info = dict(zip(('features', 'maximum', 'classes', 'trees',
                                 'leaf_vectors', 'unique_leaf_vectors', 'index_width',
                                 'source_bank_bytes', 'prepared_bytes'), info))

    def _check(self, status):
        if status:
            raise ValueError(self._lib.tc_error().decode(errors='replace'))

    def _invoke(self, values, policy, traversal, inspect):
        if type(policy) is not str or policy not in POLICIES:
            raise ValueError('unknown total policy')
        if type(traversal) is not str or traversal not in ('scalar', 'tiled'):
            raise ValueError('unknown traversal')
        try:
            view = memoryview(values)
        except TypeError as error:
            raise ValueError('uint8 buffer required') from error
        data = None
        try:
            if view.format != 'B' or view.readonly or not view.c_contiguous or view.ndim not in (1, 2):
                raise ValueError('writable contiguous uint8 buffer required')
            if view.ndim == 2 and view.shape[1] != self.features:
                raise ValueError('wrong feature width')
            rows = view.nbytes // self.features
            if view.nbytes % self.features or view.nbytes > 8000000 or rows > 65536:
                raise ValueError('input shape or row limit')
            data = (C.c_uint8 * view.nbytes).from_buffer(view) if view.nbytes else None
            with self._operation() as ptr:
                out = (C.c_int32 * rows)()
                stats = (C.c_uint64 * 6)()
                self._check(self._lib.tt_run(ptr, data, rows, self.features, POLICIES[policy],
                                            int(traversal == 'tiled'), out, stats, 6))
                indices = list(out)
                if any(not (-1 if policy == 'certificate_only' else 0) <= i < self.classes for i in indices):
                    raise ValueError('invalid native class index')
                return {'indices': indices, 'work': dict(zip(FIELDS, stats))} if inspect else indices
        finally:
            del data
            view.release()

    def predict_buffer(self, values, *, policy='total', traversal='tiled'):
        return self._invoke(values, policy, traversal, False)

    def inspect_buffer(self, values, *, policy='total', traversal='tiled'):
        return self._invoke(values, policy, traversal, True)

    def scores(self, values, *, traversal='tiled'):
        """All ordered source scores; always executes original source arithmetic."""
        if traversal not in ('scalar', 'tiled') or type(traversal) is not str:
            raise ValueError('unknown traversal')
        try:
            view = memoryview(values)
        except TypeError as error:
            raise ValueError('uint8 buffer required') from error
        data = None
        try:
            if view.format != 'B' or view.readonly or not view.c_contiguous or view.ndim not in (1, 2):
                raise ValueError('writable contiguous uint8 buffer required')
            if view.ndim == 2 and view.shape[1] != self.features:
                raise ValueError('wrong feature width')
            rows = view.nbytes // self.features
            cells = rows * self.classes
            if view.nbytes % self.features or view.nbytes > 8000000 or rows > 65536 or cells > 8000000:
                raise ValueError('input/output shape or row cap')
            data = (C.c_uint8 * view.nbytes).from_buffer(view) if view.nbytes else None
            with self._operation() as ptr:
                out = (C.c_double * cells)()
                self._check(self._lib.tt_scores(ptr, data, rows, self.features, int(traversal == 'tiled'), out, cells))
                return bytes(out)
        finally:
            del data
            view.release()
