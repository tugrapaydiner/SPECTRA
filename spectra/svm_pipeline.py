"""Framework-free, bounded execution of an explicitly exported preprocessing plan.

This is a restricted dense RBF-SVC pipeline, not arbitrary Python deserialization.
Training/export may use sklearn; loading and predicting use only the standard
library and SPECTRA's explicitly built native runtime. Numerical operators retain
separate impute/subtract/divide steps. Strings are schema values, never code.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import numbers
from pathlib import Path
import re
import threading
from types import MappingProxyType
from typing import Iterable, Mapping

from .svm_shared import PreparedModel, MAX_ELEMENTS

MAX_PLAN_BYTES = 4 * 1024 * 1024
MAX_COLUMNS = 4096
MAX_ROWS = 65536
SCHEMA = 'spectra.preprocessing.v1'


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate plan key: {key}')
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f'nonfinite JSON token: {value}')


def _name(value):
    if type(value) is not str or not value or len(value.encode('utf-8')) > 4096:
        raise ValueError('column names must be nonempty bounded UTF-8 strings')
    return value


def _float(value):
    if type(value) is not str or len(value) > 32:
        raise ValueError('expected canonical binary64 hex string')
    try:
        result = float.fromhex(value)
    except (ValueError, OverflowError) as error:
        raise ValueError('invalid binary64 constant') from error
    if not math.isfinite(result) or result.hex() != value:
        raise ValueError('constant must be canonical finite binary64')
    return result


@dataclass(frozen=True)
class _Numeric:
    column: int
    output: int
    fill: float
    mean: float
    scale: float


@dataclass(frozen=True)
class _Category:
    column: int
    offset: int
    lookup: Mapping[str, int]
    unknown: str


class Preprocessor:
    """Immutable plan; each transform allocates its own bounded binary64 output.

    Rows are sequences in ``columns`` order, not mappings/DataFrames. Values for
    numeric columns are real numbers or None/NaN (training-fitted imputation).
    Categorical values are strings. An application's conversion to a missing
    categorical token, units, or feature extraction is NOT silently performed.
    """
    def __init__(self, raw: bytes):
        if type(raw) is not bytes or not 1 <= len(raw) <= MAX_PLAN_BYTES:
            raise ValueError('invalid preprocessing byte count')
        try:
            doc = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique,
                             parse_constant=_nonfinite)
            self._initialize(doc)
        except (UnicodeError, RecursionError, TypeError, KeyError) as error:
            raise ValueError('invalid preprocessing plan') from error
        self._sha256 = hashlib.sha256(raw).hexdigest()

    def _initialize(self, doc):
        if type(doc) is not dict or set(doc) != {'schema', 'columns', 'features', 'model_sha256', 'operations'}:
            raise ValueError('unexpected preprocessing fields')
        if doc['schema'] != SCHEMA:
            raise ValueError('unsupported preprocessing schema')
        columns = doc['columns']
        if type(columns) is not list or not 1 <= len(columns) <= MAX_COLUMNS:
            raise ValueError('invalid input columns')
        self._columns = tuple(_name(name) for name in columns)
        if len(set(self._columns)) != len(self._columns):
            raise ValueError('duplicate input columns')
        self._features = doc['features']
        if type(self._features) is not int or not 1 <= self._features <= MAX_COLUMNS:
            raise ValueError('invalid transformed feature count')
        self._model_sha256 = doc['model_sha256']
        if type(self._model_sha256) is not str or not re.fullmatch('[0-9a-f]{64}', self._model_sha256):
            raise ValueError('invalid model digest')
        ops = doc['operations']
        if type(ops) is not list or not 1 <= len(ops) <= MAX_COLUMNS:
            raise ValueError('invalid operation count')
        numerical, categorical = [], []
        offset = 0
        for op in ops:
            if type(op) is not dict or type(op.get('column')) is not int or not 0 <= op['column'] < len(columns):
                raise ValueError('invalid operation column')
            if op.get('kind') == 'numeric':
                if set(op) != {'kind', 'column', 'fill', 'mean', 'scale'}:
                    raise ValueError('invalid numeric operation')
                fill, mean, scale = (_float(op[key]) for key in ('fill', 'mean', 'scale'))
                if scale <= 0:
                    raise ValueError('scale must be positive')
                numerical.append(_Numeric(op['column'], offset, fill, mean, scale))
                offset += 1
            elif op.get('kind') == 'onehot':
                if set(op) != {'kind', 'column', 'categories', 'unknown'}:
                    raise ValueError('invalid one-hot operation')
                categories = op['categories']
                if type(categories) is not list or not 1 <= len(categories) <= MAX_COLUMNS:
                    raise ValueError('invalid categories')
                if any(type(c) is not str or len(c.encode('utf-8')) > 4096 for c in categories):
                    raise ValueError('categories must be bounded strings')
                if len(set(categories)) != len(categories) or op['unknown'] not in ('ignore', 'error'):
                    raise ValueError('duplicate categories or unsupported unknown handling')
                lookup = MappingProxyType({c: offset + i for i, c in enumerate(categories)})
                categorical.append(_Category(op['column'], offset, lookup, op['unknown']))
                offset += len(categories)
            else:
                raise ValueError('unsupported preprocessing operation')
            if offset > self.features:
                raise ValueError('plan exceeds transformed feature count')
        if offset != self.features:
            raise ValueError('plan feature inventory mismatch')
        self._numerical, self._categorical = tuple(numerical), tuple(categorical)
        self._row_cap = min(MAX_ROWS, MAX_ELEMENTS // max(len(columns), self.features))

    @classmethod
    def load(cls, path: str | Path):
        with Path(path).open('rb') as stream:
            raw = stream.read(MAX_PLAN_BYTES + 1)
        return cls(raw)

    @property
    def columns(self):
        return self._columns

    @property
    def features(self):
        return self._features

    @property
    def model_sha256(self):
        return self._model_sha256

    @property
    def sha256(self):
        return self._sha256

    def transform(self, rows: Iterable[Iterable[object]], *, columns=None) -> array:
        """Return fresh row-major binary64 features; no hidden fit or caller writes.

        ``columns`` can validate a known schema. Reordered input is rejected, not
        silently remapped. Iterators are consumed with row/element caps. Infinity,
        booleans and numeric strings are rejected in numeric columns. None/NaN is
        accepted there only because the plan explicitly stores an imputation value.
        """
        return self._transform_rows(self._materialize(rows, columns=columns))

    def _materialize(self, rows, *, columns=None):
        """One bounded snapshot shared by reference and optional compiled paths."""
        if columns is not None and tuple(itertools.islice(iter(columns), len(self.columns) + 1)) != self.columns:
            raise ValueError('input schema/order mismatch')
        if isinstance(rows, (str, bytes, Mapping)):
            raise ValueError('expected a sequence of row sequences')
        materialized = []
        for row in itertools.islice(iter(rows), self._row_cap + 1):
            if len(materialized) == self._row_cap:
                raise ValueError('raw batch exceeds row/element cap')
            if isinstance(row, (str, bytes, Mapping)):
                raise ValueError('expected an ordered row sequence')
            values = tuple(itertools.islice(iter(row), len(self.columns) + 1))
            if len(values) != len(self.columns):
                raise ValueError('raw row feature count mismatch')
            materialized.append(values)
        return materialized

    def _transform_rows(self, materialized):
        """Reference arithmetic on a checked, materialized batch."""
        output = array('d', [0.0]) * (len(materialized) * self.features)
        for index, values in enumerate(materialized):
            base = index * self.features
            for op in self._numerical:
                value = values[op.column]
                if value is None:
                    x = op.fill
                else:
                    # The common case needs no abstract-class dispatch or copy.
                    # Non-built-in reals still follow the original checked path.
                    if type(value) is float:
                        x = value
                    else:
                        if isinstance(value, bool) or not isinstance(value, numbers.Real):
                            raise ValueError('real numeric feature or explicit missing value required')
                        try:
                            x = float(value)
                        except (ValueError, TypeError, OverflowError) as error:
                            raise ValueError('invalid numeric feature') from error
                    if math.isnan(x):
                        x = op.fill
                    elif not math.isfinite(x):
                        raise ValueError('infinite raw feature')
                # Match StandardScaler's two rounded operations; do NOT turn
                # division into multiply-by-reciprocal or fold an affine map.
                value = (x - op.mean) / op.scale
                if not math.isfinite(value):
                    raise ValueError('nonfinite transformed feature')
                output[base + op.output] = value
            for op in self._categorical:
                value = values[op.column]
                if not isinstance(value, str):
                    raise ValueError('string categorical feature required')
                if len(value.encode('utf-8')) > 4096:
                    raise ValueError('categorical value exceeds byte limit')
                column = op.lookup.get(value)
                if column is not None:
                    output[base + column] = 1.0
                elif op.unknown == 'error':
                    raise ValueError('unknown categorical value')
        return output


class PreparedPipeline:
    """One fitted preprocessing plan and shared native model, never pickle.

    A directory must contain exactly-named preprocessing.json and model.srt.
    Their digest/feature binding is checked against the bytes the native model
    actually consumed. Hashes provide identity, NOT author authentication. Treat
    the bundle and the supplied executable library as trusted deployment inputs.
    """
    def __init__(self, folder: str | Path, library: str | Path, *, tables=False,
                 preprocessor_library: str | Path | None = None):
        self._lock = threading.RLock()
        self._closed = True
        self._model = None
        folder = Path(folder)
        self._preprocessor = Preprocessor.load(folder / 'preprocessing.json')
        if preprocessor_library is not None:
            from .svm_preprocess_native import NativePreprocessor
            self._preprocessor = NativePreprocessor(self._preprocessor, preprocessor_library)
        model = PreparedModel(folder / 'model.srt', library, tables=tables, input_dtype='float64')
        try:
            if model.sha256 != self.preprocessor.model_sha256 or model.features != self.preprocessor.features:
                raise ValueError('preprocessing/model binding mismatch')
            self._model = model
            self._closed = False
        except BaseException:
            model.close()
            raise

    @property
    def preprocessor(self):
        return self._preprocessor

    @property
    def columns(self):
        return self.preprocessor.columns

    @property
    def labels(self):
        return self._model.labels

    def session(self):
        with self._lock:
            if self._closed:
                raise ValueError('closed prepared pipeline')
            return PipelineSession(self.preprocessor, self._model.session())

    def close(self):
        with self._lock:
            if self._model is not None:
                self._model.close()
            self._closed = True

    def __enter__(self):
        with self._lock:
            if self._closed:
                raise ValueError('closed prepared pipeline')
        return self

    def __exit__(self, *args):
        self.close()


class PipelineSession:
    """Private native worker. Existing sessions survive prepared-owner close."""
    def __init__(self, preprocessor, worker):
        self._preprocessor, self._worker = preprocessor, worker
        self._fused_runner = None

    @property
    def columns(self):
        return self._preprocessor.columns

    def predict_many(self, rows, *, columns=None, schedule='beretta_cert', hint=-1):
        values = self._preprocessor.transform(rows, columns=columns)
        return self._worker.predict_buffer(values, schedule=schedule, hint=hint)

    def predict_fused(self, rows, *, columns=None, schedule='beretta_cert', hint=-1, tile_rows=128):
        """Optional compiled preprocessing and inference in one bounded-tile call.

        Requires an explicitly supplied compiled preprocessor. Ordinary built-in
        rows use at most min(tile_rows, 16384/features) transformed rows at once;
        feature scratch <=128 KiB. Input/output storage is additional. Custom
        objects/iterators use the existing full-batch reference fallback. The
        caller must not mutate rows while this method runs. No partial labels are
        returned on error; native internal work may already have occurred.
        """
        from .svm_preprocess_native import _FusedRunner
        with self._worker._lock:
            if self._fused_runner is None:
                self._fused_runner = _FusedRunner(self._preprocessor, self._worker)
            return self._fused_runner.predict(rows, columns=columns, schedule=schedule,
                                               hint=hint, tile_rows=tile_rows)

    def predict(self, row, *, columns=None, schedule='beretta_cert', hint=-1):
        return self.predict_many([row], columns=columns, schedule=schedule, hint=hint)[0]

    def predict_with_certificate(self, row, *, columns=None, schedule='beretta_cert', hint=-1):
        values = self._preprocessor.transform([row], columns=columns)
        return self._worker.predict_with_certificate(values, schedule=schedule, hint=hint)

    def close(self):
        self._worker.close()

    def __enter__(self):
        self._worker.__enter__()
        return self

    def __exit__(self, *args):
        self.close()
