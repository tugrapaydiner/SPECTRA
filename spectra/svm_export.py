"""Explicit export from a trusted, fitted sklearn RBF SVC; never used at inference."""
from __future__ import annotations
import hashlib
from pathlib import Path
import struct
import zlib


def export_svc(model, destination: str | Path) -> dict:
    """Preserve binary64 weights in SPC SVM01 format, refusing overwrite.

    Requires scikit-learn/NumPy only in the exporting environment. Supported
    labels are exactly 0..C-1, and inputs are 16 features. Probability output and
    confidence-based tie breaking are not part of the runtime contract. Inference
    accepts binary32 inputs widened to binary64; compare sklearn on those SAME
    rounded inputs rather than arbitrarily higher-precision originals.
    """
    import numpy as np
    from sklearn.svm import SVC
    from sklearn.utils.validation import check_is_fitted
    if type(model) is not SVC:
        raise ValueError('expected a stock sklearn.svm.SVC')
    check_is_fitted(model)
    if model.kernel != 'rbf' or model._sparse or model.break_ties:
        raise ValueError('expected dense RBF SVC with default vote tie handling')
    classes, supports = len(model.classes_), len(model.support_vectors_)
    if not 2 <= classes <= 128 or not 1 <= supports <= 100000:
        raise ValueError('unsupported classes or support count')
    if not np.array_equal(model.classes_, np.arange(classes)):
        raise ValueError('class labels must be exactly 0 through C-1')
    arrays = (model.support_vectors_, model.dual_coef_, model.intercept_)
    shapes = ((supports, 16), (classes - 1, supports), (classes * (classes - 1) // 2,))
    if any(a.shape != shape or not np.isfinite(a).all() for a, shape in zip(arrays, shapes)):
        raise ValueError('invalid parameter geometry or nonfinite parameters')
    counts = np.asarray(model.n_support_)
    if counts.shape != (classes,) or (counts <= 0).any() or int(counts.sum()) != supports:
        raise ValueError('invalid support inventory')
    gamma = float(model._gamma)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError('invalid RBF gamma')
    raw = struct.pack('<d', gamma) + np.asarray(counts, dtype='<u4').tobytes()
    raw += b''.join(np.asarray(a, dtype='<f8', order='C').tobytes() for a in arrays)
    if len(raw) + 24 > 64 * 1024 * 1024:
        raise ValueError('model exceeds the native loader byte limit')
    encoded = struct.pack('<8sIIII', b'SPCSVM01', classes, supports, len(raw), zlib.crc32(raw)) + raw
    with Path(destination).open('xb') as stream:
        stream.write(encoded)
    return {'format': 'SPCSVM01', 'classes': classes, 'support_vectors': supports,
            'bytes': len(encoded), 'sha256': hashlib.sha256(encoded).hexdigest()}
