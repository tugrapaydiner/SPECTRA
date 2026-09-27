"""Export a supported fitted sklearn preprocessing+SVC pair into inert files.

Explicit trusted-training-environment operation. Never imported by inference.
Reject unsupported transformations instead of silently approximating a pipeline.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

from .svm_pipeline import Preprocessor, SCHEMA, MAX_PLAN_BYTES
from .svm_export import export_prepared_svc


def export_plan(preprocessor, *, model_sha256: str, features: int) -> bytes:
    """Return a strictly validated plan for stock, fitted, dense transformers.

    The source objects must not be mutated concurrently during export. Numerical
    transforms are median-impute then centered/scaled StandardScaler. One-hot
    categories must be strings, with no dropped/grouped levels. Domain feature
    extraction and categorical missing-token construction remain caller duties.
    """
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.utils.validation import check_is_fitted
    if type(preprocessor) is not ColumnTransformer:
        raise ValueError('expected stock fitted ColumnTransformer')
    check_is_fitted(preprocessor)
    def reject_overrides(obj):
        if any(name in vars(obj) for name in ('transform', 'fit_transform')):
            raise ValueError('instance-overridden transformations are unsupported')
    reject_overrides(preprocessor)
    if preprocessor.remainder != 'drop' or preprocessor.transformer_weights is not None or preprocessor.sparse_output_:
        raise ValueError('only dense, unweighted preprocessing with dropped remainder is supported')
    if not hasattr(preprocessor, 'feature_names_in_'):
        raise ValueError('export requires fitted unique string input names')
    columns = preprocessor.feature_names_in_.tolist()
    if any(type(x) is not str for x in columns) or len(set(columns)) != len(columns):
        raise ValueError('unique fitted string input names required')
    operations = []
    for name, transformer, selected in preprocessor.transformers_:
        if isinstance(transformer, str):
            if transformer == 'drop':
                continue
            raise ValueError('passthrough and custom transforms are unsupported')
        if not isinstance(selected, (list, tuple, np.ndarray)):
            raise ValueError('explicit fitted column selection required')
        indices = []
        for c in selected:
            if isinstance(c, str) and c in columns:
                indices.append(columns.index(c))
            elif isinstance(c, (int, np.integer)) and not isinstance(c, (bool, np.bool_)) and 0 <= int(c) < len(columns):
                indices.append(int(c))
            else:
                raise ValueError('invalid selected column')
        if not indices:
            continue
        reject_overrides(transformer)
        if type(transformer) is Pipeline:
            if len(transformer.steps) != 2:
                raise ValueError('expected imputer followed by scaler')
            imputer, scaler = (step for _, step in transformer.steps)
            if type(imputer) is not SimpleImputer or type(scaler) is not StandardScaler:
                raise ValueError('unsupported numeric steps')
            check_is_fitted(imputer); check_is_fitted(scaler)
            reject_overrides(imputer); reject_overrides(scaler)
            if (imputer.strategy != 'median' or imputer.add_indicator or
                not isinstance(imputer.missing_values, (float, np.floating)) or not np.isnan(imputer.missing_values) or
                not scaler.with_mean or not scaler.with_std):
                raise ValueError('unsupported numeric settings')
            arrays = (imputer.statistics_, scaler.mean_, scaler.scale_)
            if (any(np.asarray(a).shape != (len(indices),) or not np.isfinite(a).all() for a in arrays)
                or (scaler.scale_ <= 0).any()):
                raise ValueError('empty/dropped columns or nonfinite numeric parameters')
            if int(scaler.n_features_in_) != len(indices) or int(imputer.n_features_in_) != len(indices):
                raise ValueError('numeric feature inventory mismatch')
            for i, c in enumerate(indices):
                operations.append({'kind':'numeric', 'column':c, 'fill':float(arrays[0][i]).hex(),
                                   'mean':float(arrays[1][i]).hex(), 'scale':float(arrays[2][i]).hex()})
        elif type(transformer) is OneHotEncoder:
            check_is_fitted(transformer)
            if (transformer.sparse_output or np.dtype(transformer.dtype) != np.dtype('float64') or
                transformer.drop is not None or transformer.drop_idx_ is not None or
                transformer.min_frequency is not None or transformer.max_categories is not None or
                transformer.handle_unknown not in ('ignore','error')):
                raise ValueError('unsupported one-hot configuration')
            if len(transformer.categories_) != len(indices):
                raise ValueError('category feature inventory mismatch')
            for c, categories in zip(indices, transformer.categories_):
                values = categories.tolist()
                if any(type(value) is not str for value in values):
                    raise ValueError('one-hot categories must be strings, not numeric/missing objects')
                operations.append({'kind':'onehot','column':c,'categories':values,'unknown':transformer.handle_unknown})
        else:
            raise ValueError('unsupported transformer type')
    document = {'schema':SCHEMA,'columns':columns,'features':features,
                'model_sha256':model_sha256,'operations':operations}
    raw = (json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(',',':'), allow_nan=False)+'\n').encode('utf-8')
    if len(raw) > MAX_PLAN_BYTES:
        raise ValueError('exported plan exceeds byte limit')
    Preprocessor(raw)
    return raw


def export_pipeline(preprocessor, classifier, destination: str | Path) -> dict:
    """Create a fresh deployable directory. Never overwrite a previous export.

    No fitting is performed. Partial failures remain in their directory for
    inspection; only a fully written preprocessing/model pair can be loaded.
    Export-time model loading is trusted, but deployment does not deserialize code.
    """
    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=False)
    model = export_prepared_svc(classifier, folder/'model.srt')
    raw = export_plan(preprocessor, model_sha256=model['sha256'], features=model['features'])
    with (folder/'preprocessing.json').open('xb') as stream:
        stream.write(raw)
    return {'schema':SCHEMA,'model':model,'preprocessing_bytes':len(raw),
            'preprocessing_sha256':hashlib.sha256(raw).hexdigest()}
