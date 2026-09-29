"""Strict bundle identity and source-label validation, before native loading.

This detects incomplete or accidentally mixed bundles, not a hostile publisher
who rewrites the manifest, source and bindings together. Exact numerical proof
reconstruction is still required by VerifiedCompact after this check.
"""
from __future__ import annotations
import hashlib
import json
import math
import re
import struct
from pathlib import Path
from typing import Any

MODEL_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z')
DIGEST = re.compile(r'[0-9a-f]{64}\Z')
MAX_FILE = 64 * 1024**2


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for raw in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(raw)
    return h.hexdigest()


def load_json(path: Path, limit: int = 1024 * 1024) -> Any:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result

    def finite(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError('nonfinite JSON number')
        return value

    def constant(_):
        raise ValueError('nonfinite JSON token')

    with Path(path).open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('JSON byte cap exceeded')
    try:
        return json.loads(raw.decode('utf-8'), object_pairs_hook=unique,
                          parse_float=finite, parse_constant=constant)
    except (UnicodeError, RecursionError) as exc:
        raise ValueError('invalid JSON encoding or nesting') from exc


def source_labels(document: dict) -> list[int | str]:
    """Supported explicit CatBoost output-index order, not sorted by display label."""
    try:
        params = document['model_info']['class_params']
        labels = params['class_names']
        mapping = params['class_to_label']
    except (KeyError, TypeError) as exc:
        raise ValueError('source needs explicit class_names and class_to_label') from exc
    if (type(labels) is not list or not 2 <= len(labels) <= 64 or
            type(mapping) is not list or any(type(i) is not int for i in mapping) or
            mapping != list(range(len(labels)))):
        raise ValueError('unsupported source class-index mapping')
    valid = (all(type(v) is int and -(2**63) <= v < 2**63 for v in labels) or
             all(type(v) is str and len(v.encode('utf-8')) <= 256 for v in labels))
    if not valid or len(set(labels)) != len(labels):
        raise ValueError('invalid source labels')
    return labels


def _bounded(value: Any, low: int, high: int, name: str) -> None:
    if type(value) is not int or not low <= value <= high:
        raise ValueError('invalid ' + name)


def verify_manifest(root: Path) -> dict:
    """Validate file closure, model roles and original label mapping.

    Successful hashes alone do not establish CBM/JSON equivalence. That relation
    is checked during explicit bundle construction and bound by a retained receipt;
    trusting the bundle publisher is necessary. No native code executes here.
    """
    root = Path(root)
    manifest_path = root / 'SDK_MANIFEST.json'
    if root.is_symlink() or manifest_path.is_symlink():
        raise ValueError('symlink SDK root or manifest')
    manifest = load_json(manifest_path)
    if type(manifest) is not dict or manifest.get('format') != 'spectra.tree.sdk.v2':
        raise ValueError('unknown SDK manifest')
    files, models = manifest.get('files'), manifest.get('models')
    if type(files) is not dict or not 1 <= len(files) <= 4096:
        raise ValueError('invalid SDK file inventory')
    if type(models) is not dict or not 1 <= len(models) <= 64:
        raise ValueError('invalid SDK model inventory')
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('symlink in SDK')
        if path.is_file() and path != manifest_path:
            actual.add(path.relative_to(root).as_posix())
    if actual != set(files):
        raise ValueError('missing or extra SDK file; put output outside the SDK')
    for name, record in files.items():
        path = Path(name)
        if (path.is_absolute() or '..' in path.parts or '\\' in name or
                path.as_posix() != name or name == 'SDK_MANIFEST.json'):
            raise ValueError('unsafe SDK path')
        if type(record) is not dict or set(record) != {'bytes', 'sha256'}:
            raise ValueError('invalid file identity record')
        _bounded(record['bytes'], 0, MAX_FILE, 'file size')
        if type(record['sha256']) is not str or not DIGEST.fullmatch(record['sha256']):
            raise ValueError('invalid SHA256')
        full = root / path
        if full.stat().st_size != record['bytes'] or file_sha256(full) != record['sha256']:
            raise ValueError('changed SDK member: ' + name)
    from .packed import metadata
    required_global = {'native/portable/trees.so', 'native/avx2/trees.so',
                       'official/libcatboostmodel-linux-x86_64-1.2.10.so'}
    if not required_global <= set(files):
        raise ValueError('required native dependency absent')
    required_entry = {'rows', 'features', 'maximum', 'classes', 'certified_8', 'certified_16', 'certified_refined'}
    for name, entry in models.items():
        if type(name) is not str or not MODEL_NAME.fullmatch(name):
            raise ValueError('unsafe model name')
        if type(entry) is not dict or set(entry) != required_entry:
            raise ValueError('invalid model metadata inventory')
        _bounded(entry['rows'], 1, 65536, 'replay row count')
        _bounded(entry['features'], 1, 256, 'feature count')
        _bounded(entry['maximum'], 1, 255, 'input maximum')
        if entry['rows'] * entry['features'] > 8_000_000:
            raise ValueError('replay element cap exceeded')
        prefix = f'models/{name}/'
        roles = ['model.json', 'model.cbm', 'model-8.sct', 'model-16.sct',
                 'source_binding.json', 'input.u8', 'indices.i32']
        if not {prefix + role for role in roles} <= set(files):
            raise ValueError('required model asset absent')
        folder = root / prefix
        document = load_json(folder / 'model.json', 32 * 1024**2)
        labels = source_labels(document)
        if type(entry['classes']) is not list or entry['classes'] != labels or any(
                type(a) is not type(b) for a, b in zip(entry['classes'], labels)):
            raise ValueError('manifest class labels disagree with original source')
        binding = load_json(folder / 'source_binding.json')
        expected_binding = {
            'status': 'PASS', 'json_sha256': files[prefix + 'model.json']['sha256'],
            'cbm_sha256': files[prefix + 'model.cbm']['sha256'], 'classes': labels,
        }
        if type(binding) is not dict or any(binding.get(k) != v for k, v in expected_binding.items()):
            raise ValueError('CBM/source binding record mismatch')
        for bits in (8, 16):
            _bounded(entry[f'certified_{bits}'], 0, entry['rows'], 'certified row count')
            compact = metadata((folder / f'model-{bits}.sct').read_bytes())
            expected = {'features': entry['features'], 'maximum': entry['maximum'],
                        'classes': len(labels), 'bits': bits,
                        'source_sha256': files[prefix + 'model.json']['sha256']}
            if any(compact[k] != v for k, v in expected.items()):
                raise ValueError('compact/source geometry or precision mismatch')
        _bounded(entry['certified_refined'], max(entry['certified_8'], entry['certified_16']),
                 min(entry['rows'], entry['certified_8'] + entry['certified_16']), 'refined certified row count')
        if files[prefix + 'input.u8']['bytes'] != entry['rows'] * entry['features']:
            raise ValueError('replay input size mismatch')
        if any(value > entry['maximum'] for value in (folder / 'input.u8').read_bytes()):
            raise ValueError('replay input outside declared domain')
        indices = (folder / 'indices.i32').read_bytes()
        if len(indices) != entry['rows'] * 4:
            raise ValueError('replay output size mismatch')
        if any(not 0 <= v < len(labels) for (v,) in struct.iter_unpack('<i', indices)):
            raise ValueError('replay index outside class mapping')
    return manifest
