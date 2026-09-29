"""Independent bounded JSONL outcome audit; no native/learning imports.

Expected source-model indices, class labels, input hash and identity must come
from trusted independent inputs. This checks byte integrity and those expected
outcomes, not the mathematics of certification or the truth of work counters.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


def _object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError('duplicate output key')
        out[key] = value
    return out


def _reject(value):
    raise ValueError('nonfinite output token')


def verify_output(path: Path, *, expected_indices: Sequence[int],
                  labels: Sequence[int | str], input_sha256: str,
                  identity: dict[str, Any], features: int, maximum: int,
                  compact_only: bool, refine: bool) -> dict[str, Any]:
    if type(compact_only) is not bool or type(refine) is not bool:
        raise ValueError('expected execution policy must be explicit')
    if any(type(v) is not int or not 0 <= v < len(labels) for v in expected_indices):
        raise ValueError('expected indices must name source-model classes')
    if type(input_sha256) is not str or len(input_sha256) != 64 or any(c not in '0123456789abcdef' for c in input_sha256):
        raise ValueError('expected input digest must be SHA256')
    prefix, full = hashlib.sha256(), hashlib.sha256()
    count = unresolved = byte_count = 0
    with Path(path).open('rb') as stream:
        def record():
            nonlocal byte_count
            raw = stream.readline(1024*1024 + 1)
            if not raw or len(raw) > 1024*1024 or not raw.endswith(b'\n'):
                raise ValueError('missing, oversized or truncated output record')
            byte_count += len(raw)
            try:
                obj = json.loads(raw, object_pairs_hook=_object,
                                 parse_constant=_reject)
            except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
                raise ValueError('invalid output JSON') from exc
            if type(obj) is not dict:
                raise ValueError('output records must be objects')
            full.update(raw)
            return raw, obj
        raw, header = record()
        if set(header) != {'type','format','identity','features','maximum','classes','compact_only','refine','claim'}:
            raise ValueError('header fields differ')
        if (header['type'] != 'header' or header['format'] != 'spectra.tree.jsonl.v1' or
            type(header['features']) is not int or header['features'] != features or
            type(header['maximum']) is not int or header['maximum'] != maximum or
            type(header['compact_only']) is not bool or header['compact_only'] != compact_only or
            type(header['refine']) is not bool or header['refine'] != refine or
            header['identity'] != identity or type(header['classes']) is not list or
            len(header['classes']) != len(labels) or
            any(type(a) is not type(b) or a != b for a,b in zip(header['classes'],labels))):
            raise ValueError('output model/input policy binding differs')
        prefix.update(raw)
        while True:
            raw, obj = record()
            if obj.get('type') == 'complete':
                break
            if set(obj) != {'type','row','class_index','label','status'}:
                raise ValueError('prediction fields differ')
            if obj['type'] != 'prediction' or type(obj['row']) is not int or obj['row'] != count or count >= len(expected_indices):
                raise ValueError('prediction order or count differs')
            value = obj['class_index']
            if value is None:
                if not compact_only or obj['label'] is not None or obj['status'] != 'UNRESOLVED':
                    raise ValueError('unsupported unresolved outcome')
                unresolved += 1
            elif (type(value) is not int or value != expected_indices[count] or
                  obj['status'] != 'RESOLVED' or type(obj['label']) is not type(labels[value]) or
                  obj['label'] != labels[value]):
                raise ValueError('output disagrees with independently expected source index/label')
            count += 1
            prefix.update(raw)
        if set(obj) != {'type','rows','certified_rows','official_rows','unresolved_rows','input_sha256','prefix_sha256'}:
            raise ValueError('completion fields differ')
        if stream.read(1):
            raise ValueError('trailing bytes after completion')
        for key in ('rows','certified_rows','official_rows','unresolved_rows'):
            if type(obj[key]) is not int or obj[key] < 0:
                raise ValueError('invalid completion counter')
        if (count != len(expected_indices) or obj['rows'] != count or
            obj['unresolved_rows'] != unresolved or
            obj['certified_rows'] + obj['official_rows'] + unresolved != count or
            (compact_only and obj['official_rows'] != 0) or
            obj['input_sha256'] != input_sha256 or obj['prefix_sha256'] != prefix.hexdigest()):
            raise ValueError('incomplete output, wrong input, or coverage mismatch')
    return {'status':'PASS','rows':count,'unresolved_rows':unresolved,
            'certified_rows':obj['certified_rows'],'official_rows':obj['official_rows'],
            'output_bytes':byte_count,'output_sha256':full.hexdigest(),
            'scope':'trusted expected indices, label mapping and file integrity; not a certificate proof'}
