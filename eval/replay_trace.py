"""Compare hash-bound CPU module traces without altering replay acceptance.

The first differing *observed module output* is not necessarily the first
primitive operation to diverge. This utility is diagnostic, never a replacement
for the full checkpoint/pool replay or a way to waive its required hashes.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

DTYPES = {'float32':'torch.float32', 'float64':'torch.float64',
          'int64':'torch.int64', 'int32':'torch.int32', 'uint8':'torch.uint8', 'bool':'torch.bool'}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def tensor_digest(arrays: dict[str, np.ndarray]) -> str:
    """Independent NumPy implementation of the retained tensor identity format."""
    h = hashlib.sha256()
    for key, a in sorted(arrays.items()):
        if a.dtype.name not in DTYPES or not a.dtype.isnative or sys.byteorder != 'little':
            raise ValueError('unsupported trace dtype or byte order')
        h.update((key+':'+DTYPES[a.dtype.name]+':'+str(tuple(a.shape))+'\n').encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


@dataclass(frozen=True)
class Probe:
    report: dict
    arrays: dict[str, np.ndarray]
    report_sha256: str
    container_sha256: str


def load_probe(directory: str | Path) -> Probe:
    directory = Path(directory)
    report_raw = (directory/'diagnostic.json').read_bytes()
    report = json.loads(report_raw, object_pairs_hook=_unique_object)
    path = directory/'first_cycle.npz'
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(not n.endswith('.npy') for n in names):
            raise ValueError('duplicate or non-array trace members')
        if archive.testzip() is not None:
            raise ValueError('trace container CRC failure')
    with np.load(path, allow_pickle=False) as stored:
        arrays = {k: stored[k].copy() for k in stored.files}
    if (report['status'] != 'DIAGNOSTIC_COMPLETE_NOT_ACCEPTANCE'
        or report['trace_no_hook_equal'] is not True or report['independent_production_equal'] is not True
        or type(report['historical_hash_match']) is not bool
        or report['historical_hash_match'] != (report['observed_pool_hash'] == report['expected_pool_hash'])):
        raise ValueError('invalid diagnostic status or historical-match declaration')
    if not {'input','embedded','y','z'} <= arrays.keys() or set(arrays) != set(report['fields']):
        raise ValueError('missing required trace fields or mismatched field inventory')
    if not arrays['input'].dtype == np.int64:
        raise ValueError('trace inputs must be int64')
    for key, a in arrays.items():
        field = report['fields'][key]
        if not np.isfinite(a).all() or list(a.shape) != field['shape']:
            raise ValueError('nonfinite values or invalid trace geometry')
        if tensor_digest({key:a}) != field['sha256']:
            raise ValueError('individual trace field hash mismatch')
        a.setflags(write=False)
    if tensor_digest(arrays) != report['trace_sha256']:
        raise ValueError('combined trace hash mismatch')
    source = report['source']
    if sha256(json.dumps(source['files'], sort_keys=True).encode()) != source['source_sha256']:
        raise ValueError('source inventory hash mismatch')
    return Probe(report, arrays, sha256(report_raw), sha256(path.read_bytes()))


def _cpu_model(report: dict) -> str:
    return next((l.split(':',1)[1].strip() for l in report['cpuinfo'].splitlines()
                 if l.startswith('model name')), 'unknown')


def compare_probes(left: Probe, right: Probe) -> dict:
    a, b = left.report, right.report
    for name in ('pool','manifest_sha256','expected_pool_hash','torch','numpy'):
        if a[name] != b[name]:
            raise ValueError('incomparable probe identity: '+name)
    if a['source']['files'] != b['source']['files'] or a['source']['source_sha256'] != b['source']['source_sha256']:
        raise ValueError('cannot attribute a difference across changed executable sources')
    # Archive field order is hook execution order; sorted JSON keys are not.
    if list(left.arrays) != list(right.arrays):
        raise ValueError('trace execution order differs')
    for key in left.arrays:
        x, y = left.arrays[key], right.arrays[key]
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError('trace tensor geometry or dtype differs: '+key)
    if not np.array_equal(left.arrays['input'], right.arrays['input']):
        raise ValueError('probe inputs differ')
    fields = []
    for key, x in left.arrays.items():
        y = right.arrays[key]
        # Byte equality is the replay contract. Record numerical equality too:
        # signed-zero differences can change a hash without changing x != y.
        byte_equal = x.tobytes() == y.tobytes()
        mismatch = x != y
        difference = np.abs(x.astype(np.float64)-y.astype(np.float64))
        fields.append({'name':key,'shape':list(x.shape),'byte_equal':byte_equal,
            'element_count':int(x.size),'numerically_different_elements':int(mismatch.sum()),
            'max_abs_difference':float(difference.max(initial=0)),
            'left_sha256':a['fields'][key]['sha256'],'right_sha256':b['fields'][key]['sha256']})
    outputs = [f for f in fields if f['name'] not in ('input','embedded','y','z')]
    first = next((f for f in outputs if not f['byte_equal']),None)
    def identity(probe):
        r = probe.report
        return {'cpu_model':_cpu_model(r),'python':r['python'],
            'dispatch_environment':r['dispatch_environment'],
            'diagnostic_sha256':probe.report_sha256,'npz_sha256':probe.container_sha256,
            'trace_sha256':r['trace_sha256'],'pool_sha256':r['observed_pool_hash'],
            'historical_hash_match':r['historical_hash_match']}
    return {'schema':'spectra.cpu_trace_comparison.v1',
        'status':'DIAGNOSTIC_ONLY_NOT_REPLAY_ACCEPTANCE','pool':a['pool'],
        'source_sha256':a['source']['source_sha256'],'manifest_sha256':a['manifest_sha256'],
        'expected_pool_sha256':a['expected_pool_hash'],
        'inputs_equal':True,'embedded_bytes_equal':left.arrays['embedded'].tobytes()==right.arrays['embedded'].tobytes(),
        'declared_dispatch_equal':a['dispatch_environment']==b['dispatch_environment'],
        'torch_build_text_equal':a['torch_build']==b['torch_build'],
        'first_different_observed_module_output':first,'fields':fields,
        'left':identity(left),'right':identity(right),
        'claim_boundary':'First difference among recorded module outputs, not proof of a primitive-kernel cause. '
            'Equal environment strings do not establish equal dispatched arithmetic. '
            'Full historical replay hashes and numerical tolerances remain unchanged.'}
