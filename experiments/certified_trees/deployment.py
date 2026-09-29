"""Bounded JSONL deployment of a manifest-checked, source-verified tree bundle.

The default resolves every row, using the original model for unresolved compact
results. This proves neither ground truth nor an arbitrary model/library's safety.
A trusted manifest binds the compact source, fallback CBM, class labels and native
libraries. Hashes prevent accidental mixing, not malicious manifest replacement.
"""
from __future__ import annotations
from array import array
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import BinaryIO, Sequence, Any


def _dump(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, separators=(',', ':'), allow_nan=False) + '\n').encode('ascii')


def _row(raw: bytes, width: int, maximum: int) -> list[int]:
    def reject(value: str) -> None:
        raise ValueError('nonfinite JSON value')
    try:
        value = json.loads(raw.decode('utf-8'), parse_constant=reject)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError('each input line must be a UTF-8 JSON array') from exc
    if type(value) is not list or len(value) != width:
        raise ValueError('input row has the wrong feature count')
    if any(type(x) is not int or not 0 <= x <= maximum for x in value):
        raise ValueError('features must be exact integers in the declared domain')
    return value


def stream_predict(engine: Any, labels: Sequence[int | str], source: BinaryIO,
                   destination: Path, *, identity: dict[str, Any], refine: bool = False,
                   compact_only: bool = False, chunk_rows: int = 128,
                   max_line_bytes: int = 65536, max_rows: int = 1_000_000,
                   max_output_bytes: int = 128 * 1024**2) -> dict[str, Any]:
    """Publish only after clean EOF and complete inference, never overwrite.

    Scratch/output is staged in the existing trusted destination directory. The
    publication uses a same-filesystem hard link. Forced process death may leave
    a private .partial file; this is not power-loss durability or a sandbox.
    """
    for name, value, lo, hi in (
        ('chunk_rows', chunk_rows, 1, 65536),
        ('max_line_bytes', max_line_bytes, 16, 1024**2),
        ('max_rows', max_rows, 1, 10_000_000),
        ('max_output_bytes', max_output_bytes, 256, 2**31),
    ):
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError('invalid ' + name)
    if type(refine) is not bool or type(compact_only) is not bool:
        raise ValueError('boolean execution policy required')
    width, maximum = engine.features, engine.maximum
    if type(width) is not int or not 1 <= width <= 256 or not 1 <= maximum <= 255:
        raise ValueError('invalid engine domain')
    if chunk_rows * width > 8_000_000 or len(labels) != engine.classes:
        raise ValueError('batch or class inventory mismatch')
    if not (all(type(v) is int and -(2**63) <= v < 2**63 for v in labels) or
            all(type(v) is str and len(v.encode('utf-8')) <= 256 for v in labels)):
        raise ValueError('invalid original label mapping')
    if len(set(labels)) != len(labels):
        raise ValueError('duplicate original labels')
    target = Path(destination).absolute()
    if not target.parent.is_dir() or os.path.lexists(target):
        raise FileExistsError('output must be a new path in an existing directory')
    stats = {'rows': 0, 'certified_rows': 0, 'official_rows': 0, 'unresolved_rows': 0}
    input_hash, output_hash = hashlib.sha256(), hashlib.sha256()
    size = 0
    fd, name = tempfile.mkstemp(prefix='.' + target.name + '.', suffix='.partial', dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as output:
            def write(record: dict[str, Any]) -> None:
                nonlocal size
                data = _dump(record)
                if size + len(data) > max_output_bytes:
                    raise ValueError('output byte limit exceeded')
                output.write(data); size += len(data); output_hash.update(data)
            write({'type': 'header', 'format': 'spectra.tree.jsonl.v1', 'identity': identity,
                   'features': width, 'maximum': maximum, 'classes': list(labels),
                   'compact_only': compact_only, 'refine': refine,
                   'claim': 'conditional source-model class agreement, not ground truth'})
            data, count, pending = array('B'), 0, 0
            def flush() -> None:
                nonlocal count, pending, data
                if pending == 0:
                    return
                result = engine.inspect_buffer(data, refine=refine, fallback=not compact_only)
                values, work = result['indices'], result['work']
                if len(values) != pending or any(type(v) is not int or not -1 <= v < len(labels) for v in values):
                    raise ValueError('invalid native result inventory')
                missing = values.count(-1)
                if not compact_only and missing:
                    raise ValueError('full-coverage execution left unresolved rows')
                certified = work['certified_first'] + work['certified_second']
                official = work['official_rows']
                if any(type(v) is not int or v < 0 for v in (certified, official, work['unresolved_rows'])):
                    raise ValueError('invalid native work counters')
                if work['unresolved_rows'] != missing or certified + official + missing != pending:
                    raise ValueError('native coverage counters disagree')
                if compact_only and official:
                    raise ValueError('compact-only execution used an undeclared fallback')
                for value in values:
                    write({'type': 'prediction', 'row': count,
                           'class_index': None if value < 0 else value,
                           'label': None if value < 0 else labels[value],
                           'status': 'UNRESOLVED' if value < 0 else 'RESOLVED'})
                    count += 1
                stats['certified_rows'] += certified; stats['official_rows'] += official
                stats['unresolved_rows'] += missing
                data, pending = array('B'), 0
            while True:
                raw = source.readline(max_line_bytes + 1)
                if raw == b'':
                    break
                if not isinstance(raw, bytes) or len(raw) > max_line_bytes:
                    raise ValueError('input line byte limit exceeded')
                if count + pending >= max_rows:
                    raise ValueError('input row limit exceeded')
                row = _row(raw, width, maximum); input_hash.update(raw)
                data.extend(row); pending += 1
                if pending == chunk_rows:
                    flush()
            flush(); stats['rows'] = count
            complete = {'type': 'complete', **stats, 'input_sha256': input_hash.hexdigest(),
                        'prefix_sha256': output_hash.hexdigest()}
            write(complete); output.flush(); os.fsync(output.fileno())
        # No rename fallback: a concurrent writer must never be overwritten.
        os.link(temporary, target)
        return {**complete, 'output_bytes': size, 'output_sha256': output_hash.hexdigest()}
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--target', choices=('portable', 'avx2'), default='portable')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--compact-only', action='store_true')
    p.add_argument('--refine', action='store_true')
    p.add_argument('--chunk-rows', type=int, default=128)
    a = p.parse_args()
    root = a.sdk.resolve(strict=True)
    if a.output.resolve().is_relative_to(root):
        p.error('place the output outside the immutable SDK directory')
    sys.dont_write_bytecode = True
    from .selftest import verify_manifest
    from .session import VerifiedCompact, TreeSession
    manifest = verify_manifest(root)
    if a.model not in manifest['models']:
        p.error('model not present in the verified bundle')
    if a.target == 'avx2' and 'avx2' not in Path('/proc/cpuinfo').read_text().lower():
        p.error('the AVX2 library requires compatible hardware')
    meta = manifest['models'][a.model]; folder = root / 'models' / a.model
    bits = 8 if a.refine else 16
    first = VerifiedCompact.from_files(folder/'model.json', folder/f'model-{bits}.sct')
    second = VerifiedCompact.from_files(folder/'model.json', folder/'model-16.sct') if a.refine else None
    native = root / 'native' / a.target / 'trees.so'
    official = root / 'official' / 'libcatboostmodel-linux-x86_64-1.2.10.so'
    identity = {'source_sha256': first.info['source_sha256'],
                'compact_sha256': first.info['packed_sha256'],
                'bundle_manifest_sha256': hashlib.sha256((root/'SDK_MANIFEST.json').read_bytes()).hexdigest(),
                'native_sha256': hashlib.sha256(native.read_bytes()).hexdigest()}
    if not a.compact_only:
        identity['official_model_sha256'] = hashlib.sha256((folder/'model.cbm').read_bytes()).hexdigest()
        identity['official_library_sha256'] = hashlib.sha256(official.read_bytes()).hexdigest()
    kwargs = {} if a.compact_only else {'official_model': folder/'model.cbm', 'official_library': official}
    with TreeSession(native, first=first, second=second, **kwargs) as engine, a.input.open('rb') as stream:
        result = stream_predict(engine, meta['classes'], stream, a.output, identity=identity,
                                refine=a.refine, compact_only=a.compact_only, chunk_rows=a.chunk_rows)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
