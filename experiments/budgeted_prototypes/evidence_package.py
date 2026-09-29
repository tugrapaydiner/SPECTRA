"""Seal and audit an evidence tree with explicit source and input-byte identities.

Standard library only. These hashes are integrity checks, not signatures, proof
of clock accuracy, or evidence of independence from prior benchmark exposure.
The manifest and acceptance output must live OUTSIDE the immutable evidence tree.
"""
from __future__ import annotations
import argparse
import hashlib
import types
import json
import math
from pathlib import Path
import stat
import sys

FORMAT = 'spectra.prototype.evidence.v1'
MAX_FILES = 100_000
MAX_JSON = 32 * 1024**2


def strict_json(raw: bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def number(text):
        value = float(text)
        if not math.isfinite(value): raise ValueError('nonfinite JSON number')
        return value
    def reject(text): raise ValueError('nonfinite JSON token')
    if len(raw) > MAX_JSON: raise ValueError('manifest size limit exceeded')
    return json.loads(raw.decode('utf-8'), object_pairs_hook=pairs,
                      parse_float=number, parse_constant=reject)


def file_identity(path: Path) -> dict:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode): raise ValueError('non-regular evidence file')
    h = hashlib.sha256()
    count = 0
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            h.update(block); count += len(block)
    after = path.lstat()
    keys = ('st_dev','st_ino','st_size','st_mtime_ns')
    if count != before.st_size or any(getattr(before,k) != getattr(after,k) for k in keys):
        raise ValueError('evidence changed while reading')
    return {'bytes': count, 'sha256': h.hexdigest()}


def inventory(root: Path) -> dict:
    root = Path(root)
    if root.is_symlink() or not root.is_dir(): raise ValueError('evidence directory unavailable or symlinked')
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink(): raise ValueError('symlink in evidence tree')
        if path.is_dir(): continue
        name = path.relative_to(root).as_posix()
        if '\\' in name or '..' in Path(name).parts: raise ValueError('noncanonical evidence path')
        if len(result) >= MAX_FILES: raise ValueError('evidence file-count cap')
        result[name] = file_identity(path)
    if not result: raise ValueError('empty evidence tree')
    return result


def external_path(root: Path, destination: Path):
    if destination.resolve().is_relative_to(root.resolve()):
        raise ValueError('manifest/receipt must be outside evidence tree')
    if destination.exists() or destination.is_symlink(): raise FileExistsError(str(destination))


def seal(root: Path, manifest: Path) -> dict:
    root, manifest = Path(root), Path(manifest)
    external_path(root, manifest)
    files = inventory(root)
    result = {'format': FORMAT, 'files': files,
              'scope': 'Exact file bytes only; not scientific validity, signatures or authentic clocks.'}
    with manifest.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True); stream.write('\n')
    return {'status': 'SEALED', 'files': len(files),
            'manifest_sha256': file_identity(manifest)['sha256']}


def verify(root: Path, manifest: Path) -> dict:
    root, manifest = Path(root), Path(manifest)
    if manifest.is_symlink() or not manifest.is_file(): raise ValueError('manifest unavailable')
    if manifest.resolve().is_relative_to(root.resolve()): raise ValueError('manifest must be outside evidence tree')
    with manifest.open('rb') as stream: raw = stream.read(MAX_JSON+1)
    doc = strict_json(raw)
    if type(doc) is not dict or doc.get('format') != FORMAT or type(doc.get('files')) is not dict or not doc['files']:
        raise ValueError('invalid or empty evidence manifest')
    for name, entry in doc['files'].items():
        if (type(name) is not str or not name or '\\' in name or Path(name).is_absolute()
                or '..' in Path(name).parts or Path(name).as_posix()!=name):
            raise ValueError('invalid manifest member')
        if (type(entry) is not dict or set(entry)!={'bytes','sha256'}
                or type(entry['bytes']) is not int or entry['bytes']<0
                or type(entry['sha256']) is not str or len(entry['sha256'])!=64
                or any(c not in '0123456789abcdef' for c in entry['sha256'])):
            raise ValueError('invalid manifest identity')
    actual = inventory(root)
    if actual != doc['files']:
        missing = sorted(set(doc['files'])-set(actual))
        extra = sorted(set(actual)-set(doc['files']))
        changed = sorted(k for k in set(actual)&set(doc['files']) if actual[k]!=doc['files'][k])
        raise ValueError(f'evidence inventory mismatch: missing={missing[:3]}, extra={extra[:3]}, changed={changed[:3]}')
    return {'files': len(actual), 'manifest_sha256': hashlib.sha256(raw).hexdigest()}


def load_auditor(path: Path):
    # Execute the exact bytes we identify, not a possibly stale mtime-based .pyc.
    raw = path.read_bytes()
    module = types.ModuleType('prototype_record_audit')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    module._executed_source_sha256 = hashlib.sha256(raw).hexdigest()
    return module


def checked_audit(root: Path, manifest: Path, out: Path) -> dict:
    root, manifest, out = Path(root), Path(manifest), Path(out)
    external_path(root, out)
    path = Path(__file__).with_name('audit.py')
    source_before = {'audit.py': file_identity(path), Path(__file__).name: file_identity(Path(__file__))}
    before = verify(root, manifest)
    module = load_auditor(path)
    if getattr(module, '_executed_source_sha256', None) != source_before['audit.py']['sha256']:
        raise ValueError('loaded auditor is not source-bound to the identified bytes')
    report = module.audit(root)
    if report.get('status')!='PASS': raise ValueError('audit did not pass')
    if report.get('auditor_sha256') != source_before['audit.py']['sha256']:
        raise ValueError('auditor receipt is not bound to the executed source')
    after = verify(root, manifest)
    source_after = {'audit.py': file_identity(path), Path(__file__).name: file_identity(Path(__file__))}
    if before != after or source_before != source_after:
        raise ValueError('auditor or evidence changed during verification')
    result = {**report, 'verification_source': source_before,
              'evidence_manifest_sha256': before['manifest_sha256'],
              'evidence_files': before['files'], 'checked_before_and_after': True}
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('seal','verify','audit'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    try:
        if args.operation=='seal': result=seal(args.root,args.manifest)
        elif args.operation=='verify': result={'status':'PASS',**verify(args.root,args.manifest)}
        else:
            if args.out is None: raise ValueError('--out is required for source-bound acceptance')
            result=checked_audit(args.root,args.manifest,args.out)
    except (ValueError,OSError,KeyError,TypeError) as error:
        print(json.dumps({'status':'NOT_VERIFIED','error':str(error)}),file=sys.stderr)
        return 2
    print(json.dumps(result,indent=2,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
