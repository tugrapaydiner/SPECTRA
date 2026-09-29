"""Exercise the verification-study auditor on disposable copied JSON receipts.

Model files, measured source and libraries are read-only inputs. This duplicates
only the small process-record directory; no original measurement is modified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

if __package__:
    from .audit_verification import audit
else:
    import importlib.util
    spec = importlib.util.spec_from_file_location('verification_auditor',
        Path(__file__).with_name('audit_verification.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    audit = module.audit


def run(study, source, evidence, library, official, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    copy = out / 'disposable-study'
    shutil.copytree(study, copy)
    arguments = (copy, source, evidence, library, official)
    baseline = audit(*arguments)
    first = json.loads((copy / 'observations.jsonl').read_text().splitlines()[0])
    process_name = f"process-{first['job']:03d}.json"

    def json_edit(path, change):
        value = json.loads(path.read_text())
        change(value)
        return json.dumps(value, allow_nan=False).encode()

    def rows_edit(path, change):
        value = [json.loads(line) for line in path.read_text().splitlines()]
        change(value)
        return ('\n'.join(json.dumps(row, allow_nan=False) for row in value) + '\n').encode()

    def exchange(rows):
        rows[0], rows[1] = rows[1], rows[0]

    cases = [
        ('missing-attempt', 'observations.jsonl', lambda p: rows_edit(p, lambda a: a.pop())),
        ('reordered-attempt', 'observations.jsonl', lambda p: rows_edit(p, exchange)),
        ('boolean-duration', 'observations.jsonl', lambda p: rows_edit(p, lambda a: a[0].update(elapsed_ns=True))),
        ('negative-duration', 'observations.jsonl', lambda p: rows_edit(p, lambda a: a[0].update(elapsed_ns=-1))),
        ('forged-certificate-digest', 'observations.jsonl', lambda p: rows_edit(p, lambda a: a[0].update(oracle_sha256='0'*64))),
        ('changed-source-digest', 'PROTOCOL.json', lambda p: json_edit(p, lambda a: a['sources'].update({'experiments/certified_trees/packed.py': '0'*64}))),
        ('missing-source-role', 'PROTOCOL.json', lambda p: json_edit(p, lambda a: a['sources'].pop('experiments/certified_trees/dyadic.py'))),
        ('missing-model-role', 'PROTOCOL.json', lambda p: json_edit(p, lambda a: a['models']['letter'].pop('model.cbm'))),
        ('changed-native-library', 'PROTOCOL.json', lambda p: json_edit(p, lambda a: a.update(library_sha256='0'*64))),
        ('changed-randomization', 'PROTOCOL.json', lambda p: json_edit(p, lambda a: a.update(seed=7))),
        ('failed-process', process_name, lambda p: json_edit(p, lambda a: a.update(returncode=1))),
        ('wrong-process-output', process_name, lambda p: json_edit(p, lambda a: a.update(stdout='{}'))),
        ('changed-process-command', process_name, lambda p: json_edit(p, lambda a: a['command'].__setitem__(-1, 'skip'))),
        ('inconsistent-process-time', process_name, lambda p: json_edit(p, lambda a: a.update(wall_ns=1))),
        ('changed-cpu-affinity', 'observations.jsonl', lambda p: rows_edit(p, lambda a: a[0].update(affinity=[-1]))),
        ('duplicate-json-key', 'PROTOCOL.json', lambda p: p.read_bytes().replace(b'"seed":', b'"seed":0,"seed":', 1)),
    ]
    observations = []
    for name, relative, mutate in cases:
        path = copy / relative
        original = path.read_bytes()
        try:
            changed = mutate(path)
            if changed == original:
                raise AssertionError('mutation did not change bytes')
            path.write_bytes(changed)
            try:
                audit(*arguments)
            except ValueError as error:
                observations.append({'case': name, 'file': relative, 'rejected': True,
                    'reason': str(error), 'original_sha256': hashlib.sha256(original).hexdigest(),
                    'mutated_sha256': hashlib.sha256(changed).hexdigest()})
            else:
                raise AssertionError('altered evidence accepted: ' + name)
        finally:
            path.write_bytes(original)
    restored = audit(*arguments)
    if baseline != restored:
        raise AssertionError('original evidence was not restored')
    report = {'status': 'PASS', 'rejected': len(observations), 'cases': observations,
        'scope': 'altered copied receipts only; not new model examples or authentication of clocks'}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'source', 'evidence', 'library', 'official', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    print(json.dumps(run(a.study, a.source, a.evidence, a.library, a.official, a.out), indent=2))
