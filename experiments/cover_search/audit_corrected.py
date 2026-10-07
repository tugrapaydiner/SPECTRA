"""Explicit post-run JSON transport corrigendum, not the original frozen audit.

The original analyser compared tuple edge pairs with JSON lists and refused the
complete inventory. This wrapper proves canonical equality, temporarily presents
JSON-native expected data to the UNCHANGED analyser, then restores its function.
No solver, input, observation, threshold or statistical calculation is modified.
A single-process audit utility; not a thread-safe library configuration mechanism.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.cover_search import analyse, study


def json_native(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


@contextmanager
def expected_json_inputs():
    original = study.make_cases
    try:
        study.make_cases = lambda: json_native(original())
        yield
    finally:
        study.make_cases = original


def validate(folder: Path):
    """Validate the same full receipts after the one documented transport repair."""
    expected = study.make_cases()
    stored = json.loads((folder / 'cases.json').read_text())
    study.require(json_native(expected) == stored, 'canonical input inventory differs')
    study.require(study.digest(expected) == study.digest(stored), 'canonical input digest differs')
    with expected_json_inputs():
        return analyse.validate(folder)


def audit(folder: Path):
    validate(folder)
    with expected_json_inputs():
        result = analyse.analyse(folder)
    result['audit_correction'] = {
        'original_failure': 'tuple/list comparison after JSON edge serialization',
        'change': 'JSON-normalize reconstructed expected inputs only',
        'original_source_and_observations_changed': False,
        'correction_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'protocol_status': 'Original frozen audit failed; corrected descriptive evidence, not clean confirmation.',
        'gate_interpretation': 'Booleans are recalculated numerical criteria, not automatic promotion.'}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    report = audit(args.folder.resolve())
    with args.out.open('x') as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write('\n')

if __name__ == '__main__':
    main()
