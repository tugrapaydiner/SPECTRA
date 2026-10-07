"""Reconstruct pinned base inventory, run unchanged study, apply explicit corrigendum.

This replays an already consumed fixed study, never creates new confirmation.
A checkout needs the pinned base commit (git fetch --unshallow where applicable).
The downloadable source archive already includes the exact baseline inventory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.cover_search import study


def ensure_baseline():
    target = study.HERE / 'BASELINE_SHA256.json'
    expected = json.loads((study.HERE/'FREEZE.json').read_text())['sources'][str(target.relative_to(ROOT))]
    if not target.exists():
        names = subprocess.run(['git','ls-tree','-rz','--name-only',study.BASE_COMMIT],
            cwd=ROOT, check=True, capture_output=True).stdout.split(b'\0')
        manifest = {}
        for raw in names:
            if not raw:
                continue
            name = raw.decode('utf-8')
            content = subprocess.run(['git','show',study.BASE_COMMIT+':'+name],
                cwd=ROOT, check=True, capture_output=True).stdout
            manifest[name] = hashlib.sha256(content).hexdigest()
        # The original frozen inventory has no final newline. Preserve its exact
        # bytes rather than replacing the trust anchor to accommodate formatting.
        payload = json.dumps(manifest, sort_keys=True, indent=2).encode()
        study.require(hashlib.sha256(payload).hexdigest()==expected, 'reconstructed base inventory differs')
        with target.open('xb') as stream:
            stream.write(payload)
    study.require(hashlib.sha256(target.read_bytes()).hexdigest()==expected,'incorrect existing baseline inventory')
    return study.check_freeze()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path)
    p.add_argument('--prepare-only', action='store_true')
    args=p.parse_args()
    ensure_baseline()
    if args.prepare_only:
        print('Frozen sources and all 764 baseline files match.')
        return
    if not args.out:
        p.error('--out required unless --prepare-only')
    # Import only after the inventory is present. The runner checks isolation.
    from experiments.cover_search import run, audit_corrected
    output=args.out.resolve()
    run.run(output)
    report=audit_corrected.audit(output)
    # Derived summary is outside the immutable raw evidence directory.
    with output.with_name(output.name+'-summary.json').open('x') as stream:
        json.dump(report,stream,indent=2,sort_keys=True)
        stream.write('\n')
    print('Complete fixed-workload reproduction; see explicit audit correction and failed gates.')

if __name__ == '__main__':
    main()
