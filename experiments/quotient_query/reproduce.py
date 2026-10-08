"""One-command fixed-study reproduction with original sources and complete evidence.

A rerun of this now-exposed inventory is reproduction, not new confirmation.
No default backend, fitted model, release or acceptance threshold is changed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.quotient_query import study


def ensure_inputs():
    frozen=json.loads((study.HERE/'FREEZE.json').read_text())
    target=study.HERE/'PARENT_SHA256.json'
    if not target.exists():
        payload=(json.dumps(study.parent_inventory(),sort_keys=True,indent=2)+'\n').encode()
        study.require(hashlib.sha256(payload).hexdigest()==frozen['parent_manifest_sha256'],
                      'reconstructed parent inventory differs')
        with target.open('xb') as f:f.write(payload)
    return study.verify_freeze()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path)
    p.add_argument('--archives',type=Path)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--audit-only',type=Path)
    args=p.parse_args();frozen=ensure_inputs()
    if args.prepare_only:
        print(json.dumps({'source_files':len(frozen['sources']),'freeze':'PASS'}));return
    if args.out is None:p.error('--out required unless --prepare-only')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    from experiments.quotient_query import analyse
    if args.audit_only:
        report=analyse.analyse(args.audit_only.resolve())
    else:
        from experiments.quotient_query.build_all import build_all
        from experiments.quotient_query.run import run
        config=build_all(out/'build',args.archives.resolve() if args.archives else None)
        run(out/'evidence',config,'confirmation')
        report=analyse.analyse(out/'evidence')
    with (out/'summary.json').open('x') as f:
        json.dump(report,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps({'evidence_audited':True,
                      'primary_numerical_gate':report['primary_numerical_gate'],
                      'clean_confirmation_gate':report['clean_confirmation_gate'],
                      'flagship_established':report['flagship_established']}))

if __name__=='__main__':main()
