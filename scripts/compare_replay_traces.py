#!/usr/bin/env python3
"""Compare two existing hash-bound CPU diagnostic traces, without model inference."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.replay_trace import compare_probes, load_probe


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--left',type=Path,required=True)
    ap.add_argument('--right',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    report=compare_probes(load_probe(args.left),load_probe(args.right))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'pool':report['pool'],
        'first_difference':report['first_different_observed_module_output']},indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())
