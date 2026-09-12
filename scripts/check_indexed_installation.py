"""Run existing clean-wheel checks, then exercise the new installed-only APIs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_installation import check


def verify(wheel: Path, out: Path) -> dict:
    original = check(wheel, out)
    out = out.resolve()
    bindir = out/'venv'/('Scripts' if os.name == 'nt' else 'bin')
    python = bindir/('python.exe' if os.name == 'nt' else 'python')
    cwd = out/'outside_checkout'
    env = dict(os.environ, PYTHONNOUSERSITE='1')
    env.pop('PYTHONPATH', None)
    probe = '''import importlib.util,pathlib,sys
from spectra.cnf import CNF,PreparedCNF,solve,solve_indexed
import spectra.cnf.indexed
assert pathlib.Path(spectra.cnf.indexed.__file__).is_relative_to(pathlib.Path(sys.prefix))
assert importlib.util.find_spec('torch') is None
assert importlib.util.find_spec('numpy') is None
assert 'torch' not in sys.modules
p=CNF(3,((1,2),(-1,3),(-2,3),(1,-1),(3,3)))
a=solve(p,seed=77,max_flips=1024)
for b in [solve_indexed(p,seed=77,max_flips=1024),PreparedCNF(p).solve(seed=77,max_flips=1024)]:
 assert {k:v for k,v in a.record().items() if k!='elapsed_ns'}=={k:v for k,v in b.record().items() if k!='elapsed_ns'}
assert p.satisfied(a.witness)
print('installed indexed/prepared paths match the historical implementation')
'''
    commands = [[python, '-I', '-c', probe],
                [python, '-I', '-m', 'spectra', 'cnf', 'solve', 'input.cnf', '--backend', 'indexed', '--out', 'indexed.json'],
                [python, '-I', '-m', 'spectra', 'cnf', 'check', 'input.cnf', 'indexed.json']]
    calls = []
    for command in commands:
        p = subprocess.run([str(a) for a in command], cwd=cwd, env=env,
                           text=True, capture_output=True, timeout=120)
        calls.append(dict(argv=[str(a) for a in command], exit=p.returncode, stdout=p.stdout, stderr=p.stderr))
        (out/'indexed_calls.json').write_text(json.dumps(calls, indent=2)+'\n')
        if p.returncode != 0:
            raise RuntimeError('installed indexed command failed: '+p.stderr)
    result = dict(status='PASS', original_checks=original['checks'], indexed_checks=len(calls),
                  total_checks=original['checks']+len(calls), scope='actual wheel; clean venv; no Torch/NumPy; outside checkout')
    (out/'indexed_report.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.wheel, args.out), indent=2))
