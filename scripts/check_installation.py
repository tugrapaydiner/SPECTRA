"""Check the actual wheel in a dependency-free venv outside the source tree."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import venv
import zipfile


def check(wheel: Path, out: Path) -> dict:
    wheel = wheel.resolve(strict=True)
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(wheel) as z:
        if z.testzip() is not None:
            raise ValueError('wheel CRC failure')
        natives = sorted(p for p in z.namelist() if p.endswith('.cpp'))
        required = {'deploy/m10_dense_extension.cpp', 'deploy/replay_ordered_extension.cpp',
                    'deploy/cpp_sparse_kernel/extension.cpp', 'deploy/cpp_sparse_kernel/spectra_kernel.cpp'}
        if not required <= set(natives):
            raise ValueError('wheel is missing native sources')
    env_dir = out / 'venv'
    venv.EnvBuilder(with_pip=True, system_site_packages=False).create(env_dir)
    bindir = env_dir / ('Scripts' if os.name == 'nt' else 'bin')
    python = bindir / ('python.exe' if os.name == 'nt' else 'python')
    command = bindir / ('spectra.exe' if os.name == 'nt' else 'spectra')
    # No source tree or user site can satisfy the installed import by accident.
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env['PYTHONNOUSERSITE'] = '1'
    cwd = out / 'outside_checkout'; cwd.mkdir()
    calls = []
    def run(args, expected=0):
        p = subprocess.run([str(a) for a in args], cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
        calls.append(dict(argv=[str(a) for a in args], exit=p.returncode, stdout=p.stdout, stderr=p.stderr))
        (out / 'calls.json').write_text(json.dumps(calls, indent=2)+'\n')
        if p.returncode != expected:
            raise RuntimeError(f'installed command failed: {args}\n{p.stdout}\n{p.stderr}')
        return p.stdout
    run([python, '-m', 'pip', 'install', '--no-index', '--no-deps', wheel])
    probe = run([python, '-I', '-c', '''import importlib.util,json,pathlib,sys
import spectra,deploy
from spectra.cnf import CNF,solve,CompactCNFRepairState
assert importlib.util.find_spec("torch") is None
assert importlib.util.find_spec("numpy") is None
assert "torch" not in sys.modules and "numpy" not in sys.modules
assert pathlib.Path(spectra.__file__).is_relative_to(pathlib.Path(sys.prefix))
assert (pathlib.Path(deploy.__file__).parent/"replay_ordered_extension.cpp").is_file()
r=solve(CNF(1,((1,),)),seed=7,max_flips=128)
assert r.status=="SAT_VERIFIED" and r.witness==(True,)
print(json.dumps({"package":spectra.__file__,"version":spectra.__version__,"optional_imports":False}))
'''])
    run([command, '--version'])
    doctor = json.loads(run([command, 'doctor']))
    assert doctor['torch_installed'] is False
    formula = cwd/'input.cnf';formula.write_text('p cnf 1 1\n1 0\n')
    run([command, 'cnf', 'solve', formula, '--seed', '7', '--max-flips', '128', '--out', 'answer.json'])
    answer = json.loads((cwd/'answer.json').read_text());assert answer['status']=='SAT_VERIFIED'
    assert json.loads(run([python, '-I', '-m', 'spectra', 'cnf', 'check', formula, 'answer.json']))['valid']
    run([command, 'cnf', 'solve', formula, '--out', 'answer.json'], expected=2)
    (cwd/'false.json').write_text('{"witness":[false]}')
    run([command, 'cnf', 'check', formula, 'false.json'], expected=1)
    report = dict(wheel=wheel.name, sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
                  native_sources=natives, checks=len(calls), installed_probe=json.loads(probe),
                  status='PASS', mode='fresh venv, no dependencies, outside checkout')
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wheel',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(check(a.wheel,a.out),indent=2))
