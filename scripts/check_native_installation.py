"""Compile the installed wheel's three JIT extensions, outside the checkout.

This explicitly requires the pinned research environment/compiler. It is separate
from the dependency-free public installation contract and not a speed benchmark.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import venv

PROBE = '''import json, pathlib, sys
import numpy as np
import torch
import deploy
from deploy import torch_kernel, m10_native
from deploy.pack_ternary import pack_ternary_rows
from eval.historical_numerics import extension
assert pathlib.Path(deploy.__file__).is_relative_to(pathlib.Path(sys.prefix))
w=np.array([[1,-1,0,1],[-1,0,1,-1]],dtype=np.int8)
packed,_=pack_ternary_rows(w);wp=torch.from_numpy(packed.copy())
x=torch.tensor([[1,2,3,4],[-2,1,0,3]],dtype=torch.int8)
idx=torch.tensor([0,1],dtype=torch.int32)
got=torch_kernel.sparse_ternary_gemv(x,idx,wp,torch.ones(2,dtype=torch.int32),0,2)
expected=(x.to(torch.int32)@torch.from_numpy(w).to(torch.int32).T).clamp(-128,127).to(torch.int8)
assert torch.equal(got,expected)
linear=m10_native.dense_ternary_linear_fp32(x.float(),wp,torch.ones(2),torch.zeros(2),2)
assert torch.equal(linear,expected.float())
a=torch.tensor([[[1.,2.,3.],[4.,5.,6.]]]);b=torch.tensor([[[1.,0.],[0.,1.],[1.,1.]]])
ordered=extension().ordered_bmm(a,b)
assert torch.equal(ordered,torch.tensor([[[4.,5.],[10.,11.]]]))
print(json.dumps({"status":"PASS","package":deploy.__file__,"torch":str(torch.__version__),
"native_backend":torch_kernel.backend_name(),"dense_operator":m10_native.operator_identity(),
"installed_extensions_executed":3,"arithmetic_checks":3}))
'''


def check(wheel: Path, out: Path) -> dict:
    wheel=wheel.resolve(strict=True);out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    envdir=out/'venv';venv.EnvBuilder(with_pip=True,system_site_packages=False).create(envdir)
    python=envdir/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    # A venv created from another venv does not inherit its installed packages.
    # Hand off only the declared parent research site, explicitly. The probe
    # requires deploy to resolve inside the NEW venv, never the source checkout.
    target_site=Path(subprocess.check_output([str(python),'-I','-c',
        "import sysconfig;print(sysconfig.get_path('purelib'))"],text=True).strip())
    (target_site/'research_dependencies.pth').write_text(sysconfig.get_path('purelib')+'\n')
    env=dict(os.environ);env.pop('PYTHONPATH',None);env['PYTHONNOUSERSITE']='1'
    env.update(TORCH_EXTENSIONS_DIR=str(out/'fresh_native_builds'),MAX_JOBS='1',OMP_NUM_THREADS='1')
    cwd=out/'outside_checkout';cwd.mkdir()
    commands=[[str(python),'-m','pip','install','--no-index','--no-deps','--ignore-installed',str(wheel)],
              [str(python),'-I','-c',PROBE]]
    observations=[]
    for command in commands:
        p=subprocess.run(command,cwd=cwd,env=env,capture_output=True,text=True,timeout=600)
        observations.append(dict(argv=command,exit=p.returncode,stdout=p.stdout,stderr=p.stderr))
        (out/'calls.json').write_text(json.dumps(observations,indent=2)+'\n')
        if p.returncode:raise RuntimeError(p.stdout+'\n'+p.stderr)
    report=json.loads(observations[-1]['stdout'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wheel',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    print(json.dumps(check(a.wheel,a.out),indent=2))
