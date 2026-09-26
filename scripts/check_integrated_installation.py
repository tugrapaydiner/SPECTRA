"""Compile and execute the integrated runtime from an actual installed wheel.

The isolated venv explicitly adds the parent research site via a .pth file;
every SPECTRA module must resolve within the new installation. Compilation
uses a fresh extension directory, never source-tree or prebuilt local artifacts.
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

from check_current_installation import select_wheel

PROBE = r'''
import json,pathlib,sys,torch
import spectra,deploy,model
from deploy.m10_artifact import export_cpu_artifact,load_cpu_artifact
from deploy.m10_runtime import CPURecursiveRuntime
from deploy.validated_runtime import ValidatedCPURecursiveRuntime
from model.trm import TRM
from spectra.blocked_runtime import BlockedCPURecursiveRuntime
from spectra.inference import predict_final
root=pathlib.Path(sys.prefix).resolve()
for package in (spectra,deploy,model):
 assert pathlib.Path(package.__file__).resolve().is_relative_to(root), package.__file__
assert (pathlib.Path(spectra.__file__).parent/'_native/blocked_linear.cpp').is_file()
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(71101)
m=TRM(dim=16,num_tokens=5,seq_len=16,n_layers=1,n=1,T=1,N_sup=4,heads=4,max_grid_size=8,ternary=True,act8=True).eval()
p=pathlib.Path('installed-fixture.pt')
export_cpu_artifact(m,p,height=4,width=4,box=2,source_checkpoint_sha256='installed-contract',source_checkpoint_tensor_sha256='installed-contract',training_seed=71101,training_step=0,data_provenance={'contract_only':True},export_git_sha='installed-wheel')
a=load_cpu_artifact(p);x=torch.randint(0,5,(3,16));expected=None;arms=0
for cls in [CPURecursiveRuntime,ValidatedCPURecursiveRuntime,BlockedCPURecursiveRuntime]:
 r=cls(a)
 for mode in ['trace','final']:
  v=r.forward(x) if mode=='trace' else predict_final(r,x)
  h=v.step_outputs[-1]['halt_logit'] if mode=='trace' else v.halt_logit
  bits=(v.logits.contiguous().numpy().tobytes(),v.answer.contiguous().numpy().tobytes(),h.contiguous().numpy().tobytes())
  if expected is not None: assert bits==expected
  expected=bits;arms+=1
  assert v.work['native_call_count_matches_architecture']
print(json.dumps({'status':'PASS','runtime_arms':arms,'bitwise_equivalent':True,'package_path':spectra.__file__,'scope':'actual wheel; isolated imports; shared declared research dependencies; fresh native compilation'}))
'''


def check(directory: Path, out: Path):
    wheel = select_wheel(directory)
    out = out.resolve(); out.mkdir(parents=True, exist_ok=False)
    venv.EnvBuilder(with_pip=True, system_site_packages=False).create(out / "venv")
    python = out / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    # Nested venvs do not inherit the research packages installed in their parent.
    # Explicitly expose that declared dependency site after the new venv's site;
    # the probe still rejects any SPECTRA package resolved outside the new venv.
    target_site = Path(subprocess.check_output(
        [str(python), "-I", "-c", "import sysconfig;print(sysconfig.get_path('purelib'))"],
        text=True).strip())
    (target_site / "research_dependencies.pth").write_text(sysconfig.get_path("purelib") + "\n")
    cwd = out / "outside-checkout"; cwd.mkdir()
    env = dict(os.environ, PYTHONNOUSERSITE="1", MAX_JOBS="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               TORCH_EXTENSIONS_DIR=str(out / "fresh-native-builds"))
    env.pop("PYTHONPATH", None)
    calls = []
    for command in [[str(python), "-m", "pip", "install", "--no-index", "--no-deps", "--ignore-installed", str(wheel)],
                    [str(python), "-I", "-c", PROBE]]:
        result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=300)
        calls.append(dict(argv=command, exit=result.returncode, stdout=result.stdout, stderr=result.stderr))
        (out / "calls.json").write_text(json.dumps(calls, indent=2) + "\n")
        if result.returncode: raise RuntimeError(result.stdout + "\n" + result.stderr)
    report = json.loads(calls[-1]["stdout"])
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.dist, args.out), indent=2))
