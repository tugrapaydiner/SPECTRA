"""Execute a real regional compiler from an outside-checkout wheel installation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import check_integrated_installation as harness

PROBE=r'''
import os
os.environ['TORCHINDUCTOR_FREEZING']='1'
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
import pathlib
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(pathlib.Path.cwd()/'fresh-inductor-cache')
import json,sys,time,torch
import spectra,model,deploy
import spectra.compiler_runtime as module
from model.trm import TRM
from spectra.compiler_runtime import CompilerFPSudoku,compiler_counters
from spectra.fp_evidence import correct
root=pathlib.Path(sys.prefix).resolve()
for package in (spectra,model,deploy,module):
 assert pathlib.Path(package.__file__).resolve().is_relative_to(root),package.__file__
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(271211)
m=TRM(dim=16,num_tokens=5,seq_len=16,n_layers=1,n=1,T=1,N_sup=4,heads=2,max_grid_size=8).eval().requires_grad_(False)
solved=[1,2,3,4,3,4,1,2,2,1,4,3,4,3,2,1]
x=torch.tensor([solved],dtype=torch.int64)
records=[]
with torch.inference_mode():
 for mode in ('default','max-autotune'):
  start=time.perf_counter_ns();rt=CompilerFPSudoku(m,mode=mode)
  emb,states=rt.trace(x,4);a,w=rt.solve(x,4)
  assert correct(solved,a.flatten().tolist()) and w['final_semantic'] and w['executed_steps']==1
  assert len(states)==4 and all(bool(torch.isfinite(t).all()) for state in states for t in state)
  records.append({'mode':mode,'identity':rt.identity(),'initialization_and_calls_ns':time.perf_counter_ns()-start})
assert compiler_counters().get('stats',{}).get('unique_graphs',0)>=2
print(json.dumps({'status':'PASS','real_compiler_modes':2,'calls':records,'counters':compiler_counters(),
 'package_path':module.__file__,'scope':'installed-wheel compiler execution fixture, not task-performance evidence'}))
'''


def check(directory:Path,out:Path):
    original=harness.PROBE
    try:
        harness.PROBE=PROBE
        return harness.check(directory,out)
    finally:harness.PROBE=original


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dist',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(check(a.dist,a.out),indent=2))
