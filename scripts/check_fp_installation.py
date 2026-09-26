"""Run prepared FP and its eager reference outside the source checkout."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

# Reuse the audited wheel-isolation harness; replace its probe explicitly here.
from check_current_installation import select_wheel
import check_integrated_installation as harness

PROBE=r'''
import json,pathlib,sys,time,torch
import spectra,model,deploy
from model.trm import TRM
from spectra.fp_runtime import PreparedFPSudoku
from deploy.semantic_exit import native_semantic_exit
root=pathlib.Path(sys.prefix).resolve()
for package in (spectra,model,deploy):
 assert pathlib.Path(package.__file__).resolve().is_relative_to(root), package.__file__
assert (pathlib.Path(spectra.__file__).parent/'_native/fp_step.cpp').is_file()
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(17241)
m=TRM(dim=16,num_tokens=5,seq_len=16,n_layers=1,n=1,T=1,N_sup=4,heads=2,max_grid_size=8).eval().requires_grad_(False)
start=time.perf_counter_ns();p=PreparedFPSudoku(m);cold=time.perf_counter_ns()-start
solved=[1,2,3,4,3,4,1,2,2,1,4,3,4,3,2,1]
with torch.inference_mode():
 for values in [solved,[0]*16]:
  x=torch.tensor([values],dtype=torch.int64);a,w=native_semantic_exit(m,x,4);b,v=p.solve(x,4)
  assert torch.equal(a,b) and w==v
  emb,trace=p.trace(x);ref=m.token_embed(x)+m.encode_positions(x,4,4);y,z=torch.zeros_like(ref),torch.zeros_like(ref)
  for Y,Z,L in trace:
   y,z=m.recursive_cycle(ref,y,z);l=m.out_head(y)
   for expected,observed in ((y,Y),(z,Z),(l,L)):
    assert torch.equal(expected.view(torch.int32),observed.view(torch.int32))
print(json.dumps({'status':'PASS','solve_pairs':2,'step_bit_comparisons':24,'fresh_compile_and_prepare_ns':cold,
 'package_path':spectra.__file__,'scope':'installed-wheel contract fixture, not a trained-task benchmark'}))
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
