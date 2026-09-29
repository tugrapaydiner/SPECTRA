"""Refit ALL six frozen arms after both complete training-only searches."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.integer_interactions import study
p=argparse.ArgumentParser()
for k in ("data","selection","secondary","out","combined"):p.add_argument("--"+k,type=Path,required=True)
a=p.parse_args();a.combined.mkdir(parents=True,exist_ok=False)
primary=json.loads((a.selection/"selected.json").read_text());secondary=json.loads((a.secondary/"selected.json").read_text())
for task in primary:
    if set(primary[task])!={"uniform","diagonal","full","whitening"} or set(secondary[task])!={"local_supervised","local_unsupervised"}:raise ValueError("selection inventory mismatch")
    primary[task].update(secondary[task])
study.write(a.combined/"selected.json",primary)
study.write(a.combined/"IDENTITY.json",{"primary":study.sha(a.selection/"selected.json"),"secondary":study.sha(a.secondary/"selected.json"),"all_six_arms_retained":True})
a.selection=a.combined
study.ARMS=("uniform","diagonal","full","whitening","local_supervised","local_unsupervised")
study.refit(a)
