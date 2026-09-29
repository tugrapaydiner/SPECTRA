"""Build every previously frozen original C++ export, without editing its code."""
import argparse,json
from pathlib import Path
from .controls import build
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    for t in ('letter','pendigits','satellite','optdigits'):
        try:lib=build(a.out/t,export=a.models/t/'source.cpp');print(t,'PASS',lib,flush=True)
        except Exception as e:print(t,'FAIL',repr(e),flush=True)
