#!/usr/bin/env python3
"""CPU-only paired native and complete-solve comparisons, including dense FP32."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spectra_reliability.benchmark import run_benchmarks

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--m10-archive',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--native-cache',type=Path,default=Path('outputs/m16_native_cache'))
    a=p.parse_args()
    try:run_benchmarks(a.m10_archive,a.out,a.native_cache)
    except (ValueError,RuntimeError,OSError,KeyError) as exc:
        print(f'Benchmark failed without erasing evidence: {type(exc).__name__}: {exc}',file=sys.stderr);return 2
    print('M16_BENCHMARK_COMPLETE');return 0

if __name__=='__main__':raise SystemExit(main())
