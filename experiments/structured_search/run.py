"""Run the frozen public structured-task comparison; never overwrite observations."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.structured_search.study import (
    ARMS, ROUNDS, append, call, cases, check_freeze, native_and_domain, require,
)


def worker(arm, trace):
    case=json.load(sys.stdin)
    native,norvig=native_and_domain()
    gc.collect()
    before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if trace:
        tracemalloc.start()
    answer=call(case,arm,native,norvig)
    peak=tracemalloc.get_traced_memory()[1] if trace else None
    if trace:
        tracemalloc.stop()
    print(json.dumps({'id':case['id'], 'arm':arm, 'status':answer['result']['status'],
                     'python_peak_bytes':peak, 'baseline_rss_kib':before,
                     'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))


def run_memory(out, inputs):
    relay='import subprocess,sys; raise SystemExit(subprocess.call(sys.argv[1:]))'
    with (out/'memory.jsonl').open('x') as stream:
        for case in inputs[:5]:
            for arm in ARMS:
                command=[sys.executable,'-I','-S','-c',relay,sys.executable,'-I',
                         str(Path(__file__).resolve()),'worker','--arm',arm]
                result=subprocess.run(command,input=json.dumps(case),capture_output=True,
                                      text=True,timeout=60,check=True)
                row=json.loads(result.stdout)
                if arm!='glucose4':
                    traced=subprocess.run(command+['--trace'],input=json.dumps(case),capture_output=True,
                                          text=True,timeout=60,check=True)
                    traced_row=json.loads(traced.stdout)
                    require(traced_row['status']==row['status'],'memory outcome differs')
                    row['python_peak_bytes']=traced_row['python_peak_bytes']
                append(stream,row)


def run(out, freeze_path):
    freeze=json.loads(freeze_path.read_text())
    check_freeze(freeze)
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
            ==freeze['commit'],'run from published frozen commit')
    native,norvig=native_and_domain()
    import pysat, pysat.solvers, pysolvers
    require(sys.platform=='linux','Linux affinity and RSS units required')
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    out.mkdir(parents=True,exist_ok=False)
    lock={'freeze':freeze,'python':sys.version,'platform':platform.platform(),
          'python_sat':pysat.__version__,'affinity':sorted(os.sched_getaffinity(0)),
          'native_sources':{Path(m.__file__).name:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                            for m in (pysat,pysat.solvers,pysolvers)},
          'cpu':next(s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
                     if s.startswith('model name')),'status':'RUN_STARTED'}
    (out/'LOCK.json').write_text(json.dumps(lock,indent=2)+'\n')
    try:
        inputs=cases('evaluation')
        with (out/'cases.jsonl').open('x') as stream:
            for case in inputs:
                append(stream,case)
        with (out/'rows.jsonl').open('x') as stream:
            for round_id in range(ROUNDS):
                for i,case in enumerate(inputs):
                    offset=(i+round_id)%len(ARMS)
                    order=ARMS[offset:]+ARMS[:offset]
                    if (i+round_id)%2:
                        order=order[::-1]
                    for arm in order:
                        cpu_start,wall_start=time.process_time_ns(),time.perf_counter_ns()
                        result=call(case,arm,native,norvig)
                        wall_ns,cpu_ns=time.perf_counter_ns()-wall_start,time.process_time_ns()-cpu_start
                        append(stream,{'id':case['id'],'sha256':case['sha256'],'arm':arm,
                                       'round':round_id,'wall_ns':wall_ns,'cpu_ns':cpu_ns,**result})
                print(f'timing round {round_id+1}/{ROUNDS} complete',flush=True)
        run_memory(out,inputs)
        manifest={'status':'COMPLETE','files':{name:{'bytes':(out/name).stat().st_size,
                    'sha256':hashlib.sha256((out/name).read_bytes()).hexdigest()}
                    for name in ('LOCK.json','cases.jsonl','rows.jsonl','memory.jsonl')}}
        (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('timing and memory inventories complete',flush=True)
    except BaseException as error:
        (out/'FAILURE.json').write_text(json.dumps({'type':type(error).__name__,'message':str(error)},indent=2)+'\n')
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    runner=sub.add_parser('run')
    runner.add_argument('--out',type=Path,required=True)
    runner.add_argument('--freeze',type=Path,required=True)
    child=sub.add_parser('worker')
    child.add_argument('--arm',choices=ARMS,required=True)
    child.add_argument('--trace',action='store_true')
    args=parser.parse_args()
    if args.command=='worker':
        worker(args.arm,args.trace)
    else:
        run(args.out.resolve(),args.freeze.resolve())


if __name__=='__main__':
    main()
