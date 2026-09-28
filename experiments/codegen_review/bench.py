"""Fixed fresh-process generation comparison; exact source hashes are mandatory.

Trusted frozen pickles only. 120-second process cap, 4-GiB address space, one core.
No source writing in generation timer; retain whole-process wall time and RSS.
Stock HAR timeout is a censored failure, never a speedup denominator.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import resource
import signal
import subprocess
import sys
import time

NAMES=('wine-101','wdbc-101','chess-101','penguins-101','titanic-101','zoo-101','har')
VARIANTS=('stock','guard','single')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def worker(args):
    sys.path.insert(0,str(args.package))
    import m2cgen, numpy, sklearn
    from m2cgen.assemblers import get_assembler_cls
    from m2cgen.interpreters import CInterpreter
    raw=args.model.read_bytes()
    if sha(args.model)!=args.digest:raise ValueError('model identity mismatch')
    model=pickle.loads(raw);sys.setrecursionlimit(12000)
    start=time.perf_counter_ns();cpu=time.process_time_ns()
    tree=get_assembler_cls(model)(model).assemble();assembly=time.perf_counter_ns()-start
    text=CInterpreter().interpret(tree)
    elapsed=time.perf_counter_ns()-start;cpu=time.process_time_ns()-cpu
    raw=text.encode('utf-8')
    with (args.out/'model.c').open('xb') as stream:stream.write(raw)
    record={'generation_ns':elapsed,'generation_cpu_ns':cpu,'assembly_ns':assembly,
      'source_sha256':hashlib.sha256(raw).hexdigest(),'source_bytes':len(raw),
      'peak_process_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'm2cgen':m2cgen.__version__,'numpy':numpy.__version__,'sklearn':sklearn.__version__}
    (args.out/'generation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))


def limits():
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})


def run(args):
    args.out.mkdir(parents=True,exist_ok=False)
    models={n:sha(args.models/n/'model.pkl') for n in NAMES}
    packages={v:{p.relative_to(args.packages/(v+'-review')).as_posix():sha(p)
                for p in sorted((args.packages/(v+'-review')/'m2cgen').rglob('*.py'))}
                for v in VARIANTS}
    jobs=[(n,repeat,v) for n in NAMES for repeat in range(1 if n=='har' else 3) for v in VARIANTS]
    random.Random(20260928).shuffle(jobs)
    protocol={'seed':20260928,'jobs':jobs,'models':models,'packages':packages,
       'script_sha256':sha(__file__),'cap_seconds':120,'address_bytes':4*1024**3,
       'cpu':next(l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),
       'scope':'assembly plus interpretation, excludes loading/import/output write; single pinned CPU; no training'}
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONHASHSEED':'0'}
    env.pop('PYTHONPATH',None)
    with (args.out/'attempts.jsonl').open('x') as log:
        for name,repeat,variant in jobs:
            dest=args.out/f'{name}-{repeat}-{variant}';dest.mkdir()
            cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--model',str(args.models/name/'model.pkl'),
                '--digest',models[name],'--package',str(args.packages/(variant+'-review')),'--out',str(dest)]
            start=time.perf_counter();proc=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                                          preexec_fn=limits,start_new_session=True)
            status='COMPLETE'
            try:stdout,stderr=proc.communicate(timeout=120)
            except subprocess.TimeoutExpired:
                status='TIMEOUT';os.killpg(proc.pid,signal.SIGKILL);stdout,stderr=proc.communicate()
            elapsed=time.perf_counter()-start
            (dest/'stdout.txt').write_bytes(stdout);(dest/'stderr.txt').write_bytes(stderr)
            if status=='COMPLETE' and proc.returncode!=0:status='ERROR'
            record={'model':name,'repeat':repeat,'variant':variant,'status':status,
                    'returncode':proc.returncode,'process_seconds':elapsed,'command':cmd}
            if status=='COMPLETE':
                value=json.loads((dest/'generation.json').read_text())
                assert sha(dest/'model.c')==value['source_sha256'];record.update(value)
            log.write(json.dumps(record)+'\n');log.flush()
            print(name,repeat,variant,status,round(elapsed,3),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['run','worker'])
    for name in ('model','package','models','packages','out'):p.add_argument('--'+name,type=Path)
    p.add_argument('--digest');a=p.parse_args()
    (run if a.operation=='run' else worker)(a)
