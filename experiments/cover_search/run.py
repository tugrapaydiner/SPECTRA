"""One-command fresh measurement. No resume, overwrites, selective retries or fitting."""
from __future__ import annotations
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import sysconfig
import time

# Also supports isolated fresh-process probes without importing site hooks.
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.cover_search import study
from spectra.cnf.cover import build_cover_runtime


def add_dependency_paths():
    # Python <=3.13 does not initialize venv prefixes with -S. Do not run site
    # hooks simply to locate an explicitly installed optional dependency.
    version=f'python{sys.version_info.major}.{sys.version_info.minor}'
    for path in (Path(sys.executable).parent.parent/'lib'/version/'site-packages',
                 Path(sysconfig.get_path('purelib'))):
        if path.is_dir() and str(path) not in sys.path:sys.path.append(str(path))


def machine():
    return {'platform':platform.platform(),'python':sys.version,
            'cpu':Path('/proc/cpuinfo').read_text().split('model name\t: ')[1].splitlines()[0],
            'affinity':sorted(os.sched_getaffinity(0)),'pid':os.getpid(),
            'threads':1,'address_space_cap_bytes':4*1024**3}


def rss():
    result={}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:','VmHWM:')):result[line.split(':')[0]+'_KiB']=int(line.split()[1])
    return result


def bounds():
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    allowed=os.sched_getaffinity(0);os.sched_setaffinity(0,{min(allowed)})


def child(connection,case,arm,library,controls):
    try:
        connection.send({'ready':True})
        result=study.attempt(case,arm,library,controls)
        connection.send({'result':result})
    except BaseException as exc:
        connection.send({'error':type(exc).__name__+': '+str(exc)})
    finally:connection.close()


def timed(case,arm,library,controls):
    # Fork only from this single-threaded, dependency-light supervisor.
    context=mp.get_context('fork');parent,other=context.Pipe(duplex=False)
    process=context.Process(target=child,args=(other,case,arm,str(library),controls))
    process.start();other.close()
    if not parent.poll(10):process.kill();process.join();raise RuntimeError('worker did not initialize')
    first=parent.recv();study.require(first=={'ready':True},'invalid worker readiness')
    start=time.perf_counter_ns()
    if parent.poll(study.DEADLINE_SECONDS):
        packet=parent.recv();process.join(5)
        if process.is_alive():process.kill();process.join();raise RuntimeError('worker did not exit')
        if 'error' in packet:raise RuntimeError(packet['error'])
        result=packet['result'];study.require(process.exitcode==0,'worker failed after result')
    else:
        process.kill();process.join()
        result={'status':'TIMEOUT','witness':None,'details':{},'cpu_ns':None,
                'encoding_ns':None,'elapsed_ns':time.perf_counter_ns()-start}
    parent.close()
    result['deadline_overrun']=result['elapsed_ns']>int(study.DEADLINE_SECONDS*1e9)
    return result


def append(stream,row):
    stream.write(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n');stream.flush();os.fsync(stream.fileno())


def run(out,case_limit=None):
    freeze=study.check_freeze();out.mkdir(parents=True,exist_ok=False)
    bounds()
    study.require(not any(k in sys.modules for k in ('numpy','torch','scipy','sklearn')),
                  'use python -I -S to exclude environment startup imports')
    add_dependency_paths()
    library=build_cover_runtime(out/'native')
    controls=study.load_controls()
    import pysat,pysolvers
    (out/'controls.json').write_text(json.dumps({'python_sat':pysat.__version__,
       'extension_sha256':hashlib.sha256(Path(pysolvers.__file__).read_bytes()).hexdigest(),
       'arms':['minicard','gluecard4','glucose42','cadical300','kissat404']},indent=2)+'\n')
    # Import/load before primary timing; actual cold processes are separate below.
    import ctypes
    ctypes.CDLL(str(library))
    cases=study.make_cases()
    if case_limit is not None:cases=cases[:case_limit]
    (out/'cases.json').write_text(json.dumps(cases,sort_keys=True,indent=2)+'\n')
    (out/'freeze.json').write_text(json.dumps(freeze,sort_keys=True,indent=2)+'\n')
    (out/'environment.json').write_text(json.dumps(machine(),sort_keys=True,indent=2)+'\n')
    lookup={c['id']:c for c in cases};jobs=study.schedule(cases)
    (out/'schedule.json').write_text(json.dumps(jobs)+'\n')
    with (out/'timings.jsonl').open('x') as stream:
        for index,(identity,arm,round_) in enumerate(jobs):
            result=timed(lookup[identity],arm,library,controls)
            append(stream,{'job':index,'case':identity,'arm':arm,'round':round_,**result})
            if (index+1)%100==0:print(f'{index+1}/{len(jobs)} complete',flush=True)
    # Five already exposed public tasks and first two tasks per fresh stratum.
    samples=[];per={}
    for case in cases:
        count=per.get(case['stratum'],0);limit=5 if case['kind']=='sudoku' else 2
        if count<limit:samples.append(case);per[case['stratum']]=count+1
    with (out/'resources.jsonl').open('x') as stream:
        for case in samples:
            for arm in study.arms(case):
                start=time.perf_counter_ns()
                command=[sys.executable,'-I','-S',str(Path(__file__).resolve()),'--probe',
                         '--library',str(library),'--arm',arm]
                try:
                    completed=subprocess.run(command,input=json.dumps(case),capture_output=True,text=True,timeout=15)
                except subprocess.TimeoutExpired:
                    append(stream,{'case':case['id'],'arm':arm,'cold_process_wall_ns':time.perf_counter_ns()-start,
                                   'resource_error':'TIMEOUT','result':None})
                    continue
                wall=time.perf_counter_ns()-start
                if completed.returncode:
                    append(stream,{'case':case['id'],'arm':arm,'cold_process_wall_ns':wall,
                                   'resource_error':completed.stderr,'result':None})
                    continue
                payload=json.loads(completed.stdout)
                append(stream,{'case':case['id'],'arm':arm,'cold_process_wall_ns':wall,**payload})
    manifest={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print('measurement complete',flush=True)


def probe(arm,library):
    bounds()
    # Native-only arms do not import PySAT/numerical frameworks in deployment.
    if arm in ('minicard','gluecard4','glucose42','cadical300','kissat404'):
        add_dependency_paths()
        controls=study.load_controls()
    elif arm=='norvig':
        from experiments.structured_search.study import domain_solver
        controls=(None,domain_solver())
    else:controls=(None,None)
    case=json.loads(sys.stdin.read())
    result=study.attempt(case,arm,library,controls)
    print(json.dumps({'result':result,**rss(),'numerical_frameworks_loaded':
         sorted(k for k in ('numpy','torch','scipy','sklearn','pysat') if k in sys.modules)}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path)
    parser.add_argument('--probe',action='store_true');parser.add_argument('--arm')
    parser.add_argument('--library');parser.add_argument('--smoke',type=int)
    args=parser.parse_args()
    if args.probe:probe(args.arm,args.library)
    elif args.out:run(args.out.resolve(),args.smoke)
    else:parser.error('--out required')
if __name__=='__main__':main()
