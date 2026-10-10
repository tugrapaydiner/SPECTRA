"""Fixed study identities and the full in-memory domain/session boundary."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
BASE='73ad600c3df0b68938c9a322ddf25bdf621f6290'
ARMS=('scc','hybrid','parity','none','domain','minicard','cadical','cadical_simplify','cache_cadical')
CONVENTIONAL=('domain','parity','minicard','cadical','cadical_simplify','cache_cadical')
COUNTS=(1,8,64)
ROUNDS=3
DEADLINE=2.0
ADDRESS_BYTES=4*1024**3
PAYLOAD_BYTES=64*1024**2
WORK=10000000
CONFLICTS=1000000
BOOTSTRAPS=2000
ANALYSIS_SEED=931405


def require(ok,message):
    if not ok:raise ValueError(message)


def sha_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_inventory():
    names=[]
    for p in HERE.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.cpp'):
            names.append(str(p.relative_to(ROOT)))
    names += ['experiments/quotient_query/STUDY_DESIGN.md','spectra/_native/quotient_query.cpp','spectra/cnf/quotient_query.py',
              'spectra/cnf/quotient_certificate.py','tests/public/test_quotient_query.py',
              'tests/public/test_quotient_certificate.py','tests/public/test_quotient_study_tasks.py',
              'tests/public/test_quotient_audit.py']
    return {name:sha_file(ROOT/name) for name in sorted(set(names))}


def parent_inventory():
    """Reconstruct byte identities from the fixed ancestor, not current text."""
    names=subprocess.check_output(['git','ls-tree','-rz','--name-only',BASE],cwd=ROOT).split(b'\0')
    names=[n.decode() for n in names if n]
    # Batch protocol avoids hundreds of subprocesses; binary payload lengths are
    # explicit so old numerical assets cannot desynchronize the parser.
    objects=subprocess.run(['git','cat-file','--batch'],cwd=ROOT,
            input=''.join(BASE+':'+n+'\n' for n in names).encode(),capture_output=True,check=True).stdout
    at=0;result={}
    for name in names:
        end=objects.index(b'\n',at);header=objects[at:end].split();at=end+1
        require(len(header)==3 and header[1]==b'blob','invalid fixed ancestor object')
        length=int(header[2]);payload=objects[at:at+length];at+=length
        require(objects[at:at+1]==b'\n','invalid blob delimiter');at+=1
        result[name]=hashlib.sha256(payload).hexdigest()
    require(at==len(objects),'extra ancestor object bytes')
    return result


def verify_freeze():
    frozen=json.loads((HERE/'FREEZE.json').read_text())
    require(frozen['base_commit']==BASE,'unexpected ancestor')
    require(frozen['sources']==source_inventory(),'frozen source inventory differs')
    parent=json.loads((HERE/'PARENT_SHA256.json').read_text())
    require(sha_file(HERE/'PARENT_SHA256.json')==frozen['parent_manifest_sha256'],'ancestor inventory differs')
    for name,digest in parent.items():
        require(sha_file(ROOT/name)==digest,'parent file modified: '+name)
    require(frozen['arms']==list(ARMS) and frozen['counts']==list(COUNTS) and
            frozen['rounds']==ROUNDS and frozen['deadline_seconds']==DEADLINE,
            'frozen study constants differ')
    return frozen


def schedule(cases,counts=COUNTS,rounds=ROUNDS):
    jobs=[(c['id'],a,q,r) for c in cases for a in ARMS for q in counts for r in range(rounds)]
    random.Random(613721).shuffle(jobs)
    return jobs


def load(config,only=None):
    from spectra.cnf.quotient_query import QuotientRuntime
    from experiments.quotient_query.controls.native import NativeControl,DomainControl
    qr=QuotientRuntime(config['quotient'])
    result={'quotient':qr}
    if only is None or only=='domain':result['domain']=DomainControl(config['domain'])
    if only is None or only=='minicard':result['minicard']=NativeControl('minicard',config['minicard'])
    if only is None or only in ('cadical','cadical_simplify','cache_cadical'):
        result['cadical']=NativeControl('cadical',config['cadical'])
    return result


def prepare(case,arm,controls,count):
    n,k=case['n'],case['k'];edges=tuple(map(tuple,case['edges']));masks=tuple(case['masks'])
    universe=tuple(sorted({v for q in case['queries'][:count] for v,_ in q}));qr=controls['quotient']
    if arm in ('scc','hybrid','parity','none'):
        prepared=qr.prepare(n,k,edges,masks=masks,mode=arm,max_build_bytes=PAYLOAD_BYTES)
    elif arm=='domain':prepared=controls['domain'].prepare(qr,n,k,edges,masks,binary=False)
    else:prepared=controls['minicard' if arm=='minicard' else 'cadical'].prepare(
            qr,n,k,edges,masks,simplify=arm=='cadical_simplify',universe=universe,
            cache_size=16 if arm=='cache_cadical' else 0)
    return prepared


def session(case,arm,count,controls):
    """All original input adaptation, setup, checked queries and release charged.

    Full returned bytes are retained as outputs, not deferred behind lazy handles.
    Their later SHA/compression/transport is evidence I/O outside this timer.
    """
    require(arm in ARMS and count<=len(case['queries']),'bad session request')
    begin=time.perf_counter_ns();cpu=time.process_time_ns();setup_begin=begin
    prepared=prepare(case,arm,controls,count)
    setup_ns=time.perf_counter_ns()-setup_begin
    info=prepared.info if arm in ('scc','hybrid','parity','none') else {}
    rows=[]
    for i in range(count):
        start=time.perf_counter_ns();q=tuple(map(tuple,case['queries'][i]))
        if arm in ('scc','hybrid','parity','none'):
            answer=prepared.solve(q,max_work=WORK,max_state_bytes=PAYLOAD_BYTES)
        elif arm=='domain':answer=prepared.solve(q,max_work=WORK)
        else:answer=prepared.solve(q,conflicts=CONFLICTS)
        elapsed=time.perf_counter_ns()-start
        if isinstance(answer,dict):
            status=answer['status'];labels=answer['labels'];detail=answer['details'];hit=answer['cache_hit']
            internal=answer['elapsed_ns']
        else:
            status=answer.status;labels=answer.labels;hit=False;internal=answer.elapsed_ns
            detail={k:v for k,v in vars(answer).items() if k not in ('status','labels','elapsed_ns')}
        rows.append({'query':i,'status':status,'labels':labels,'elapsed_ns':elapsed,
                     'internal_ns':internal,'details':detail,'cache_hit':hit})
        del answer,q
    start=time.perf_counter_ns();prepared.close();del prepared
    release_ns=time.perf_counter_ns()-start
    elapsed=time.perf_counter_ns()-begin
    cpu_ns=time.process_time_ns()-cpu
    return {'status':'COMPLETE','setup_ns':setup_ns,'release_ns':release_ns,
            'query_ns':sum(r['elapsed_ns'] for r in rows),'session_ns':elapsed,
            'cpu_ns':cpu_ns,'queries':rows,'info':info}


def machine():
    cpu=Path('/proc/cpuinfo').read_text()
    return {'python':sys.version,'platform':platform.platform(),'pid':os.getpid(),
            'cpu_model':next((l.split(':',1)[1].strip() for l in cpu.splitlines() if l.startswith('model name')),'unknown'),
            'cpu_flags':next((l.split(':',1)[1].strip() for l in cpu.splitlines() if l.startswith('flags')),'unknown'),
            'affinity':sorted(os.sched_getaffinity(0)),'thread_envelope':1,
            'address_space_cap':ADDRESS_BYTES,'threads_observed':int(next(l.split(':',1)[1].strip() for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('Threads:'))),
            'numerical_frameworks_loaded':[x for x in ('numpy','torch','scipy','sklearn') if x in sys.modules]}


def rss():
    d={}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:','VmHWM:')):d[line.split(':')[0]+'_KiB']=int(line.split()[1])
    return d
