"""Verify the non-destructive workspace migration against the retained Git base."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BASE='181e236528b5b58263c98b09925f0d7612c919d3'
RELOCATIONS=ROOT/'maintenance/relocations.json'
RELOCATION_UPDATES=ROOT/'maintenance/relocation-updates.json'
HEX64=re.compile(r'^[0-9a-f]{64}$')


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args])


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_repo_path(value):
    if type(value) is not str or not value:
        raise ValueError('invalid update path')
    path=PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or value!=path.as_posix():
        raise ValueError('unsafe update path: '+value)
    return value


def load_updates(base, entries, protected_paths):
    if not RELOCATION_UPDATES.is_file():
        return {}
    payload=json.loads(RELOCATION_UPDATES.read_text())
    if set(payload)!=set(('schema','base','base_manifest','updates')):
        raise ValueError('unexpected relocation-update fields')
    if payload['schema']!='spectra.relocation_updates.v1':
        raise ValueError('unsupported relocation-update schema')
    if payload['base']!=base or payload['base_manifest']!='maintenance/relocations.json':
        raise ValueError('relocation-update base differs')
    if type(payload['updates']) is not list:
        raise ValueError('relocation updates must be a list')
    relocated={e['to']:e for e in entries if e['disposition']=='relocated'}
    updates={}
    for item in payload['updates']:
        if type(item) is not dict or set(item)!=set(('path','previous_sha256','current_sha256','reason')):
            raise ValueError('invalid relocation-update entry')
        path=safe_repo_path(item['path'])
        if path in updates:
            raise ValueError('duplicate relocation-update path: '+path)
        previous=item['previous_sha256'];current=item['current_sha256']
        if type(previous) is not str or not HEX64.fullmatch(previous):
            raise ValueError('invalid previous update hash: '+path)
        if type(current) is not str or not HEX64.fullmatch(current):
            raise ValueError('invalid current update hash: '+path)
        if previous==current:
            raise ValueError('non-changing relocation update: '+path)
        if type(item['reason']) is not str or not item['reason'].strip():
            raise ValueError('missing relocation-update reason: '+path)
        if path in relocated:
            expected_previous=relocated[path]['current_sha256']
        elif path in protected_paths:
            expected_previous=digest(git('show',base+':'+path))
        else:
            raise ValueError('relocation update is outside the audited set: '+path)
        if previous!=expected_previous:
            raise ValueError('relocation-update previous hash mismatch: '+path)
        target=ROOT/path
        if not target.is_file() or digest(target.read_bytes())!=current:
            raise ValueError('relocation-update current hash mismatch: '+path)
        updates[path]=item
    return updates


def audit():
    manifest=json.loads(RELOCATIONS.read_text())
    base=manifest['base'];entries=manifest['moves']
    if base!=BASE:raise ValueError('wrong cleanup base')
    if len({e['from'] for e in entries})!=len(entries):raise ValueError('duplicate migration entry')
    protected_paths={
        path for path in git('ls-tree','-r','--name-only',base).decode().splitlines()
        if path.startswith(('common/','data/','deploy/','eval/','model/','train/','results/','tests/','scripts/','config/'))
    }
    updates=load_updates(base,entries,protected_paths)
    for e in entries:
        data=git('show',base+':'+e['from'])
        if digest(data)!=e['original_sha256']:raise ValueError('original hash mismatch')
        if git('rev-parse',base+':'+e['from']).decode().strip()!=e['original_blob']:raise ValueError('original blob mismatch')
        if e['disposition']=='git_history':
            if e['retained_commit']!=base:raise ValueError('unretained original')
            if e['from']!='README.md' and (ROOT/e['from']).exists():raise ValueError('retired file remains active')
        elif e['disposition']=='relocated':
            expected=updates.get(e['to'],{}).get('current_sha256',e['current_sha256'])
            if digest((ROOT/e['to']).read_bytes())!=expected:raise ValueError('relocated hash mismatch')
        else:raise ValueError('unknown disposition')
    # Historical executable sources, tests, protocols and raw evidence remain
    # byte-identical unless an explicit update receipt binds both exact versions.
    protected=[];updated_protected=[]
    for path in sorted(protected_paths):
        current=(ROOT/path).read_bytes();original=git('show',base+':'+path)
        if path in updates:
            if digest(current)!=updates[path]['current_sha256']:
                raise ValueError('authorized historical update differs: '+path)
            updated_protected.append(path)
        elif current!=original:
            raise ValueError('historical content changed: '+path)
        protected.append(path)
    # Every update must have been consumed by either a relocated or protected path.
    consumed={e['to'] for e in entries if e['disposition']=='relocated'}|set(updated_protected)
    if set(updates)-consumed:
        raise ValueError('unused relocation update')
    # The imported branch helper is recovered exactly, not silently rewritten.
    recovered=git('show','095c1ed65ffa3beabc55e42052c2f3b038a3150e:common/evidence_freeze.py')
    if (ROOT/'spectra/evidence.py').read_bytes()!=recovered:raise ValueError('recovered helper changed')
    links=0
    for name in ['README.md','docs/README.md','docs/STATUS.md','docs/DEVELOPMENT.md','docs/history/README.md']:
        path=ROOT/name
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)',path.read_text()):
            if re.match(r'^[a-z]+:',link) or link.startswith('#'):continue
            target=link.split('#')[0]
            if not (path.parent/target).exists():raise ValueError('broken current guide link: '+name+':'+link)
            links+=1
    return dict(status='PASS',base=base,retired_files=sum(e['disposition']=='git_history' for e in entries),
                relocated_workflows=sum(e['disposition']=='relocated' for e in entries),
                historical_files_verified=len(protected),
                historical_files_byte_identical=len(protected)-len(updated_protected),
                authorized_current_updates=sorted(updates),
                recovered_helper_byte_identical=True,current_local_guide_links=links)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=audit()
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
