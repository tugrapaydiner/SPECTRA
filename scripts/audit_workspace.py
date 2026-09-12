"""Verify the non-destructive workspace migration against the retained Git base."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args])


def audit():
    manifest=json.loads((ROOT/'maintenance/relocations.json').read_text())
    base=manifest['base'];entries=manifest['moves']
    if base!='181e236528b5b58263c98b09925f0d7612c919d3':raise ValueError('wrong cleanup base')
    if len({e['from'] for e in entries})!=len(entries):raise ValueError('duplicate migration entry')
    for e in entries:
        data=git('show',base+':'+e['from'])
        if hashlib.sha256(data).hexdigest()!=e['original_sha256']:raise ValueError('original hash mismatch')
        if git('rev-parse',base+':'+e['from']).decode().strip()!=e['original_blob']:raise ValueError('original blob mismatch')
        if e['disposition']=='git_history':
            if e['retained_commit']!=base:raise ValueError('unretained original')
            if e['from']!='README.md' and (ROOT/e['from']).exists():raise ValueError('retired file remains active')
        elif e['disposition']=='relocated':
            if hashlib.sha256((ROOT/e['to']).read_bytes()).hexdigest()!=e['current_sha256']:raise ValueError('relocated hash mismatch')
        else:raise ValueError('unknown disposition')
    # Actual historical executable sources, tests, protocols and raw evidence stay
    # byte-identical, rather than relying only on an unchanged aggregate score.
    protected=[]
    for path in git('ls-tree','-r','--name-only',base).decode().splitlines():
        if path.startswith(('common/','data/','deploy/','eval/','model/','train/','results/','tests/','scripts/','config/')):
            if (ROOT/path).read_bytes()!=git('show',base+':'+path):raise ValueError('historical content changed: '+path)
            protected.append(path)
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
                historical_files_byte_identical=len(protected),recovered_helper_byte_identical=True,
                current_local_guide_links=links)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=audit()
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
