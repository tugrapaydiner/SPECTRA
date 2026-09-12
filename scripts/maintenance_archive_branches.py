"""Archive reviewed branch tips with lease-guarded atomic Git ref updates.

No branch is removed unless an archive tag at its EXACT expected tip is created
in the same transaction. A missing branch is accepted only if already archived.
A concurrently changed branch or
tag aborts the transaction. main and explicitly kept branches are never pushed.
Default is a dry plan; --apply is only for the merged, reviewed manifest.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import subprocess


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(['git', '-C', str(root), *args], text=True, stderr=subprocess.PIPE).strip()


def remote_refs(root: Path, remote: str) -> dict[str, str]:
    return {ref: sha for sha, ref in (line.split() for line in git(root,'ls-remote','--heads','--tags',remote).splitlines()) if not ref.endswith('^{}')}


def checked_name(root: Path, name: str, kind: str) -> None:
    if type(name) is not str or not name or name.startswith('-'):
        raise ValueError('invalid ref name')
    git(root, 'check-ref-format', f'refs/{kind}/{name}')


def make_plan(root: Path, manifest: dict, merge_commit: str, *, remote: str = 'origin', apply: bool = False) -> dict:
    if set(manifest) != {'schema','repository','reviewed_main','archive_prefix','keep','merge_branch','branches'}:
        raise ValueError('unexpected manifest fields')
    if manifest['schema'] != 'spectra.branch_cleanup.v1' or manifest['repository'] != 'tugrapaydiner/SPECTRA':
        raise ValueError('wrong repository or cleanup schema')
    if not re.fullmatch(r'[0-9a-f]{40}', merge_commit) or not re.fullmatch(r'[0-9a-f]{40}',manifest['reviewed_main']):
        raise ValueError('full commit SHA required')
    if manifest['archive_prefix'] != 'archive/2026-09-12/':
        raise ValueError('unexpected archive namespace')
    keep=manifest['keep']
    if type(keep) is not list or 'main' not in keep or len(keep)!=len(set(keep)):
        raise ValueError('explicit distinct keep inventory required')
    entries=[dict(e) for e in manifest['branches']]
    parents=git(root,'show','-s','--format=%P',merge_commit).split()
    if len(parents)!=2 or parents[0]!=manifest['reviewed_main']:
        raise ValueError('expected merge of the reviewed base; re-audit changed main')
    merge_branch=manifest['merge_branch']
    if merge_branch!='maintenance/consolidate-research-workspace':
        raise ValueError('unexpected consolidation branch')
    entries.append(dict(name=merge_branch,sha=parents[1],disposition='merged',reason='Reviewed consolidation head is the merge second parent.'))
    if len({e['name'] for e in entries})!=len(entries):raise ValueError('duplicate branch')
    refs=remote_refs(root,remote)
    permitted_main={merge_commit} if apply else {merge_commit,manifest['reviewed_main']}
    if refs.get('refs/heads/main') not in permitted_main:
        raise ValueError('remote main moved; re-audit before mutation')
    leases=[];refspecs=[];records=[]
    for e in entries:
        if set(e)!={'name','sha','disposition','reason'}:raise ValueError('unexpected branch fields')
        name,sha=e['name'],e['sha'];checked_name(root,name,'heads')
        if name in keep:raise ValueError('cannot retire a kept branch')
        if not re.fullmatch(r'[0-9a-f]{40}',sha):raise ValueError('full branch tip SHA required')
        if e['disposition'] not in {'merged','archive_only'} or not e['reason']:
            raise ValueError('explicit reviewed disposition required')
        if e['disposition']=='merged':
            git(root,'merge-base','--is-ancestor',sha,merge_commit)
        if git(root,'cat-file','-t',sha)!='commit':raise ValueError('branch must point to a commit')
        branch='refs/heads/'+name;tag='refs/tags/'+manifest['archive_prefix']+name
        checked_name(root,manifest['archive_prefix']+name,'tags')
        current=refs.get(branch);saved=refs.get(tag)
        if saved not in {None,sha}:raise ValueError('archive tag exists at a different object')
        if current is None:
            if saved!=sha:raise ValueError('missing branch has no matching archive tag')
            records.append({**e,'tag':tag,'state':'already_archived'});continue
        if current!=sha:raise ValueError('branch advanced; refusing to retire unreviewed work')
        if saved is not None:raise ValueError('live branch already has an archive tag; review partial external changes')
        leases.append(f'--force-with-lease={branch}:{sha}')
        if saved is None:
            leases.append(f'--force-with-lease={tag}:')
            refspecs.append(f'{sha}:{tag}')
        refspecs.append(':'+branch)
        records.append({**e,'tag':tag,'state':'planned'})
    command=['git','-C',str(root),'push','--atomic','--porcelain',*leases,remote,*refspecs] if refspecs else []
    return dict(merge_commit=merge_commit,remote=remote,records=records,command=command,before_refs=refs)


def execute_plan(root: Path, plan: dict) -> dict:
    if plan['command']:
        result=subprocess.run(plan['command'],text=True,capture_output=True,check=True)
        output=result.stdout+result.stderr
    else:
        output='All reviewed branches already archived.'
    after=remote_refs(root,plan['remote'])
    for e in plan['records']:
        if after.get(e['tag'])!=e['sha'] or 'refs/heads/'+e['name'] in after:
            raise ValueError('post-push archive verification failed')
    return {**plan,'after_refs':after,'push_output':output,'verified':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--merge-commit',required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    if args.receipt.exists():raise FileExistsError('refusing to overwrite receipt')
    root=Path(git(Path.cwd(),'rev-parse','--show-toplevel'))
    plan=make_plan(root,json.loads(args.manifest.read_text()),args.merge_commit,apply=args.apply)
    result=execute_plan(root,plan) if args.apply else {**plan,'dry_run':True}
    with args.receipt.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'before_refs','after_refs'}},indent=2))

if __name__=='__main__':main()
