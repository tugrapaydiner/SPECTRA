import importlib.util
import json
from pathlib import Path
import subprocess
import pytest

spec=importlib.util.spec_from_file_location('archival',Path(__file__).resolve().parents[2]/'scripts/maintenance_archive_branches.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

@pytest.fixture
def repo(tmp_path):
    remote=tmp_path/'remote.git';root=tmp_path/'work'
    subprocess.run(['git','init','--bare',str(remote)],check=True,capture_output=True)
    subprocess.run(['git','clone',str(remote),str(root)],check=True,capture_output=True)
    a.git(root,'config','user.name','Test');a.git(root,'config','user.email','test@example.invalid')
    a.git(root,'checkout','-b','main')
    (root/'base').write_text('base');a.git(root,'add','.');a.git(root,'commit','-m','base')
    base=a.git(root,'rev-parse','HEAD')
    a.git(root,'branch','old',base)
    a.git(root,'checkout','-b','maintenance/consolidate-research-workspace')
    (root/'new').write_text('new');a.git(root,'add','.');a.git(root,'commit','-m','new')
    head=a.git(root,'rev-parse','HEAD');a.git(root,'checkout','main')
    a.git(root,'merge','--no-ff','maintenance/consolidate-research-workspace','-m','merge')
    merge=a.git(root,'rev-parse','HEAD');a.git(root,'push','origin','--all')
    manifest=dict(schema='spectra.branch_cleanup.v1',repository='tugrapaydiner/SPECTRA',reviewed_main=base,
        archive_prefix='archive/2026-09-12/',keep=['main'],merge_branch='maintenance/consolidate-research-workspace',
        branches=[dict(name='old',sha=base,disposition='merged',reason='Test ancestor')])
    return root,manifest,merge

def test_atomic_archival_and_idempotence(repo):
    root,m,merge=repo;plan=a.make_plan(root,m,merge,apply=True)
    assert '--atomic' in plan['command']
    result=a.execute_plan(root,plan);assert result['verified']
    assert a.remote_refs(root,'origin')['refs/heads/main']==merge
    assert not a.make_plan(root,m,merge,apply=True)['command']

def test_advanced_branch_blocks_entire_transaction(repo):
    root,m,merge=repo;plan=a.make_plan(root,m,merge,apply=True)
    a.git(root,'push','origin',merge+':refs/heads/old')
    with pytest.raises(subprocess.CalledProcessError):a.execute_plan(root,plan)
    refs=a.remote_refs(root,'origin')
    assert 'refs/heads/maintenance/consolidate-research-workspace' in refs
    assert not any(ref.startswith('refs/tags/archive/') for ref in refs)
    with pytest.raises(ValueError,match='advanced'):a.make_plan(root,m,merge,apply=True)

def test_wrong_archive_tag_and_protected_branch(repo):
    root,m,merge=repo
    a.git(root,'push','origin',merge+':refs/tags/archive/2026-09-12/old')
    with pytest.raises(ValueError,match='different'):a.make_plan(root,m,merge,apply=True)
    m['branches'][0]['name']='main'
    with pytest.raises(ValueError,match='kept'):a.make_plan(root,m,merge,apply=True)

def test_missing_branch_without_tag_rejected(repo):
    root,m,merge=repo;a.git(root,'push','origin',':refs/heads/old')
    with pytest.raises(ValueError,match='no matching'):a.make_plan(root,m,merge,apply=True)

def test_dry_plan_does_not_change_remote(repo):
    root,m,merge=repo;before=a.remote_refs(root,'origin')
    a.make_plan(root,m,merge)
    assert a.remote_refs(root,'origin')==before
