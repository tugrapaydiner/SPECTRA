from pathlib import Path
import hashlib,json
import pytest
from .runner import run_bound,SourceMismatch
from .seal import create,verify

def test_result_binds_executed_bytes(tmp_path):
    p=tmp_path/'audit.py';p.write_text('def audit(root): return {"status":"PASS", "synthetic_test":True}\n')
    h=hashlib.sha256(p.read_bytes()).hexdigest();out=tmp_path/'receipt.json'
    r=run_bound(p,tmp_path,out,h)
    assert r['auditor_sha256']==h and json.loads(out.read_text())==r
    with pytest.raises(FileExistsError):run_bound(p,tmp_path,out,h)

@pytest.mark.parametrize('change',[b'\n# appended comment',b'\nraise RuntimeError("must not execute")'])
def test_mismatched_source_never_executes(tmp_path,change):
    marker=tmp_path/'executed'
    p=tmp_path/'audit.py';p.write_text(f'from pathlib import Path\nPath({str(marker)!r}).touch()\ndef audit(root):return {{"status":"PASS"}}\n')
    h=hashlib.sha256(p.read_bytes()).hexdigest();p.write_bytes(p.read_bytes()+change)
    with pytest.raises(SourceMismatch):run_bound(p,tmp_path,tmp_path/'out.json',h)
    assert not marker.exists() and not (tmp_path/'out.json').exists()

@pytest.mark.parametrize('source',['def audit(root):return {"status":"FAIL"}\n','def audit(root):raise ValueError("bad evidence")\n'])
def test_audit_failure_does_not_publish(tmp_path,source):
    p=tmp_path/'audit.py';p.write_text(source)
    with pytest.raises(ValueError):run_bound(p,tmp_path,tmp_path/'out',hashlib.sha256(p.read_bytes()).hexdigest())
    assert not (tmp_path/'out').exists()

@pytest.mark.parametrize('damage',['changed','missing','extra','symlink','manifest'])
def test_sealed_closure_rejects_changes(tmp_path,damage):
    root=tmp_path/'pack';root.mkdir();(root/'source.py').write_text('a=1\n');(root/'model.bin').write_bytes(b'fixture')
    h=create(root,{'scope':'synthetic test fixture'})
    assert verify(root,h)['files']==2
    if damage=='changed':(root/'source.py').write_text('a=2\n')
    elif damage=='missing':(root/'model.bin').unlink()
    elif damage=='extra':(root/'extra').touch()
    elif damage=='symlink':(root/'link').symlink_to(root/'source.py')
    else:(root/'SEAL.json').write_text('{}')
    with pytest.raises(ValueError):verify(root,h)
