import hashlib,json,zipfile
import pytest
from scripts.m16_bundle import checked_name,verify_bundle

@pytest.mark.parametrize('name',['/absolute','../escape','ok/../../bad','dir\\file'])
def test_archive_paths_reject_traversal(name):
    with pytest.raises(ValueError):checked_name(name)


def make_zip(tmp_path,*,bad=False):
    p=tmp_path/'example.zip';raw=b'valid payload'
    with zipfile.ZipFile(p,'w') as z:
        z.writestr('source/one.py',raw if not bad else b'changed')
        z.writestr('MANIFEST.json',json.dumps({'files':{'source/one.py':{'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}}}))
        z.writestr('BUNDLE_README.md','fixture')
    return p


def test_bundle_payloads_rehashed(tmp_path):assert verify_bundle(make_zip(tmp_path))


def test_bundle_corrupt_payload_rejected(tmp_path):
    with pytest.raises(ValueError,match='hash mismatch'):verify_bundle(make_zip(tmp_path,bad=True))
