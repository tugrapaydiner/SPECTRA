"""Reject incomplete identities without re-running public evaluation inputs."""
import hashlib
import json
import zipfile

import pytest

from experiments.structured_search.analyse import load, RAW_FILES
from experiments.structured_search.study import check_freeze, SOURCE_PATHS, ROOT


def test_empty_or_incomplete_source_inventory_is_rejected():
    sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}
    check_freeze({'sources':sources})
    for removed in SOURCE_PATHS:
        with pytest.raises(ValueError,match='incomplete'):
            check_freeze({'sources':{p:h for p,h in sources.items() if p!=removed}})
    with pytest.raises(ValueError,match='incomplete'):
        check_freeze({'sources':{}})


@pytest.mark.parametrize('missing',sorted(RAW_FILES))
def test_manifest_must_bind_every_required_artifact(tmp_path,missing):
    for name in RAW_FILES:
        (tmp_path/name).write_bytes(b'{}\n')
    files={n:{'bytes':3,'sha256':hashlib.sha256(b'{}\n').hexdigest()} for n in RAW_FILES if n!=missing}
    (tmp_path/'MANIFEST.json').write_text(json.dumps({'status':'COMPLETE','files':files}))
    with pytest.raises(ValueError,match='incomplete artifact'):
        load(tmp_path)


def test_duplicate_archive_member_rejected(tmp_path):
    path=tmp_path/'bad.zip'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('LOCK.json','{}')
        with pytest.warns(UserWarning):
            z.writestr('LOCK.json','{}')
    with pytest.raises(ValueError,match='duplicate archive'):
        load(path)
