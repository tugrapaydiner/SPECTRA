"""Small archive fixtures exercise integrity failures without study data."""
import io
import tarfile
import zipfile

import pytest

from data.ancestry import digest
from scripts.verify_retained_results import read_bound_archive


PAYLOAD = b'{"scientific_status":"NEGATIVE"}\n'
INVENTORY = (digest(PAYLOAD)+'  summary.json\n').encode()


def tar_archive(path, entries):
    with tarfile.open(path, 'w:gz') as archive:
        for name, value, kind in entries:
            info = tarfile.TarInfo(name)
            if kind == 'symlink':
                info.type = tarfile.SYMTYPE
                info.linkname = '/etc/passwd'
                archive.addfile(info)
            else:
                info.size = len(value)
                archive.addfile(info, io.BytesIO(value))


def read(path):
    return read_bound_archive(path, inventory_sha=digest(INVENTORY), zip_sha='a'*64)


def test_canonical_tar_preserves_exact_negative_result(tmp_path):
    path = tmp_path/'evidence.tar.gz'
    tar_archive(path, [('./summary.json', PAYLOAD, 'file'), ('./evidence_sha256.txt', INVENTORY, 'file')])
    assert read(path)['summary.json'] == PAYLOAD


@pytest.mark.parametrize('mutation', ['changed', 'missing', 'extra', 'duplicate', 'traversal', 'symlink', 'inventory'])
def test_invalid_archive_never_reports_integrity_pass(tmp_path, mutation):
    entries = [('summary.json', PAYLOAD, 'file'), ('evidence_sha256.txt', INVENTORY, 'file')]
    if mutation == 'changed': entries[0] = ('summary.json', PAYLOAD+b'changed', 'file')
    elif mutation == 'missing': entries = entries[1:]
    elif mutation == 'extra': entries += [('unbound.json', b'{}', 'file')]
    elif mutation == 'duplicate': entries += [('./summary.json', PAYLOAD, 'file')]
    elif mutation == 'traversal': entries += [('../escape', b'bad', 'file')]
    elif mutation == 'symlink': entries += [('link', b'', 'symlink')]
    else: entries[-1] = ('evidence_sha256.txt', b'changed', 'file')
    path = tmp_path/'evidence.tar.gz'
    tar_archive(path, entries)
    with pytest.raises(ValueError):
        read(path)
    assert not (tmp_path.parent/'escape').exists()


def test_zip_requires_both_container_and_inventory_identity(tmp_path):
    path = tmp_path/'evidence.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('summary.json', PAYLOAD)
        archive.writestr('evidence_sha256.txt', INVENTORY)
    with pytest.raises(ValueError, match='ZIP identity'):
        read(path)
    assert read_bound_archive(path, inventory_sha=digest(INVENTORY),
                              zip_sha=digest(path.read_bytes()))['summary.json'] == PAYLOAD
