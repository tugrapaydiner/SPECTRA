"""Portable file/source closure check. Integrity relative to a trusted manifest,
not independent verification of scientific conclusions or authenticity.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def inventory(root,exclude='SEAL.json'):
    root=Path(root);files={}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():raise ValueError('symlinks not supported in sealed artifact')
        if p.is_file():
            name=p.relative_to(root).as_posix()
            if name==exclude:continue
            files[name]={'bytes':p.stat().st_size,'sha256':digest(p)}
    return files

def create(root,metadata):
    root=Path(root);path=root/'SEAL.json'
    if path.exists():raise FileExistsError(path)
    document={'format':'spectra.file-closure.v1','files':inventory(root),'metadata':metadata}
    with path.open('x') as f:json.dump(document,f,sort_keys=True,indent=2,allow_nan=False)
    return digest(path)

def verify(root,expected_manifest_sha256=None):
    root=Path(root);path=root/'SEAL.json'
    if path.is_symlink():raise ValueError('manifest symlink')
    if expected_manifest_sha256 is not None and digest(path)!=expected_manifest_sha256:raise ValueError('manifest identity differs')
    def unique(pairs):
        out={}
        for k,v in pairs:
            if k in out:raise ValueError('duplicate manifest key')
            out[k]=v
        return out
    doc=json.loads(path.read_text(),object_pairs_hook=unique)
    if doc.get('format')!='spectra.file-closure.v1':raise ValueError('manifest format')
    actual=inventory(root)
    if actual!=doc['files']:
        changed=sorted(n for n in set(actual)|set(doc['files']) if actual.get(n)!=doc['files'].get(n))
        raise ValueError('artifact differs: '+', '.join(changed[:8]))
    return {'status':'PASS','files':len(actual),'manifest_sha256':digest(path),
            'scope':'exact file inventory and raw bytes; not a scientific or authenticity proof'}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--sha256');a=p.parse_args()
    print(json.dumps(verify(a.root,a.sha256),indent=2))
