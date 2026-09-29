"""Bind an audit result to the exact trusted auditor bytes that executed it.

This is provenance/consistency tooling, not a sandbox, digital signature, or a
replacement for the scientific checks performed by the selected auditor.
"""
from __future__ import annotations
import hashlib,json,os,re,sys,tempfile,types
from pathlib import Path

class SourceMismatch(ValueError):
    pass

def publish(path:Path, data:bytes):
    path=Path(path)
    if path.exists() or path.is_symlink():raise FileExistsError(path)
    fd,temporary=tempfile.mkstemp(prefix='.audit-',suffix='.partial',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.link(temporary,path) # same-directory create-if-absent, no overwrite fallback
    finally:
        try:os.unlink(temporary)
        except FileNotFoundError:pass

def run_bound(auditor,evidence,output,expected_sha256):
    auditor,evidence,output=map(Path,(auditor,evidence,output))
    if output.exists() or output.is_symlink():raise FileExistsError(output)
    if not isinstance(expected_sha256,str) or re.fullmatch('[0-9a-f]{64}',expected_sha256) is None:
        raise ValueError('expected raw SHA256 must come from a trusted source reference')
    if auditor.is_symlink() or not auditor.is_file():raise ValueError('regular auditor source required')
    with auditor.open('rb') as f:raw=f.read(2*1024**2+1)
    if len(raw)>2*1024**2:raise ValueError('auditor source cap')
    digest=hashlib.sha256(raw).hexdigest()
    if digest!=expected_sha256:raise SourceMismatch('auditor bytes differ; refusing to run or copy an old PASS')
    raw.decode('utf-8') # reject an accidental non-source target before executing
    module=types.ModuleType('_spectra_bound_auditor');module.__file__=str(auditor.resolve())
    # Execute the exact bytes just hashed, not a second filesystem import.
    sys.modules[module.__name__]=module
    try:
        exec(compile(raw,str(auditor.resolve()),'exec'),module.__dict__)
        if not callable(getattr(module,'audit',None)):raise ValueError('auditor lacks audit(root)')
        result=module.audit(evidence)
    finally:sys.modules.pop(module.__name__,None)
    if type(result) is not dict or result.get('status')!='PASS':raise ValueError('scientific auditor did not return PASS')
    receipt={'status':'PASS','format':'spectra.source-bound-audit.v1',
             'auditor_sha256':digest,'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'auditor_path':str(auditor.resolve()),'evidence_path':str(evidence.resolve()),
             'result':result,'scope':'exact auditor bytes executed; scientific scope is that of the nested result'}
    publish(output,(json.dumps(receipt,indent=2,allow_nan=False)+'\n').encode())
    return receipt

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('auditor','evidence','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--expected-sha256',required=True);a=p.parse_args()
    print(json.dumps(run_bound(a.auditor,a.evidence,a.output,a.expected_sha256),indent=2))
