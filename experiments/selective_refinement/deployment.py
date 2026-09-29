"""Model-bound experimental deployment policy, without numerical dependencies.

Hash binding prevents accidental mixing of models/policies. It is not a signature,
a risk guarantee or a sandbox for untrusted native libraries/directories.
"""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
from .session import RefinementSession

def read_policy(folder,expected_sha256=None):
    folder=Path(folder);path=folder/'policy.json'
    if path.is_symlink():raise ValueError('policy symlink not supported')
    with path.open('rb') as f:raw=f.read(65537)
    if len(raw)>65536:raise ValueError('policy byte cap')
    digest=hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest!=expected_sha256:raise ValueError('policy identity mismatch')
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise ValueError('duplicate policy key')
            result[k]=v
        return result
    def bad(value):raise ValueError('nonfinite policy number')
    doc=json.loads(raw,object_pairs_hook=unique,parse_constant=bad)
    if type(doc) is not dict or doc.get('format')!='spectra.refinement.policy.v1' or doc.get('experimental') is not True:
        raise ValueError('unsupported experimental policy')
    if set(doc.get('models',{}))!={'fast.spp','strong.srt'}:raise ValueError('policy model inventory')
    for name,wanted in doc['models'].items():
        if type(wanted) is not dict or type(wanted.get('bytes')) is not int or not 1<=wanted['bytes']<=64*1024**2 or re.fullmatch('[0-9a-f]{64}',wanted.get('sha256','')) is None:
            raise ValueError('invalid model identity')
        p=folder/name
        if p.is_symlink() or not p.is_file() or p.stat().st_size!=wanted['bytes']:raise ValueError('model file mismatch')
        if hashlib.sha256(p.read_bytes()).hexdigest()!=wanted['sha256']:raise ValueError('model content mismatch')
    if set(doc.get('policies',{}))!={'primary','strict','loose'}:raise ValueError('policy names')
    return doc,digest

def load_deployment(folder,library,*,policy='primary',expected_policy_sha256=None):
    if type(policy) is not str or policy not in ('primary','strict','loose'):raise ValueError('unknown deployment policy')
    doc,digest=read_policy(folder,expected_policy_sha256)
    p=doc['policies'][policy]
    if type(p) is not dict or 'threshold' not in p or 'accept_fraction' not in p:raise ValueError('missing routing settings')
    folder=Path(folder)
    model=RefinementSession(folder/'fast.spp',folder/'strong.srt',library,
                            threshold=p['threshold'],accept_fraction=p['accept_fraction'])
    model.policy_sha256=digest
    return model
