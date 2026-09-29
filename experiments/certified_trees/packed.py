"""Source-verified packed integer tree certificates; no model fitting.

Bounds implement the preserved exact-rational oracle policy. The exact dyadic
backend avoids per-leaf Fraction objects. A checksum-valid binary is not a proof:
`verify` reconstructs every byte from source; the reference backend stays available.
"""
from __future__ import annotations
import argparse, hashlib, json, struct, time, zlib
from pathlib import Path
from .reference import certificate_oracle as oracle

HEADER = struct.Struct('<8s5Ii7I32s32s')
MAGIC = b'SPCERT02'
MAX_BYTES = 64 * 1024**2

def sha(raw): return hashlib.sha256(raw).hexdigest()

def pack(model):
    if model['format'] != oracle.FORMAT:
        raise ValueError('unknown oracle model')
    d = model['domain']['features']; maximum = model['domain']['maximum']
    c = model['classes']; trees = model['trees']; t = len(trees)
    bits = model['quantization']['bits']; exp = model['quantization']['step_exponent']
    fine = model['quantization']['bound_fraction_bits']
    predicates = sorted({(f,cut) for tr in trees for f,cut in zip(tr['features'],tr['thresholds'])})
    ids = {value:i for i,value in enumerate(predicates)}
    refs = []; descriptors = []; leaves = []
    for tr in trees:
        depth = len(tr['features'])
        descriptors.append((len(refs),depth,len(leaves)//c))
        refs.extend(ids[(f,cut)] for f,cut in zip(tr['features'],tr['thresholds']))
        leaves.extend(v for row in tr['leaves'] for v in row)
    body = b''.join(struct.pack('<Hh',*p) for p in predicates)
    body += b''.join(struct.pack('<III',*tr) for tr in descriptors)
    body += struct.pack('<'+'H'*len(refs),*refs)
    body += struct.pack('<'+'i'*c,*model['bias'])
    body += struct.pack('<'+'q'*(2*c),*(model['error_low']+model['error_high']))
    body += struct.pack('<'+('b' if bits==8 else 'h')*len(leaves),*leaves)
    pair = model['pair_upper']
    if pair is not None:
        body += struct.pack('<'+'q'*(c*c),*(v for row in pair for v in row))
    flags = int(model['binary_source']) | (2 if pair is not None else 0)
    raw = HEADER.pack(MAGIC,d,maximum,c,t,bits,exp,fine,flags,len(predicates),len(refs),len(leaves)//c,
                      len(body),zlib.crc32(body),bytes.fromhex(model['source_sha256']),
                      bytes.fromhex(sha(oracle.canonical(model))))+body
    if len(raw)>MAX_BYTES: raise ValueError('packed model cap')
    metadata(raw)
    return raw

def metadata(raw):
    if type(raw) is not bytes or not HEADER.size<=len(raw)<=MAX_BYTES:
        raise ValueError('packed model byte cap')
    (magic,d,D,c,t,bits,exp,fine,flags,npred,nsplit,nleaf,n,crc,source,compact)=HEADER.unpack_from(raw)
    if magic!=MAGIC or not 1<=d<=256 or not 1<=D<=255 or not 2<=c<=64 or not 1<=t<=4096:
        raise ValueError('unsupported tree geometry')
    if bits not in (8,16) or not -900<=exp<=900 or fine!=20 or flags & ~3 or (flags&1 and c!=2):
        raise ValueError('unsupported numerical policy')
    if npred>65535 or nsplit>12*t or not t<=nleaf or nleaf*c>oracle.MAX_SCALARS:
        raise ValueError('packed inventory cap')
    expected=4*npred+12*t+2*nsplit+20*c+(bits//8)*nleaf*c+(8*c*c if flags&2 else 0)
    if n!=expected or len(raw)!=HEADER.size+n or zlib.crc32(raw[HEADER.size:])!=crc:
        raise ValueError('packed payload/CRC mismatch')
    return {'features':d,'maximum':D,'classes':c,'trees':t,'bits':bits,'step_exponent':exp,
            'bound_fraction_bits':fine,'flags':flags,'predicates':npred,'splits':nsplit,'leaf_rows':nleaf,
            'source_sha256':source.hex(),'oracle_sha256':compact.hex(),'packed_sha256':sha(raw),'bytes':len(raw)}

def compile_bytes(source, maximum, *, features=None, bits=16, pairwise=False, backend='dyadic'):
    if type(backend) is not str or backend not in ('dyadic', 'reference'):
        raise ValueError('unknown exact verification backend')
    if backend == 'dyadic':
        from .dyadic import compile_source
    else:
        compile_source = oracle.compile_source
    obj = compile_source(source, maximum, features=features, bits=bits, pairwise=pairwise)
    return pack(obj), obj

def verify(source, raw, *, backend='dyadic'):
    meta=metadata(raw)
    if sha(source)!=meta['source_sha256']: raise ValueError('original source identity mismatch')
    expected,_=compile_bytes(source,meta['maximum'],features=meta['features'],bits=meta['bits'],pairwise=bool(meta['flags']&2),backend=backend)
    if raw!=expected: raise ValueError('packed tree/bounds differ from exact-rational reconstruction')
    return {'status':'PASS',**meta,'compiler_sha256':sha(Path(__file__).read_bytes()),
            'oracle_source_sha256':sha(Path(oracle.__file__).read_bytes()),
            'verification_backend':backend,
            'verification_source_sha256':sha(Path(__file__).with_name('dyadic.py').read_bytes()) if backend=='dyadic' else sha(Path(oracle.__file__).read_bytes()),
            'scope':'deterministic original-source reconstruction; not authentication or empirical label quality'}

def compile_panel(models, out):
    models=Path(models);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    lock=json.loads((models/'MODEL_LOCK.json').read_text())
    for name,h in lock['files'].items():
        if sha((models/name).read_bytes())!=h: raise ValueError('frozen source changed '+name)
    results=[]
    for task in ('letter','pendigits','satellite','optdigits'):
        info=json.loads((models/task/'FIT.json').read_text())
        source=(models/task/'model.json').read_bytes()
        for bits in (8,16):
            start=time.process_time(); raw,obj=compile_bytes(source,info['maximum'],features=info['features'],bits=bits)
            dest=out/f'{task}-{bits}.sct';dest.write_bytes(raw)
            (out/f'{task}-{bits}.oracle.json').write_bytes(oracle.canonical(obj))
            receipt=verify(source,raw)
            receipt['cpu_seconds_including_reconstruction']=time.process_time()-start
            (out/f'{task}-{bits}.receipt.json').write_text(json.dumps(receipt,indent=2))
            results.append(receipt);print(task,bits,len(raw),'bytes, verified',flush=True)
    (out/'COMPILATION.json').write_text(json.dumps({'models':results,'original_lock_sha256':sha((models/'MODEL_LOCK.json').read_bytes())},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();compile_panel(a.models,a.out)
