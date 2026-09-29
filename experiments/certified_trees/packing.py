"""Compact binary serialization of the exact-rational checkpoint policy.

Source verification is offline. CRC is corruption detection, not authentication.
The runtime must receive trusted compiler output (or an independently trusted hash).
"""
from __future__ import annotations
from array import array
import argparse,hashlib,json,struct,sys,zlib,time
from pathlib import Path
try:
    from . import certificate_oracle as co
except ImportError:
    import certificate_oracle as co
HEADER=struct.Struct('<8s12I32s32s')
LIMIT=64*1024**2

def packed(kind,values):
    a=array(kind,values)
    wanted={'b':1,'h':2,'H':2,'i':4,'I':4,'q':8,'d':8}[kind]
    if a.itemsize!=wanted:raise ValueError('unexpected native scalar width')
    if sys.byteorder!='little':a.byteswap()
    return a.tobytes()

def topology(trees):
    predicates=[];lookup={};splits=[];descriptors=[];offset=0
    for t in trees:
        features=t['features'] if isinstance(t,dict) else t.features
        thresholds=t['thresholds'] if isinstance(t,dict) else t.thresholds
        leaves=t['leaves'] if isinstance(t,dict) else t.leaves
        descriptors.append((offset,len(splits),len(features)))
        for f,threshold in zip(features,thresholds):
            key=(f,threshold)
            if key not in lookup:lookup[key]=len(predicates);predicates.append(key)
            splits.append(lookup[key])
        offset+=sum(len(row) for row in leaves)
    if len(predicates)>65535:raise ValueError('predicate index cap')
    return predicates,splits,descriptors,offset

def header(magic,features,classes,maximum,trees,predicates,splits,scalars,bits,flags,body,source,compiled):
    if len(body)>LIMIT-HEADER.size:raise ValueError('compiled payload cap')
    return HEADER.pack(magic,features,classes,maximum,trees,predicates,splits,scalars,
                       bits,20,flags,len(body),zlib.crc32(body),bytes.fromhex(source),bytes.fromhex(compiled))+body

def pack_compact(model):
    if model['format']!=co.FORMAT:raise ValueError('oracle format')
    pp,sp,tr,n=topology(model['trees']);c=model['classes'];bits=model['quantization']['bits']
    pair=model['pair_upper'];flags=int(pair is not None)
    body=packed('i',model['bias'])+packed('q',model['error_low'])+packed('q',model['error_high'])
    if pair is not None:body+=packed('q',(x for row in pair for x in row))
    body+=b''.join(struct.pack('<Hh',f,t) for f,t in pp)
    body+=b''.join(struct.pack('<III',*row) for row in tr)+packed('H',sp)
    body+=packed('b' if bits==8 else 'h',(x for tree in model['trees'] for row in tree['leaves'] for x in row))
    return header(b'SPTCQ001',model['domain']['features'],c,model['domain']['maximum'],len(tr),len(pp),len(sp),n,bits,flags,body,model['source_sha256'],co.sha(co.canonical(model)))

def pack_original(source):
    pp,sp,tr,n=topology(source.trees)
    body=packed('d',[float(source.scale),*(float(v) for v in source.bias)])
    body+=b''.join(struct.pack('<Hh',f,t) for f,t in pp)
    body+=b''.join(struct.pack('<III',*row) for row in tr)+packed('H',sp)
    body+=packed('d',(float(v) for t in source.trees for row in t.leaves for v in row))
    return header(b'SPTCF001',source.features,source.classes,source.maximum,len(tr),len(pp),len(sp),n,64,0,body,source.digest,'0'*64)

def metadata(raw):
    if type(raw) is not bytes or not HEADER.size<=len(raw)<=LIMIT:raise ValueError('binary byte cap')
    h=HEADER.unpack_from(raw)
    if h[0] not in (b'SPTCQ001',b'SPTCF001') or len(raw)!=HEADER.size+h[11] or zlib.crc32(raw[HEADER.size:])!=h[12]:raise ValueError('binary header/CRC')
    return dict(magic=h[0].decode(),features=h[1],classes=h[2],maximum=h[3],trees=h[4],
                predicates=h[5],splits=h[6],leaf_scalars=h[7],bits=h[8],fraction_bits=h[9],
                pairwise=bool(h[10]),source_sha256=h[13].hex(),oracle_sha256=h[14].hex())

def compile_binary(raw_source,maximum,*,features=None,bits=16,pairwise=False):
    model=co.compile_source(raw_source,maximum,features=features,bits=bits,pairwise=pairwise)
    return pack_compact(model),model

def verify_binary(raw_source,raw):
    m=metadata(raw)
    if m['source_sha256']!=co.sha(raw_source):raise ValueError('source identity mismatch')
    if m['magic']=='SPTCF001':
        expected=pack_original(co.parse_source(raw_source,m['maximum'],m['features']))
    else:expected,_=compile_binary(raw_source,m['maximum'],features=m['features'],bits=m['bits'],pairwise=m['pairwise'])
    if expected!=raw:raise ValueError('binary does not match reconstructed source bounds')
    return dict(status='PASS',source_sha256=co.sha(raw_source),binary_sha256=co.sha(raw),bytes=len(raw))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['compile','verify'])
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--maximum',type=int);p.add_argument('--features',type=int);p.add_argument('--bits',type=int,default=16);p.add_argument('--pairwise',action='store_true')
    a=p.parse_args();raw=a.source.read_bytes()
    if a.mode=='verify':print(json.dumps(verify_binary(raw,a.out.read_bytes()),indent=2));return
    begin=time.perf_counter();blob,model=compile_binary(raw,a.maximum,features=a.features,bits=a.bits,pairwise=a.pairwise)
    with a.out.open('xb') as f:f.write(blob)
    print(json.dumps(dict(**metadata(blob),binary_sha256=co.sha(blob),bytes=len(blob),compile_seconds=time.perf_counter()-begin),indent=2))
if __name__=='__main__':main()
