"""Independent reconstruction via the preserved Fraction oracle, not Analysis.

This intentionally slower reference checks every certificate and original leaf
bit pattern. It is a checking implementation, not outside researcher replication.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
import zlib
from ..certified_trees.reference import certificate_oracle
from ..certified_trees import packed


def reconstruct(source: bytes, maximum: int, features: int, interned: bool) -> bytes:
    certificate = certificate_oracle.compile_source(source, maximum, features=features, bits=16)
    base = packed.pack(certificate)
    # This parser is separate from the integer-array analyzer in the compiler.
    document = certificate_oracle.loads(source)
    scale, original_bias = document['scale_and_bias']
    binary = len(original_bias) == 1
    classes = 2 if binary else len(original_bias)
    vectors = []
    for tree in document['oblivious_trees']:
        values = tree['leaf_values']
        width = 1 if binary else classes
        for offset in range(0, len(values), width):
            leaf = [float(v) for v in values[offset:offset+width]]
            if binary:
                leaf.insert(0, 0.0)
            vectors.append(struct.pack('<' + 'd'*classes, *leaf))
    bias = [0.0, float(original_bias[0])] if binary else list(map(float, original_bias))
    bank, lookup, indices = [], {}, []
    for vector in vectors:
        if not interned:
            bank.append(vector)
        else:
            if vector not in lookup:
                lookup[vector] = len(bank)
                bank.append(vector)
            indices.append(lookup[vector])
    width = (2 if len(bank) <= 65536 else 4) if interned else 0
    code = struct.pack('<' + ('H' if width == 2 else 'I')*len(indices), *indices) if width else b''
    body = base + struct.pack('<' + 'd'*(classes+1), float(scale), *bias) + code + b''.join(bank)
    header = struct.pack('<8s8I32s32s', b'SPCTOT01', len(base), classes, len(vectors), len(bank), width,
                         len(body), zlib.crc32(body), int(interned), hashlib.sha256(source).digest(),
                         hashlib.sha256(base).digest())
    return header + body


def main(a):
    records = []
    for task in ('letter','pendigits','satellite','optdigits'):
        report = json.loads((a.models/task/'result.json').read_text())
        geometry = report['models']['interned']['info']
        source = (a.sources/'models'/task/'model.json').read_bytes()
        for layout in ('flat','interned'):
            start = time.process_time()
            original = (a.models/task/(layout+'.sctt')).read_bytes()
            checked = reconstruct(source, geometry['maximum'], geometry['features'], layout == 'interned')
            if checked != original:
                raise ValueError('independent reconstruction mismatch: ' + task + '/' + layout)
            records.append({'task':task,'layout':layout,'bytes':len(original),
                            'sha256':hashlib.sha256(checked).hexdigest(),
                            'source_sha256':hashlib.sha256(source).hexdigest(),
                            'cpu_seconds':time.process_time()-start})
    output = {'status':'PASS','files':records,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'oracle_sha256':hashlib.sha256(Path(certificate_oracle.__file__).read_bytes()).hexdigest(),
              'scope':'every artifact byte reconstructed with the preserved Fraction proof implementation'}
    with a.out.open('x') as f: json.dump(output,f,indent=2)
    print(json.dumps(output,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('models','sources','out'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
