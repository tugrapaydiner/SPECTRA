"""Source-bound total tree model: original proof plus lossless binary64 leaves.

Only verified bytes are accepted by the public runtime. Storage layouts are
model-only choices, never calibrated against inputs. Verification reconstructs
all proof, routing, original arithmetic and lossless-bank bytes from the source.
"""
from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
import hashlib
import struct
import zlib
from pathlib import Path
from .exact import Analysis, scalar, need
from ..certified_trees import packed
from ..certified_trees.reference import certificate_oracle as oracle

HEADER = struct.Struct('<8s8I32s32s')
MAGIC = b'SPCTOT01'
MAX_BYTES = 64 * 1024**2


def compile_bytes(source: bytes, maximum: int, *, features=None, layout='interned') -> bytes:
    need(type(layout) is str and layout in ('flat', 'interned'), 'unknown lossless layout')
    analysis = Analysis(source, maximum, features)
    base = packed.pack(analysis.compile(16, False))
    c = analysis.c
    doc = oracle.loads(source)
    rows, bank, ids, lookup = 0, [], [], {}
    for tree in doc['oblivious_trees']:
        values = tree['leaf_values']
        stride = 1 if analysis.binary else c
        for start in range(0, len(values), stride):
            leaf = tuple(scalar(v) for v in values[start:start+stride])
            if analysis.binary:
                leaf = (0.0, leaf[0])
            raw = struct.pack('<'+'d'*c, *leaf)
            # Intern by binary64 bytes, never rounded values or approximate equality.
            if layout == 'interned':
                i = lookup.get(raw)
                if i is None:
                    i = len(bank)
                    lookup[raw] = i
                    bank.append(raw)
                ids.append(i)
            else:
                bank.append(raw)
            rows += 1
    id_width = (2 if len(bank) <= 65536 else 4) if layout == 'interned' else 0
    mapping = struct.pack('<'+('H' if id_width == 2 else 'I')*rows, *ids) if id_width else b''
    biases = tuple(map(scalar, doc['scale_and_bias'][1]))
    if analysis.binary:
        biases = (0.0, biases[0])
    body = base + struct.pack('<'+'d'*(c+1), scalar(doc['scale_and_bias'][0]), *biases)
    body += mapping + b''.join(bank)
    result = HEADER.pack(MAGIC, len(base), c, rows, len(bank), id_width,
                         len(body), zlib.crc32(body), int(layout == 'interned'),
                         hashlib.sha256(source).digest(), hashlib.sha256(base).digest()) + body
    need(len(result) <= MAX_BYTES, 'total model byte cap exceeded')
    return result


def metadata(raw: bytes) -> dict:
    need(type(raw) is bytes and HEADER.size <= len(raw) <= MAX_BYTES, 'total model byte cap')
    magic, nb, c, rows, unique, width, body, crc, flags, source, basehash = HEADER.unpack_from(raw)
    need(magic == MAGIC and 2 <= c <= 64 and flags in (0, 1), 'unsupported total-model header')
    need(1 <= unique <= rows and rows*c <= 500000, 'lossless leaf inventory')
    need((flags == 0 and width == 0 and unique == rows) or
         (flags == 1 and width == (2 if unique <= 65536 else 4)), 'invalid leaf-index width')
    need(body == nb + 8*(c+1) + width*rows + 8*unique*c and len(raw) == HEADER.size + body,
         'total model size mismatch')
    need(zlib.crc32(raw[HEADER.size:]) == crc, 'total-model CRC mismatch')
    base = raw[HEADER.size:HEADER.size+nb]
    need(hashlib.sha256(base).digest() == basehash, 'compact base hash mismatch')
    m = packed.metadata(base)
    need(m['classes'] == c and m['leaf_rows'] == rows and m['bits'] == 16 and not m['flags'] & 2,
         'compact/lossless geometry mismatch')
    need(bytes.fromhex(m['source_sha256']) == source, 'source identity mismatch')
    return {**m, 'model_sha256': hashlib.sha256(raw).hexdigest(), 'model_bytes': len(raw),
            'base_bytes': nb, 'layout': 'interned' if flags else 'flat', 'unique_leaf_vectors': unique,
            'leaf_index_bytes': width*rows, 'source_bank_bytes': unique*c*8,
            'exact_tail_bytes': 8*(c+1)+width*rows+unique*c*8}


def verify(source: bytes, raw: bytes) -> dict:
    m = metadata(raw)
    need(hashlib.sha256(source).hexdigest() == m['source_sha256'], 'source hash differs')
    expected = compile_bytes(source, m['maximum'], features=m['features'], layout=m['layout'])
    need(raw == expected, 'total proof or lossless source differs from exact reconstruction')
    return {'status': 'PASS', **m, 'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope': 'full source reconstruction, not a hash-only proof or author authentication'}


@dataclass(frozen=True, init=False)
class VerifiedTotal:
    raw: bytes
    info: object

    def __init__(self, source: bytes, raw: bytes):
        result = verify(source, raw)
        object.__setattr__(self, 'raw', raw)
        object.__setattr__(self, 'info', MappingProxyType(result))

    @classmethod
    def from_files(cls, source, model):
        with Path(source).open('rb') as stream:
            text = stream.read(oracle.MAX_SOURCE_BYTES+1)
        with Path(model).open('rb') as stream:
            raw = stream.read(MAX_BYTES+1)
        return cls(text, raw)
