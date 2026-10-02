import hashlib
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path.cwd()))
from experiments.tree_total.compiler import VerifiedTotal
from experiments.tree_total.session import TotalSession

root = Path('.review/06-retained')
sdk = root / 'original/SPECTRA-residual-evidence/baseline_sdk'
records = []
for task in ('letter', 'pendigits', 'satellite', 'optdigits'):
    folder = root / 'replay-portable' / task
    original = sdk / 'models' / task
    raw = (original / 'model.json').read_bytes()
    data = bytearray((original / 'input.u8').read_bytes())
    indices = (original / 'indices.i32').read_bytes()
    expected = list(struct.unpack('<' + 'i' * (len(indices) // 4), indices))
    scores = (folder / 'reference_scores.f64').read_bytes()
    for layout in ('flat', 'interned'):
        proof = VerifiedTotal(raw, (folder / (layout + '.sctt')).read_bytes())
        with TotalSession(proof, root / 'native-avx2/total.so') as engine:
            for traversal in ('scalar', 'tiled'):
                if engine.scores(data, traversal=traversal) != scores:
                    raise ValueError(task + '/' + layout + '/' + traversal + ': score mismatch')
                for policy in ('total', 'exact', 'audit', 'certificate_only'):
                    got = engine.inspect_buffer(data, policy=policy, traversal=traversal)
                    if len(got['indices']) != len(expected) or any(a != b and not (policy == 'certificate_only' and a == -1) for a, b in zip(got['indices'], expected)):
                        raise ValueError(task + '/' + layout + '/' + policy + ': index mismatch')
            records.append({'task': task, 'layout': layout, 'kind': 'retained', 'rows': len(expected),
                            'score_values': len(scores) // 8, 'score_sha256': hashlib.sha256(scores).hexdigest()})
            if layout == 'interned':
                for kind in ('uniform', 'boundary'):
                    q = bytearray((folder / (kind + '.u8')).read_bytes())
                    wanted = (folder / (kind + '-scores.f64')).read_bytes()
                    label_bytes = (folder / (kind + '-indices.i32')).read_bytes()
                    labels = list(struct.unpack('<' + 'i' * (len(label_bytes) // 4), label_bytes))
                    for traversal in ('scalar', 'tiled'):
                        if engine.scores(q, traversal=traversal) != wanted:
                            raise ValueError(task + '/' + kind + ': stress score mismatch')
                        for policy in ('total', 'exact', 'audit'):
                            if engine.predict_buffer(q, policy=policy, traversal=traversal) != labels:
                                raise ValueError(task + '/' + kind + ': stress label mismatch')
                    records.append({'task': task, 'layout': layout, 'kind': kind, 'rows': len(labels),
                                    'score_values': len(wanted) // 8, 'score_sha256': hashlib.sha256(wanted).hexdigest()})
    print(task, 'AVX2 retained/stress scores and policies PASS', flush=True)
result = {'status': 'PASS', 'target': 'avx2', 'traversals': ['scalar', 'tiled'], 'comparisons': records,
          'unique_rows': 27679, 'unique_score_values': 381942,
          'scope': 'fresh AVX2 execution against portable independent scalar reference; repeated policies/layouts are not independent examples'}
with (root / 'avx2-comparison.json').open('x') as f:
    json.dump(result, f, indent=2)
