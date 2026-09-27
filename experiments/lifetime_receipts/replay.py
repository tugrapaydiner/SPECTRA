"""Create or independently replay all retained prepared-input decision receipts.

Use `python -I -S ... --mode verify` to run the verifier without site packages,
ctypes or the native prediction implementation. Inputs come from the separately
supplied frozen transformed.f64 files, not from the receipt's own input field.
"""
from __future__ import annotations
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from spectra.svm_receipt import create_receipt, verify_receipt


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    if args.mode == 'create':
        from spectra.svm_shared import PreparedModel
        args.receipts.mkdir(parents=True, exist_ok=False)
    forbidden = {'ctypes', 'numpy', 'sklearn', 'torch', 'spectra.svm', 'spectra.svm_shared'}
    report = {'mode': args.mode, 'status': 'PASS', 'models': {}, 'pairs_replayed': 0,
              'kernels_replayed': 0, 'model_input_pairs': 0}
    begin = time.perf_counter()
    for folder in sorted(args.inputs.iterdir()):
        if not folder.is_dir(): continue
        model = folder / 'model.srt'
        raw = model.read_bytes()
        d = struct.unpack_from('<I', raw, 16)[0]
        cases = json.loads((folder / 'cases.json').read_text())
        values = array('d'); values.frombytes((folder / 'transformed.f64').read_bytes())
        if sys.byteorder != 'little': values.byteswap()
        if len(values) != len(cases['expected']) * d: raise ValueError('frozen feature inventory differs')
        target = args.receipts / (folder.name + '.jsonl')
        if args.mode == 'create':
            with PreparedModel(model, args.library, input_dtype='float64') as owner, owner.session() as worker, target.open('x') as out:
                for i, expected in enumerate(cases['expected']):
                    record = create_receipt(worker, values[i*d:(i+1)*d], schedule='binary_stream')
                    if record['label'] != expected: raise ValueError('native prediction changed')
                    out.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + '\n')
        else:
            count = 0
            with target.open() as source:
                for i, line in enumerate(source):
                    if i >= len(cases['expected']): raise ValueError('extra receipt')
                    record = json.loads(line)
                    check = verify_receipt(model, record, expected_input=values[i*d:(i+1)*d], input_dtype='float64')
                    if check['label'] != cases['expected'][i]: raise ValueError('frozen prediction changed')
                    report['pairs_replayed'] += check['pairs_replayed']
                    report['kernels_replayed'] += check['kernels_replayed']
                    count += 1
            if count != len(cases['expected']): raise ValueError('missing receipts')
            if forbidden & sys.modules.keys(): raise ValueError('independent replay loaded native or numerical code')
        report['model_input_pairs'] += len(cases['expected'])
        report['models'][folder.name] = {'pairs': len(cases['expected']), 'model_sha256': sha(model),
            'input_sha256': sha(folder / 'transformed.f64'), 'receipts_sha256': sha(target)}
        print(folder.name, 'PASS', flush=True)
    report['seconds'] = time.perf_counter() - begin
    report['forbidden_modules_loaded'] = sorted(forbidden & sys.modules.keys())
    with args.out.open('x') as stream: json.dump(report, stream, indent=2)
    print('pairs', report['model_input_pairs'], 'seconds', report['seconds'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'receipts', 'out'): p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--library', type=Path)
    p.add_argument('--mode', choices=['create', 'verify'], required=True)
    main(p.parse_args())
