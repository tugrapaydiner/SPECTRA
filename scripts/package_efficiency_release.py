"""Build a new versioned evidence package from three matching CI artifacts.

This selects evidence/receipts explicitly, records new hashes for that selection,
and rejects mixed source, incomplete checks and replacement output. It does not
claim that selected files reproduce the outer GitHub artifact ZIP bytes.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import bench_indexed_search as cnf_bench


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_members(raw):
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:*') as archive:
        members = archive.getmembers()
        names = [m.name for m in members if m.isfile()]
        if len(set(names)) != len(names) or any(m.issym() or m.islnk() for m in members):
            raise ValueError('duplicate or linked source member')
        return {m.name: (m.mode, hashlib.sha256(archive.extractfile(m).read()).hexdigest())
                for m in members if m.isfile()}


def check_tests(path):
    root = ET.parse(path).getroot()
    cases = root.findall('.//testcase')
    if not cases or root.findall('.//failure') or root.findall('.//error') or root.findall('.//skipped'):
        raise ValueError('missing, failed or skipped required contracts')
    return len(cases)


def verify_output(folder):
    files = {'config.json', 'source_sha256.json', 'environment.json', 'rows.jsonl', 'summary.json'}
    files |= {f'artifact-{dim}-{depth}.pt' for dim in (16, 64) for depth in (1, 4, 16)}
    hashes = load(folder/'SHA256.json')
    if set(hashes) != files or {p.name for p in folder.iterdir()} != files | {'SHA256.json'}:
        raise ValueError('output evidence inventory mismatch')
    for name in files:
        if sha(folder/name) != hashes[name]:
            raise ValueError('output evidence hash mismatch')
    source_paths = ['scripts/bench_output_only.py', 'spectra/inference.py',
                    'deploy/m10_runtime.py', 'deploy/m10_artifact.py',
                    'deploy/m10_native.py', 'deploy/m10_dense_extension.cpp']
    if load(folder/'source_sha256.json') != {p: sha(ROOT/p) for p in source_paths}:
        raise ValueError('output executable source mismatch')
    config = dict(dims=[16, 64], depths=[1, 4, 16], batches=[1, 4], seed=91426, rounds=5)
    if load(folder/'config.json') != config:
        raise ValueError('output configuration mismatch')
    rows = [json.loads(line) for line in (folder/'rows.jsonl').read_text().splitlines()]
    expected, cells = [], {}
    for dim in config['dims']:
        for depth in config['depths']:
            for batch in config['batches']:
                case = f'd{dim}:depth{depth}:batch{batch}'
                for r in range(5):
                    order = ['trace', 'final'] if (r+batch+depth) % 2 else ['final', 'trace']
                    expected.extend((case, mode, r, dim, depth, batch) for mode in order)
    if len(rows) != len(expected):
        raise ValueError('output row inventory mismatch')
    outputs = {}
    keys = {'case', 'mode', 'round', 'wall_ns', 'logits_sha256', 'answer_sha256',
            'halt_sha256', 'retained_tensor_bytes', 'native_calls', 'halt_calls'}
    for row, (case, mode, r, dim, depth, batch) in zip(rows, expected):
        if set(row) != keys or (row['case'], row['mode'], row['round']) != (case, mode, r):
            raise ValueError('output schedule mismatch')
        for field in ('wall_ns', 'retained_tensor_bytes', 'native_calls', 'halt_calls'):
            if type(row[field]) is not int or row[field] <= 0:
                raise ValueError('invalid output measurement')
        if type(row['round']) is not int:
            raise ValueError('invalid output round')
        digest = tuple(row[k] for k in ('logits_sha256', 'answer_sha256', 'halt_sha256'))
        if any(type(d) is not str or len(d) != 64 or any(c not in '0123456789abcdef' for c in d) for d in digest):
            raise ValueError('invalid tensor digest')
        if outputs.setdefault(case, digest) != digest:
            raise ValueError('non-equivalent final outputs')
        expected_calls = 13*depth if mode == 'trace' else 12*depth+1
        expected_storage = batch*(depth*(128*dim+324)+128) if mode == 'trace' else batch*452
        if (row['native_calls'] != expected_calls or row['retained_tensor_bytes'] != expected_storage or
                row['halt_calls'] != (depth if mode == 'trace' else 1)):
            raise ValueError('output work/storage mismatch')
        cells.setdefault(case, {'trace': [], 'final': [], 'storage': {}})[mode].append(row['wall_ns'])
        cells[case]['storage'][mode] = row['retained_tensor_bytes']
    derived = {}
    for case, values in cells.items():
        derived[case] = dict(mean_ms={m: statistics.mean(values[m])/1e6 for m in ('trace', 'final')},
                            mean_ratio=statistics.mean(values['final'])/statistics.mean(values['trace']),
                            retained_tensor_bytes=values['storage'],
                            retained_ratio=values['storage']['final']/values['storage']['trace'])
    report = load(folder/'summary.json')
    if (report['schema'] != 'spectra.output_only_api.v1' or report['measured_calls'] != 120 or
            report['independent_artifacts'] != 6 or report['cells'] != derived or
            report['exact_final_outputs'] is not True or report['quality_benchmark'] is not False or
            report['untrained_artifacts'] is not True):
        raise ValueError('output summary mismatch')
    return report


def package(inputs, out):
    if out.exists():
        raise FileExistsError('release output already exists')
    folders = sorted(p for p in inputs.iterdir() if p.is_dir())
    cnf = [p for p in folders if p.name.startswith('efficiency-cnf-')]
    neural = [p for p in folders if p.name.startswith('efficiency-output-')]
    if len(folders) != 3 or len(cnf) != 2 or len(neural) != 1:
        raise ValueError('expected exactly two CNF and one output-only artifacts')
    if {p.name.split('-')[2] for p in cnf} != {'3.10', '3.13'}:
        raise ValueError('Python-version evidence matrix mismatch')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    tree = subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT, text=True).strip()
    source = subprocess.check_output(['git', 'archive', '--format=tar.gz', 'HEAD'], cwd=ROOT)
    expected_source = archive_members(source)
    validation = {'source_commit': head, 'source_tree': tree, 'artifacts': {}}
    for folder in folders:
        if ((folder/'source_commit.txt').read_text().strip() != head or
                (folder/'source_tree.txt').read_text().strip() != tree or
                archive_members((folder/'source.tar.gz').read_bytes()) != expected_source):
            raise ValueError('mixed source artifact')
        tests = check_tests(folder/'pytest.xml')
        if folder in cnf:
            report = cnf_bench.verify(folder/'evidence')
            replay = load(folder/'replay.txt')
            install = load(folder/'install/indexed_report.json')
            if (report['systems_gate'] != 'PASS' or replay['exact_path_replays'] != 384 or
                    replay['integrity'] != 'PASS' or replay['answers'] != 1152 or
                    replay['primary'] != report['primary'] or install['status'] != 'PASS' or
                    install['total_checks'] != 11):
                raise ValueError('incomplete required execution checks')
        else:
            report = verify_output(folder/'evidence')
            if tests != 16:
                raise ValueError('output contract inventory mismatch')
        validation['artifacts'][folder.name] = dict(tests=tests, report=report)
    wheel_folder = next(p for p in cnf if '-3.13-' in p.name)
    wheel = wheel_folder/'dist/spectra-0.7.1-py3-none-any.whl'
    with zipfile.ZipFile(wheel) as archive:
        if archive.testzip() is not None:
            raise ValueError('wheel CRC failure')
        for name in ('spectra/cnf/indexed.py', 'spectra/cnf/ranked.py', 'spectra/inference.py'):
            if archive.read(name) != (ROOT/name).read_bytes():
                raise ValueError('wheel does not match tested source')
    out.mkdir(parents=True)
    (out/'SPECTRA-source.tar.gz').write_bytes(source)
    shutil.copy2(wheel, out/wheel.name)
    selected = {}
    with tarfile.open(out/'SPECTRA-efficiency-evidence.tar.gz', 'w:gz') as archive:
        for folder in folders:
            for path in sorted(folder.rglob('*')):
                if not path.is_file() or path.is_symlink():
                    continue
                relative = path.relative_to(folder)
                if relative.parts[0] in ('dist', 'venv') or 'venv' in relative.parts or relative == Path('source.tar.gz'):
                    continue
                name = f'{folder.name}/{relative.as_posix()}'
                selected[name] = dict(sha256=sha(path), size=path.stat().st_size)
                archive.add(path, arcname=name)
        receipt = dict(validation=validation, selected_files=selected,
                       selection_scope='literal evidence and receipts; redundant source archives and wheels omitted')
        raw = (json.dumps(receipt, indent=2, sort_keys=True)+'\n').encode()
        info = tarfile.TarInfo('RELEASE_RECEIPT.json'); info.size = len(raw); info.mode = 0o644
        archive.addfile(info, io.BytesIO(raw))
    lines = ['# SPECTRA 0.7.1: exact search efficiency', '', f'Source `{head}`; tree `{tree}`.', '',
             'Opt-in indexed CPU search preserves every seeded path and original-clause answer.',
             'Historical search and neural replay implementations are unchanged.', '']
    for name, value in validation['artifacts'].items():
        if 'primary' in value['report']:
            primary = value['report']['primary']
            lines.append(f"- {name}: cold/reference ratio {primary['mean_ratio']:.6f}; formula-bootstrap 95% interval {primary['ratio_ci95']}; 1,152 answers and 384 unique execution replays checked.")
    lines += ['', 'Primary scope: 16 large random/planted formulas, 4,096-flip executions, complete cold call cost.',
              'Large timed cases return UNKNOWN. This is lower cost of identical bounded search, not improved SAT success or external-solver superiority.',
              'Every smaller/shorter cell and memory observation remains in the evidence; regressions are not removed.',
              'Prepared warm calls exclude separately measured reusable indexing. Cold allocation can increase.',
              'Output-only inference uses six untrained artifacts with bitwise-equivalent final outputs. Returned tensor storage is not peak inference RAM or RSS; latency is not universally lower.',
              '', 'Source, wheel and selected raw evidence are attached with SHA256SUMS. See docs/EFFICIENCY_GUIDE.md for reproduction and limitations.',
              'Old slow retraining tests are not included in the fast-suite count. No hiring, novelty, energy or new learned-capability claim.', '']
    (out/'RELEASE_NOTES.md').write_text('\n'.join(lines))
    paths = sorted(out.iterdir())
    (out/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in paths))
    return validation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.inputs, args.out), indent=2))
