"""Rebuild README figures from preserved published summaries (standard library only).

This is documentation rendering, NOT a benchmark or independent timing audit.
Changing a source hash requires a deliberate provenance review, not a quiet refresh.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets' / 'readme'
SOURCE_COMMIT = 'b9d9f90d28a3e7071080be38555cd6ca09afcccf'
SOURCES = {
    'docs/EFFICIENCY_GUIDE.md': '98d049b94d7b68a09e112b1f6d5c2b22457865cc782aa1fe3d3454444644292f',
    'experiments/native_baselines/RESULTS.md': 'df69bf9b0cc8d0d798c5160d9341755525de2054eb2663fbb8092ef7cadf6b9f',
}
BG, INK, MUTED, GRID = '#faf8f3', '#292c29', '#686b64', '#e2dfd6'
RUST, SAGE = '#c67b61', '#8a9f98'


def table(text: str, header: str) -> list[list[str]]:
    lines = text.splitlines()
    start = lines.index(header) + 2
    rows = []
    for line in lines[start:]:
        if not line.startswith('|'):
            break
        rows.append([cell.strip() for cell in line.strip('|').split('|')])
    return rows


def load_data() -> dict:
    texts = {}
    for name, digest in SOURCES.items():
        raw = (ROOT / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f'historical chart source changed: {name}; review provenance')
        texts[name] = raw.decode('utf-8')
    cnf = texts['docs/EFFICIENCY_GUIDE.md']
    native = texts['experiments/native_baselines/RESULTS.md']
    indexed = [dict(variables=int(n), family=f, short_ratio=float(s), long_ratio=float(l))
               for n, f, s, l in table(cnf, '| Variables | Family | 128 flips | 4096 flips |')]
    if [(r['variables'], r['family']) for r in indexed] != [
            (n, f) for n in (512, 4096, 16384) for f in ('planted', 'uniform')]:
        raise ValueError('indexed chart requires every published cell in original order')
    rows = table(native, '| Model | LIBSVM AVX2 | Generated C AVX2 | SPECTRA AVX2 default | SPECTRA AVX2 binary_stream |')
    svm = [dict(model=n, libsvm_us=float(l), generated_us=None if g == 'generation timeout' else float(g),
                spectra_default_us=float(s), spectra_binary_stream_us=float(b)) for n, l, g, s, b in rows]
    if [r['model'] for r in svm] != ['Wine', 'WDBC', 'Chess', 'Penguins', 'Titanic', 'Zoo', 'HAR']:
        raise ValueError('native chart requires all seven retained models')
    primary = float(re.search(r'a ratio of \*\*([0-9.]+)\*\*', cnf).group(1))
    linear = float(re.search(r'linear classifier is \*\*([0-9.]+)us', native).group(1))
    return dict(format='spectra.readme-charts.v1', source_commit=SOURCE_COMMIT,
                source_sha256=SOURCES, evidence_scope='Preserved published summaries; rounded table values. No new timing or raw-data re-audit.',
                indexed=dict(cold_reference_primary_ratio=primary, cells=indexed),
                native=dict(unit='microseconds per row', batch_rows=32, cells=svm,
                            har_different_linear_model_us=linear))


def text(x, y, value, *, size=18, fill=INK, anchor='start', serif=False, weight='400'):
    family = 'Georgia, serif' if serif else 'Arial, Helvetica, sans-serif'
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{html.escape(str(value))}</text>')


def line(x1, y1, x2, y2, *, stroke=GRID, dash=''):
    dashed = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}"{dashed}/>'


def rect(x, y, width, height, fill, radius=3):
    return f'<rect x="{x}" y="{y}" width="{width:.3f}" height="{height}" rx="{radius}" fill="{fill}"/>'


def canvas(title, description, height):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(description)}</desc>',
            f'<rect x="0.5" y="0.5" width="999" height="{height-1}" rx="20" fill="{BG}" stroke="{GRID}"/>']


def legend(parts, x, y, label, color):
    parts.extend([rect(x, y-13, 14, 14, color), text(x+24, y, label, size=17)])


def indexed_svg(data):
    info = data['indexed']
    parts = canvas('Indexed CNF search: cold-call cost versus reference',
                   'All six size/family cells, at 128 and 4096 flips. Values above one are regressions. Large cases return UNKNOWN. Retained measurements, not new benchmarks.', 728)
    parts += [text(40, 38, 'SPECTRA  /  BOUNDED SEARCH', size=13, fill=MUTED, weight='700'),
              text(40, 83, 'The gain depends on workload size.', size=32, serif=True),
              text(40, 117, 'Indexed / reference cold-call time. Lower is better.', size=19, fill=MUTED)]
    legend(parts, 40, 154, '128-flip budget', SAGE)
    legend(parts, 245, 154, '4,096-flip budget', RUST)
    parts += [text(945, 148, f"{1/info['cold_reference_primary_ratio']:.2f}x", size=27, anchor='end', weight='700'),
              text(945, 172, 'pooled large-case speedup at 4,096 flips', size=14, fill=MUTED, anchor='end')]
    x, scale, top, step = 236, 550, 215, 66
    bottom = top + 6 * step - 8
    for tick in (0, .25, .5, .75, 1, 1.25):
        px = x + tick * scale
        parts += [line(px, top-17, px, bottom, stroke=MUTED if tick == 1 else GRID, dash='4 5' if tick == 1 else ''),
                  text(px, bottom+29, f'{tick:.2f}', size=15, anchor='middle', fill=MUTED)]
    for i, row in enumerate(info['cells']):
        y = top + i * step
        parts += [text(40, y+16, f"{row['variables']:,} variables", size=18),
                  text(40, y+38, row['family'], size=15, fill=MUTED)]
        for offset, key, color in ((0, 'short_ratio', SAGE), (25, 'long_ratio', RUST)):
            value = row[key]
            parts += [rect(x, y+offset, value*scale, 18, color),
                      text(x+value*scale+9, y+offset+15, f'{value:.3f}', size=16)]
    parts += [line(40, 659, 960, 659),
              text(40, 686, 'Includes preparation. Large cases: UNKNOWN, not improved SAT success.', size=16),
              text(40, 709, 'Retained local summary | 24 formulas, 3 rounds | CPython 3.13.5 | Reference = 1.00', size=14, fill=MUTED), '</svg>']
    return '\n'.join(parts) + '\n'


def native_svg(data):
    rows = data['native']['cells']
    parts = canvas('Same-model native SVM cost, relative to LIBSVM',
                   'All seven retained models; fixed default SPECTRA and generated C relative to native LIBSVM. Generated C is faster than SPECTRA on five of six available small models. The missing HAR export is not a loss.', 806)
    wins = sum(r['generated_us'] is not None and r['generated_us'] < r['spectra_default_us'] for r in rows)
    parts += [text(40, 38, 'SPECTRA  /  SAME-MODEL CPU INFERENCE', size=13, fill=MUTED, weight='700'),
              text(40, 83, 'Native acceleration is not universal.', size=32, serif=True),
              text(40, 117, 'Prepared-input job cost / native LIBSVM. Lower is better.', size=19, fill=MUTED)]
    legend(parts, 40, 154, 'SPECTRA default', RUST)
    legend(parts, 265, 154, 'Generated C', SAGE)
    parts += [text(945, 148, f'{wins}/6', size=27, anchor='end', weight='700'),
              text(945, 172, 'small models favor generated C', size=14, fill=MUTED, anchor='end')]
    x, scale, top, step = 205, 556, 215, 65
    bottom = top+7*step-8
    for tick in (0, .25, .5, .75, 1, 1.25):
        px = x+tick*scale
        parts += [line(px, top-17, px, bottom, stroke=MUTED if tick == 1 else GRID, dash='4 5' if tick == 1 else ''),
                  text(px, bottom+29, f'{tick:.2f}', size=15, anchor='middle', fill=MUTED)]
    for i, row in enumerate(rows):
        y = top+i*step
        parts.append(text(40, y+26, row['model'], size=19))
        for offset, key, color in ((0, 'spectra_default_us', RUST), (25, 'generated_us', SAGE)):
            value = row[key]
            if value is None:
                parts.append(text(x+8, y+offset+15, 'Not available: generation timed out', size=16, fill=MUTED))
            else:
                ratio = value/row['libsvm_us']
                parts += [rect(x, y+offset, ratio*scale, 18, color),
                          text(x+ratio*scale+9, y+offset+15, f'{ratio:.3f}', size=16)]
    parts += [line(40, 719, 960, 719),
              text(40, 748, 'HAR: a different linear model is faster still, with higher observed accuracy.', size=16),
              text(40, 773, 'Retained 2026-09-27 summary | EPYC 9V74, matched AVX2 | Batch 32, 7 rounds', size=14, fill=MUTED),
              text(40, 794, 'Includes validation, conversion and fresh labels; excludes loading and raw preprocessing.', size=13, fill=MUTED), '</svg>']
    return '\n'.join(parts) + '\n'


def outputs():
    data = load_data()
    return {'chart-data.json': json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+'\n',
            'indexed-search.svg': indexed_svg(data), 'native-comparison.svg': native_svg(data)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='fail on missing or stale generated files; never write')
    args = parser.parse_args(argv)
    rendered = outputs()
    if args.check:
        stale = [name for name, value in rendered.items()
                 if not (OUT/name).is_file() or (OUT/name).read_bytes() != value.encode('utf-8')]
        if stale:
            parser.exit(1, 'stale README charts: '+', '.join(stale)+'\n')
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        for name, value in rendered.items():
            (OUT/name).write_bytes(value.encode('utf-8'))
    print(json.dumps({'status': 'PASS', 'files': list(rendered),
                      'scope': 'deterministic summary rendering, not a benchmark'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
