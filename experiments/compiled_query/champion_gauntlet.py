"""Development-only strong-competitor gauntlet and chart/report generator.

This study diagnoses where SPECTRA wins and loses before a later untouched
holdout.  It never rewrites the frozen WAP result and never turns post-open tuning
into confirmation evidence.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
import statistics

from .audit import audit_study
from .cases import canonical, digest, write_case
from .study import Config, create_study, run_study
from .transfer_panel import make_case

SCHEMA = 'spectra.champion_gauntlet.development.v1'
FAMILY_REPLICAS = (
    ('random_tree', 0),
    ('geometric', 1),
    ('preferential', 2),
    ('rewired_ring', 3),
    ('communities', 4),
    ('random_bipartite', 5),
)
COUNTS = (1, 64, 1024)
REPEATS = 3
SPECTRA_ARMS = ('exact-raw', 'exact-scc', 'exact-route')
COMPETITOR_ARMS = (
    'exact-raw',
    'cadical3',
    'minicard',
    'cadical195',
    'cudd-reduced-bfs',
    'ortools-cpsat',
    'kissat4-rebuild',
)
ARMS = tuple(dict.fromkeys(SPECTRA_ARMS + COMPETITOR_ARMS))
ARM_LABELS = {
    'exact-raw': 'SPECTRA raw + proofs',
    'exact-scc': 'SPECTRA SCC + proofs',
    'exact-route': 'SPECTRA route + proofs',
    'cadical3': 'CaDiCaL 3.0.1 persistent',
    'minicard': 'MiniCard persistent',
    'cadical195': 'CaDiCaL 1.9.5 persistent',
    'cudd-reduced-bfs': 'Reduced CUDD BFS',
    'ortools-cpsat': 'OR-Tools CP-SAT',
    'kissat4-rebuild': 'Kissat 4.0.4 rebuild/query',
}
EXTERNAL_IDENTITIES = {
    'cadical3': dict(name='cadical3', version='3.0.1',
                     upstream_commit='c60730422e758ef1cebe7aeddf2dda31c996bf04'),
    'kissat4': dict(name='kissat4', version='4.0.4',
                    upstream_commit='8af8e56f174b778aef3aa45af9f739b2a5f492c2'),
}

ARM_CLASSES = {
    'exact-raw': 'proof-carrying specialized',
    'exact-scc': 'proof-carrying specialized',
    'exact-route': 'proof-carrying specialized',
    'cadical3': 'persistent general SAT',
    'minicard': 'persistent cardinality SAT',
    'cadical195': 'persistent general SAT',
    'cudd-reduced-bfs': 'compile-once symbolic',
    'ortools-cpsat': 'retained model / fresh CP-SAT search',
    'kissat4-rebuild': 'one-shot solver rebuilt per query',
}



def verify_external_receipt(path: Path, key: str) -> dict:
    expected = EXTERNAL_IDENTITIES[key]
    receipt_path = path.with_name(path.name + '.receipt.json')
    if not path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f'pinned {key} library/receipt missing')
    receipt = json.loads(receipt_path.read_bytes())
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise ValueError(f'{key} receipt {field} differs')
    if receipt.get('library_sha256') != digest(path.read_bytes()):
        raise ValueError(f'{key} receipt/library digest differs')
    return receipt

def gmean(values: list[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError('geometric mean requires positive finite values')
    return math.exp(statistics.fmean(math.log(value) for value in values))


def write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload)


def freeze_cases(destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=False)
    paths = []
    inventory = []
    for family, replica in FAMILY_REPLICAS:
        case = make_case(family, replica, count=max(COUNTS))
        path = destination / f'{case.name}.json'
        write_case(path, case)
        paths.append(path)
        inventory.append(dict(name=case.name, family=family, replica=replica,
                              n=case.problem.n, edges=len(case.problem.edges),
                              domain_regime=replica % 3,
                              case_sha256=digest(path.read_bytes())))
    write_once(destination / 'inventory.json', canonical(dict(
        schema=SCHEMA, classification='exposed development diagnosis; never a holdout',
        counts=COUNTS, repeats=REPEATS, arms=ARMS, cases=inventory)))
    return paths


def prepare(output: Path, *, quotient_library: Path, exact_library: Path,
            cadical3_library: Path, kissat4_library: Path) -> None:
    verify_external_receipt(Path(cadical3_library), 'cadical3')
    verify_external_receipt(Path(kissat4_library), 'kissat4')
    root = output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    cases_dir = root / 'gauntlet-cases'
    freeze_cases(cases_dir)
    paths = sorted(p for p in cases_dir.glob('*.json') if p.name != 'inventory.json')
    create_study(paths, quotient_library, root / 'study',
                 Config(arms=ARMS, prefixes=COUNTS, repeats=REPEATS,
                        seed=20261017, wall_seconds=120.0,
                        address_space_bytes=3 << 30,
                        native_bytes=512 << 20,
                        bdd_nodes=2_000_000),
                 exact_library=exact_library,
                 external_libraries={'cadical3': cadical3_library,
                                     'kissat4': kissat4_library})
    write_once(root / 'freeze.json', canonical(dict(
        schema=SCHEMA, study_manifest_sha256=digest((root/'study/manifest.json').read_bytes()),
        case_inventory_sha256=digest((cases_dir/'inventory.json').read_bytes()),
        statement=('All algorithms, versions, cases, counts, orders and accounting '
                   'were fixed before this development execution.'))))


def _lookup(audit: dict) -> dict[tuple[str, int, str], dict]:
    return {(row['case'], row['count'], row['arm']): row
            for row in audit['session_summaries']}


def _ratios(audit: dict, count: int, candidate: str, baseline: str) -> dict[str, float]:
    lookup = _lookup(audit)
    result = {}
    for case in sorted({row['case'] for row in audit['session_summaries']}):
        a = lookup.get((case, count, candidate))
        b = lookup.get((case, count, baseline))
        if a and b and a['eligible'] and b['eligible']:
            result[case] = a['mean_complete_ns'] / b['mean_complete_ns']
    return result


def _oracle_ratios(audit: dict, count: int, baseline: str) -> tuple[dict[str, float], dict[str, str]]:
    lookup = _lookup(audit)
    ratios, modes = {}, {}
    candidate_arms = tuple(arm for arm in SPECTRA_ARMS if arm != baseline)
    for case in sorted({row['case'] for row in audit['session_summaries']}):
        competitor = lookup.get((case, count, baseline))
        candidates = [(arm, lookup.get((case, count, arm))) for arm in candidate_arms]
        if (competitor and competitor['eligible'] and
                all(row and row['eligible'] for _, row in candidates)):
            arm, row = min(candidates, key=lambda item: item[1]['mean_complete_ns'])
            ratios[case] = row['mean_complete_ns'] / competitor['mean_complete_ns']
            modes[case] = arm
    return ratios, modes


def _svg_bar(path: Path, title: str, rows: list[tuple[str, float]], *,
             x_label: str = 'SPECTRA / competitor complete-time ratio') -> None:
    width, left, right, top, row_h = 1000, 290, 60, 80, 42
    height = top + row_h * len(rows) + 80
    plot_w = width - left - right
    maximum = max([1.25, *(value for _, value in rows)])
    def x(value: float) -> float:
        return left + min(value / maximum, 1.0) * plot_w
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<style>text{font-family:system-ui,sans-serif;fill:#111}.title{font-size:24px;font-weight:700}.label{font-size:15px}.value{font-size:14px;font-weight:650}.axis{stroke:#555;stroke-width:1}.gate{stroke:#111;stroke-width:2;stroke-dasharray:7 6}.bar{fill:#555}.win{fill:#222}.bg{fill:#fff}</style>',
             f'<rect class="bg" width="{width}" height="{height}"/>',
             f'<text class="title" x="24" y="38">{html.escape(title)}</text>',
             f'<line class="gate" x1="{x(1):.1f}" x2="{x(1):.1f}" y1="55" y2="{height-55}"/>',
             f'<text class="label" x="{x(1)+6:.1f}" y="72">tie = 1.0</text>']
    for index, (label, value) in enumerate(rows):
        y = top + index * row_h
        bar_w = max(1.0, x(value) - left)
        css = 'win' if value < 1 else 'bar'
        parts += [f'<text class="label" x="20" y="{y+24}">{html.escape(label)}</text>',
                  f'<rect class="{css}" x="{left}" y="{y+7}" width="{bar_w:.1f}" height="24"/>',
                  f'<text class="value" x="{min(x(value)+8, width-70):.1f}" y="{y+25}">{value:.3f}</text>']
    parts += [f'<line class="axis" x1="{left}" x2="{width-right}" y1="{height-50}" y2="{height-50}"/>',
              f'<text class="label" x="{left}" y="{height-18}">{html.escape(x_label)}; lower is better for SPECTRA</text>',
              '</svg>']
    path.write_text('\n'.join(parts), encoding='utf-8')


def report(study: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    audit = audit_study(study)
    write_once(output/'audit.json', canonical(audit))
    summary = dict(schema=SCHEMA, classification='development-only; exposed cases',
                   primary_mode='exact-scc',
                   functional_complete=audit['all_scheduled_decisions_complete'],
                   scheduled_jobs=audit['scheduled_jobs'], outcomes=audit['outcomes'],
                   audited_decisions=audit['audited_decisions'], counts={})
    csv_rows = []
    for count in COUNTS:
        count_result = {}
        for baseline in COMPETITOR_ARMS:
            fixed = {}
            for candidate in SPECTRA_ARMS:
                if candidate == baseline:
                    fixed[candidate] = None
                    continue
                values = _ratios(audit, count, candidate, baseline)
                fixed[candidate] = (dict(eligible_cases=len(values), geomean_ratio=gmean(list(values.values())),
                                         wins=sum(v < .99 for v in values.values()),
                                         ties=sum(.99 <= v <= 1.01 for v in values.values()),
                                         losses=sum(v > 1.01 for v in values.values()),
                                         per_case=values) if values else None)
            oracle, modes = _oracle_ratios(audit, count, baseline)
            oracle_row = (dict(eligible_cases=len(oracle), geomean_ratio=gmean(list(oracle.values())),
                               wins=sum(v < .99 for v in oracle.values()),
                               ties=sum(.99 <= v <= 1.01 for v in oracle.values()),
                               losses=sum(v > 1.01 for v in oracle.values()),
                               selected_modes=modes, per_case=oracle,
                               warning='post-hoc best SPECTRA mode; diagnostic only, not deployable')
                          if oracle else None)
            count_result[baseline] = dict(fixed=fixed, oracle_best=oracle_row,
                                          competitor_class=ARM_CLASSES[baseline])
            if oracle_row:
                csv_rows.append(dict(count=count, competitor=baseline,
                    competitor_label=ARM_LABELS[baseline], competitor_class=ARM_CLASSES[baseline],
                    spectra_mode='oracle-best-of-raw-scc-route', ratio=oracle_row['geomean_ratio'],
                    wins=oracle_row['wins'], ties=oracle_row['ties'], losses=oracle_row['losses'],
                    eligible_cases=oracle_row['eligible_cases']))
            for candidate, row in fixed.items():
                if row:
                    csv_rows.append(dict(count=count, competitor=baseline,
                        competitor_label=ARM_LABELS[baseline], competitor_class=ARM_CLASSES[baseline],
                        spectra_mode=candidate, ratio=row['geomean_ratio'], wins=row['wins'],
                        ties=row['ties'], losses=row['losses'], eligible_cases=row['eligible_cases']))
        summary['counts'][str(count)] = count_result
    write_once(output/'summary.json', canonical(summary))
    fields = ('count', 'competitor', 'competitor_label', 'competitor_class',
              'spectra_mode', 'ratio', 'wins', 'ties', 'losses', 'eligible_cases')
    with (output/'matrix.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(csv_rows)

    for count in (1, 1024):
        primary_rows, oracle_rows = [], []
        for baseline in COMPETITOR_ARMS:
            primary = summary['counts'][str(count)][baseline]['fixed'].get('exact-scc')
            oracle = summary['counts'][str(count)][baseline]['oracle_best']
            if primary:
                primary_rows.append((ARM_LABELS[baseline], primary['geomean_ratio']))
            if oracle:
                oracle_rows.append((ARM_LABELS[baseline], oracle['geomean_ratio']))
        primary_rows.sort(key=lambda item: item[1])
        oracle_rows.sort(key=lambda item: item[1])
        _svg_bar(output/f'gauntlet-primary-q{count}.svg',
                 f'Fixed SPECTRA SCC — {count} query' + ('ies' if count != 1 else ''), primary_rows)
        _svg_bar(output/f'gauntlet-oracle-diagnostic-q{count}.svg',
                 f'Oracle diagnostic only — {count} query' + ('ies' if count != 1 else ''), oracle_rows)

    lines = [
        '# SPECTRA strong-competitor development gauntlet', '',
        '> Exposed development cases. This report diagnoses wins and losses; it is not a new holdout or world-leading claim.', '',
        f"Functional completion: **{summary['functional_complete']}**  ",
        f"Scheduled jobs: **{summary['scheduled_jobs']}**  ",
        f"Audited decisions: **{sum(summary['audited_decisions'].values()):,}**", '',
    ]
    for count in COUNTS:
        noun = 'query' if count == 1 else 'queries'
        lines += [f'## {count} {noun}', '',
                  '| Competitor | Class | Fixed SCC ratio | SCC W/T/L | Oracle diagnostic | Eligible cases |',
                  '|---|---|---:|---:|---:|---:|']
        for baseline in COMPETITOR_ARMS:
            cell = summary['counts'][str(count)][baseline]
            primary = cell['fixed'].get('exact-scc')
            oracle = cell['oracle_best']
            if primary:
                oracle_text = f"{oracle['geomean_ratio']:.4f}" if oracle else 'incomplete'
                lines.append(f"| {ARM_LABELS[baseline]} | {ARM_CLASSES[baseline]} | {primary['geomean_ratio']:.4f} | {primary['wins']}/{primary['ties']}/{primary['losses']} | {oracle_text} | {primary['eligible_cases']} |")
            else:
                lines.append(f'| {ARM_LABELS[baseline]} | {ARM_CLASSES[baseline]} | incomplete | — | incomplete | 0 |')
        lines += ['', 'Ratios below 1 favour SPECTRA. The fixed SCC column is the predeclared public diagnostic. The oracle column chooses among the remaining fixed SPECTRA modes after seeing each case and is not deployable.', '']
    lines += ['## Interpretation rules', '',
              '- The primary same-contract threat is `exact-raw`; SCC and route must beat it without selecting raw after the fact.',
              '- CaDiCaL 3 and MiniCard retain learned state across queries. CaDiCaL freezes every original variable so arbitrary later restrictions remain legal.',
              '- Kissat has no incremental-assumption API and is rebuilt for every query; it belongs in a separate one-shot/rebuild class.',
              '- OR-Tools retains the model but starts a fresh single-worker CP-SAT search per query.',
              '- Generic UNSAT decisions are independently cross-checked for truth but do not carry SPECTRA portable proofs.',
              '- Any timeout, crash, unavailable backend or loss remains in the audit and cannot be dropped from a chart.', '']
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--out', type=Path, required=True)
    prep.add_argument('--quotient-library', type=Path, required=True)
    prep.add_argument('--exact-library', type=Path, required=True)
    prep.add_argument('--cadical3-library', type=Path, required=True)
    prep.add_argument('--kissat4-library', type=Path, required=True)
    run = sub.add_parser('run'); run.add_argument('--out', type=Path, required=True)
    audit = sub.add_parser('report'); audit.add_argument('--out', type=Path, required=True); audit.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.out, quotient_library=args.quotient_library,
                exact_library=args.exact_library,
                cadical3_library=args.cadical3_library,
                kissat4_library=args.kissat4_library)
    elif args.command == 'run':
        run_study(args.out/'study')
    else:
        result = report(args.out/'study', args.report)
        print(json.dumps({k: v for k, v in result.items() if k != 'counts'}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
