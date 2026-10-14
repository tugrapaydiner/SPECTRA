"""Common-mask projection for fair development comparisons, with no SAT filtering.

Some old controls reject unused palette bits instead of intersecting them away.
The wide-mask robustness bank is retained separately. This projection changes
only those irrelevant bits; every original effective domain is exactly preserved.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .cases import load_case, write_case


def project_queries(problem, queries):
    full = (1 << problem.palette) - 1
    return tuple(tuple((v, mask & full) for v, mask in problem.freeze_query(query))
                 for query in queries)


def export_common(source: Path, output: Path):
    files = sorted(p for p in source.glob('*.json') if not p.name.endswith('.generation.json'))
    if not files:
        raise ValueError('no source cases')
    output.mkdir(parents=True, exist_ok=False)
    for path in files:
        case, original = load_case(path)
        queries = project_queries(case.problem, case.queries)
        for old, new in zip(case.queries, queries):
            assert len(old) == len(new)
            for (u, a), (v, b) in zip(old, new):
                assert u == v and (a & case.problem.masks[u]) == (b & case.problem.masks[v])
        case = replace(case, name=case.name+'-common-mask', queries=queries,
                       provenance=case.provenance + '; common-mask projection only; '
                       'no query rejection or answer access; source_case_sha256=' + original)
        write_case(output / path.name, case)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export_common(args.source, args.output)
