"""Outcome-blind request banks on ALREADY EXPOSED inputs; not new holdouts."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import random

from .cases import load_case, write_case


def unfiltered_requests(problem, count: int = 1024, seed: int = 20261013):
    """Keep every generated row; no answer, solver, rejection loop, or repair.

    Includes variable pin widths, contradictory duplicates, empty domains, and
    masks containing colour 63 regardless of whether that colour is available.
    The old generated graph/lists are reused; their construction is not erased.
    """
    if type(count) is not int or not 1 <= count <= 1_000_000:
        raise ValueError('count outside [1, 1000000]')
    if type(seed) is not int or not 0 <= seed < 1 << 64:
        raise ValueError('seed must be uint64')
    rng = random.Random(seed)
    choices = [tuple(c for c in range(problem.palette) if d >> c & 1) for d in problem.masks]
    queries = [()]
    for index in range(1, count):
        width = min(problem.n, (1, 2, 4, 8, 16)[index % 5])
        row = [(v, 1 << rng.choice(choices[v])) for v in rng.sample(range(problem.n), width)]
        if row and index % 17 == 0:
            vertex, mask = row[0]
            row.append((vertex, problem.masks[vertex] ^ mask))
        if row and index % 19 == 0:
            row[0] = (row[0][0], problem.masks[row[0][0]])
        if row and index % 23 == 0:
            row.append((row[-1][0], 0))
        if row and index % 29 == 0:
            row[0] = (row[0][0], row[0][1] | (1 << 63))
        queries.append(tuple(row))
    return tuple(queries)


def export_unfiltered(source: Path, output: Path, count: int = 1024, seed: int = 20261013):
    paths = sorted(p for p in source.glob('*.json') if not p.name.endswith('.generation.json'))
    if not paths:
        raise ValueError('no previously exposed source cases')
    output.mkdir(parents=True, exist_ok=False)
    for index, path in enumerate(paths):
        case, original_hash = load_case(path)
        queries = unfiltered_requests(case.problem, count, seed+1009*index)
        case = replace(case, name=case.name+'-unfiltered', queries=queries,
                       provenance=(case.provenance + '; NEW unfiltered request bank replaces old requests; '
                                   f'no SAT rejection; seed={seed+1009*index}; source_case_sha256={original_hash}'))
        write_case(output/path.name, case)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--queries', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=20261013)
    args = parser.parse_args()
    export_unfiltered(args.source, args.output, args.queries, args.seed)
