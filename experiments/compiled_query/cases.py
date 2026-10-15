"""Answer-free, immutable development inputs; never a new WAP holdout."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from .binary_list import BinaryListProblem

SCHEMA = 'spectra.compiled_query.case.v1'
MAX_INPUT_BYTES = 128 << 20


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       allow_nan=False) + '\n').encode('utf-8')


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


@dataclass(frozen=True)
class Case:
    name: str
    cluster: str
    provenance: str
    problem: BinaryListProblem
    queries: tuple[tuple[tuple[int, int], ...], ...]

    def payload(self) -> bytes:
        p = self.problem
        return canonical(dict(schema=SCHEMA, name=self.name, cluster=self.cluster,
                              provenance=self.provenance, n=p.n, palette=p.palette,
                              edges=p.edges, masks=p.masks, queries=self.queries))

    @classmethod
    def decode(cls, payload: bytes) -> 'Case':
        if type(payload) is not bytes or len(payload) > MAX_INPUT_BYTES:
            raise ValueError('case must be bytes within the input limit')
        value = json.loads(payload, object_pairs_hook=_unique_object,
                           parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        fields = {'schema', 'name', 'cluster', 'provenance', 'n', 'palette',
                  'edges', 'masks', 'queries'}
        if type(value) is not dict or set(value) != fields or value['schema'] != SCHEMA:
            raise ValueError('case schema/field inventory differs')
        for key in ('name', 'cluster', 'provenance'):
            if type(value[key]) is not str or not value[key] or len(value[key]) > 4096:
                raise ValueError('invalid case ' + key)
        p = BinaryListProblem.create(value['n'], value['palette'],
                                     value['edges'], value['masks'])
        if type(value['queries']) is not list or not 1 <= len(value['queries']) <= 1_000_000:
            raise ValueError('invalid query bank')
        queries = tuple(p.freeze_query(q) for q in value['queries'])
        return cls(value['name'], value['cluster'], value['provenance'], p, queries)


def load_case(path: Path) -> tuple[Case, str]:
    with Path(path).open('rb') as stream:
        payload = stream.read(MAX_INPUT_BYTES + 1)
    return Case.decode(payload), digest(payload)


def write_case(path: Path, case: Case) -> None:
    payload = case.payload()
    Case.decode(payload)
    with Path(path).open('xb') as stream:
        stream.write(payload)


def synthetic_cases(count: int = 128, n: int = 96, seed: int = 20261013) -> tuple[Case, ...]:
    """Disclosed constructions; queries are drawn without a solver/rejection loop.

    Opposite restrictions and infeasible requests are intentionally retained.
    This function never imports or calls any solver.
    """
    if type(count) is not int or not 1 <= count <= 1_000_000:
        raise ValueError('count must be a positive bounded integer')
    if type(n) is not int or not 4 <= n <= 10000:
        raise ValueError('n must be an integer in [4, 10000]')
    result = []
    for index, name in enumerate(('isolated', 'pairs', 'path', 'mixed_path', 'dense_bipartite')):
        if name == 'isolated':
            edges = ()
        elif name == 'pairs':
            edges = tuple((i, i + 1) for i in range(0, n - 1, 2))
        elif name in ('path', 'mixed_path'):
            edges = tuple((i, i + 1) for i in range(n - 1))
        else:
            edges = tuple((i, j) for i in range(n // 2) for j in range(n // 2, n))
        masks = tuple((3, 6, 5)[i % 3] if name == 'mixed_path' else 3 for i in range(n))
        p = BinaryListProblem.create(n, 3, edges, masks)
        choices = p.colors()
        rng = random.Random(seed + 1009 * index)
        queries = [()]
        for q in range(1, count):
            width = 1 if q % 3 else min(8, n)
            restrictions = [(v, 1 << rng.choice(choices[v]))
                            for v in rng.sample(range(n), width)]
            if q % 17 == 0:
                v, allowed = restrictions[0]
                restrictions.append((v, masks[v] ^ allowed))
            queries.append(tuple(restrictions))
        result.append(Case(f'{name}-{n}', f'synthetic:{name}:{n}:{seed}',
                           f'synthetic construction; seed={seed}; no query filtering',
                           p, tuple(queries)))
    return tuple(result)


def export_wap(graph_dir: Path, destination: Path) -> None:
    """Only the previously designated WAP05/06 development graphs are processed.

    The historical generator uses native SAT to retain feasible restrictions.
    That selection is disclosed; no DSATUR/model answer is exported.
    This is development reuse, NOT another prospective confirmation.
    """
    from experiments.wap_support.workload import build_case
    root = Path(__file__).resolve().parents[2]
    inventory_path = root / 'experiments/quotient_application/WAP_UPSTREAM_INVENTORY.json'
    inventory = json.loads(inventory_path.read_text())
    destination.mkdir(parents=True, exist_ok=False)
    for name in ('wap05a.col', 'wap06a.col'):
        source = Path(graph_dir) / name
        record = inventory['graphs'][name]
        if record['role'] not in ('development', 'development_consumed'):
            raise ValueError('refusing to promote another graph to development implicitly')
        old, receipt = build_case(source, graph_name=name,
                                 expected_git_blob_sha1=record['git_blob_sha1'])
        p = BinaryListProblem.create(old.n, old.palette, old.edges, old.masks)
        provenance = ('exposed WAP development; real conflict topology; synthetic lists; '
                      'SAT-accepted restrictions from historical generator; upstream='
                      + inventory['commit'] + '; graph_sha256=' + old.graph_sha256)
        case = Case(name, 'wap:' + old.graph_sha256, provenance, p, old.queries)
        write_case(destination / (name + '.json'), case)
        (destination / (name + '.generation.json')).write_bytes(canonical(receipt))
