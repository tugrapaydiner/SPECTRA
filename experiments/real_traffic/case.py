"""Build outcome-blind repeated-support cases from measured Abilene traffic."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

from experiments.real_traffic.plan import align_target_plan, color_graph
from experiments.real_traffic.observer import (
    masks_from_plans,
    solve_binary,
    verify_coloring_edges,
    verify_labels,
)
from experiments.real_traffic.sources import (
    MATRIX_WIDTH,
    SNAPSHOTS_PER_WEEK,
    ConflictGraph,
    TrafficWeek,
    TraceFormatError,
    build_conflict_graph,
    hash_json,
    mean_traffic,
    parse_demands,
    parse_links,
    parse_routes,
    parse_week,
)


@dataclass(frozen=True)
class TrafficCase:
    schema: str
    source_week: str
    source_sha256: dict[str, str]
    vertices: int
    edges: tuple[tuple[int, int], ...]
    demand_indices: tuple[int, ...]
    demand_names: tuple[str, ...]
    route_links: tuple[tuple[int, ...], ...]
    palette: int
    plan_a: tuple[int, ...]
    plan_b: tuple[int, ...]
    masks: tuple[int, ...]
    queries: tuple[tuple[tuple[int, int], ...], ...]
    timestamps: tuple[int, ...]
    statuses: tuple[str, ...]
    witness_sha256: tuple[str | None, ...]
    proposal_attempts: int
    duplicate_queries: int
    sat_queries: int
    unsat_queries: int
    unique_witnesses: int
    protect_count: int
    migrate_count: int
    case_sha256: str

    def canonical_without_hash(self) -> dict:
        data = asdict(self)
        data.pop("case_sha256")
        return data

    def validate(self) -> None:
        if self.schema != "spectra.real_traffic.case.v1":
            raise TraceFormatError("unsupported case schema")
        if self.vertices != len(self.demand_indices) or self.vertices != len(self.masks):
            raise TraceFormatError("case geometry differs")
        if self.vertices != len(self.plan_a) or self.vertices != len(self.plan_b):
            raise TraceFormatError("plan geometry differs")
        if self.palette < 1 or self.palette > 64:
            raise TraceFormatError("palette outside native contract")
        if not (len(self.queries) == len(self.timestamps) == len(self.statuses)
                == len(self.witness_sha256)):
            raise TraceFormatError("query records differ")
        if len(set(self.queries)) != len(self.queries):
            raise TraceFormatError("duplicate accepted query")
        if any(mask <= 0 or mask >> self.palette or mask.bit_count() > 2
               for mask in self.masks):
            raise TraceFormatError("invalid binary list")
        if any(not (self.masks[v] >> self.plan_a[v] & 1)
               for v in range(self.vertices)):
            raise TraceFormatError("plan A outside lists")
        if any(not (self.masks[v] >> self.plan_b[v] & 1)
               for v in range(self.vertices)):
            raise TraceFormatError("plan B outside lists")
        if not verify_labels(self.edges, self.masks, self.plan_a):
            raise TraceFormatError("plan A is not a valid coloring")
        if not verify_labels(self.edges, self.masks, self.plan_b):
            raise TraceFormatError("plan B is not a valid coloring")

        observed_sat = 0
        observed_unsat = 0
        observed_witnesses: set[str] = set()
        for query, status, witness_digest in zip(
                self.queries, self.statuses, self.witness_sha256):
            if len(query) != self.protect_count + self.migrate_count:
                raise TraceFormatError("query width differs")
            previous = -1
            for vertex, allowed in query:
                if vertex <= previous or not 0 <= vertex < self.vertices:
                    raise TraceFormatError("query vertices are not canonical")
                if allowed.bit_count() != 1 or not (allowed & self.masks[vertex]):
                    raise TraceFormatError("query mask differs")
                previous = vertex
            labels = solve_binary(self.edges, self.masks, query)
            if labels is None:
                observed_unsat += 1
                if status != "UNSAT" or witness_digest is not None:
                    raise TraceFormatError("UNSAT query receipt differs")
            else:
                observed_sat += 1
                digest = hashlib.sha256(bytes(labels)).hexdigest()
                observed_witnesses.add(digest)
                if status != "SAT" or witness_digest != digest:
                    raise TraceFormatError("SAT query receipt differs")
        if (observed_sat != self.sat_queries or observed_unsat != self.unsat_queries
                or observed_sat + observed_unsat != len(self.queries)
                or len(observed_witnesses) != self.unique_witnesses):
            raise TraceFormatError("query totals differ")
        if self.case_sha256 != hash_json(self.canonical_without_hash()):
            raise TraceFormatError("case digest mismatch")

    def write(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")) + "\n"
        )

    @classmethod
    def read(cls, path: str | Path) -> "TrafficCase":
        raw = json.loads(Path(path).read_text())
        for field in ("edges", "route_links"):
            raw[field] = tuple(tuple(row) for row in raw[field])
        for field in (
            "demand_indices", "demand_names", "plan_a", "plan_b", "masks",
            "timestamps", "statuses", "witness_sha256",
        ):
            raw[field] = tuple(raw[field])
        raw["queries"] = tuple(
            tuple(tuple(item) for item in row) for row in raw["queries"]
        )
        case = cls(**raw)
        case.validate()
        return case


def build_queries(graph: ConflictGraph, week: TrafficWeek, plan_a: Sequence[int],
                  plan_b: Sequence[int], masks: Sequence[int], *, protect_count: int = 4,
                  migrate_count: int = 4, max_offsets: int = 32
                  ) -> tuple[tuple[tuple[tuple[int, int], ...], ...], tuple[int, ...],
                             tuple[str, ...], tuple[str | None, ...], dict]:
    """Let traffic fix each proposal first, then retain either exact outcome."""
    vertices = graph.n
    if not week.snapshots or len(week.snapshots[0]) != vertices:
        raise ValueError("traffic and graph geometry differ")
    if protect_count < 1 or migrate_count < 1 or protect_count + migrate_count > vertices:
        raise ValueError("invalid query width")
    changed = {v for v in range(vertices) if plan_a[v] != plan_b[v]}
    if len(changed) < migrate_count:
        raise TraceFormatError("target plan changes too few routes")

    accepted: list[tuple[tuple[int, int], ...]] = []
    timestamps: list[int] = []
    statuses: list[str] = []
    witness_receipts: list[str | None] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    witnesses: set[str] = set()
    attempts = duplicates = sat = unsat = 0
    previous = week.snapshots[0]
    for timestamp, row in enumerate(week.snapshots[1:], start=1):
        protected = sorted(
            range(vertices), key=lambda v: (row[v], -v), reverse=True
        )[:protect_count]
        protected_set = set(protected)
        migration = sorted(
            (v for v in changed if v not in protected_set),
            key=lambda v: (abs(row[v] - previous[v]), row[v], -v),
            reverse=True,
        )
        chosen = None
        available_offsets = max(1, len(migration) - migrate_count + 1)
        for offset in range(min(max_offsets, available_offsets)):
            moving = migration[offset: offset + migrate_count]
            if len(moving) != migrate_count:
                break
            proposal = [(v, 1 << plan_a[v]) for v in protected]
            proposal.extend((v, 1 << plan_b[v]) for v in moving)
            query = tuple(sorted(proposal))
            attempts += 1
            if query in seen:
                duplicates += 1
                continue
            chosen = query
            break
        previous = row
        if chosen is None:
            continue
        seen.add(chosen)
        labels = solve_binary(graph.edges, masks, chosen)
        accepted.append(chosen)
        timestamps.append(timestamp)
        if labels is None:
            unsat += 1
            statuses.append("UNSAT")
            witness_receipts.append(None)
        else:
            sat += 1
            digest = hashlib.sha256(bytes(labels)).hexdigest()
            witnesses.add(digest)
            statuses.append("SAT")
            witness_receipts.append(digest)
    stats = {
        "proposal_attempts": attempts,
        "duplicate_queries": duplicates,
        "sat_queries": sat,
        "unsat_queries": unsat,
        "unique_witnesses": len(witnesses),
    }
    return (
        tuple(accepted), tuple(timestamps), tuple(statuses),
        tuple(witness_receipts), stats,
    )


def build_case(*, week_name: str, routing_bytes: bytes, demands_bytes: bytes,
               links_bytes: bytes, traffic_bytes: bytes, protect_count: int = 4,
               migrate_count: int = 4, expected_demand_count: int = MATRIX_WIDTH,
               expected_snapshots: int = SNAPSHOTS_PER_WEEK) -> TrafficCase:
    demands = parse_demands(
        demands_bytes.decode("ascii"), expected_count=expected_demand_count
    )
    links = parse_links(links_bytes.decode("ascii"))
    routes = parse_routes(routing_bytes.decode("ascii"), demands, links)
    graph = build_conflict_graph(
        demands, routes,
        source_bytes=(routing_bytes, demands_bytes, links_bytes),
    )
    week = parse_week(
        traffic_bytes, demands, expected_snapshots=expected_snapshots
    )
    weights = mean_traffic(week)
    plan_a = color_graph(graph.adjacency)
    raw_target = color_graph(graph.adjacency, weights)
    plan_b = align_target_plan(plan_a, raw_target, weights)
    if not verify_coloring_edges(graph.edges, plan_b):
        raise AssertionError("aligned target plan is invalid")
    masks = masks_from_plans(plan_a, plan_b)
    palette = max(max(plan_a), max(plan_b)) + 1
    if palette > 64:
        raise TraceFormatError("traffic plans exceed the native 64-color contract")
    queries, timestamps, statuses, witness_sha256, stats = build_queries(
        graph, week, plan_a, plan_b, masks,
        protect_count=protect_count, migrate_count=migrate_count,
    )
    provisional = {
        "schema": "spectra.real_traffic.case.v1",
        "source_week": week_name,
        "source_sha256": {
            "routing": hashlib.sha256(routing_bytes).hexdigest(),
            "demands": hashlib.sha256(demands_bytes).hexdigest(),
            "links": hashlib.sha256(links_bytes).hexdigest(),
            "traffic": week.source_sha256,
            "route_graph": graph.source_sha256,
        },
        "vertices": graph.n,
        "edges": graph.edges,
        "demand_indices": tuple(d.index for d in graph.demands),
        "demand_names": tuple(f"{d.source},{d.target}" for d in graph.demands),
        "route_links": graph.routes,
        "palette": palette,
        "plan_a": plan_a,
        "plan_b": plan_b,
        "masks": masks,
        "queries": queries,
        "timestamps": timestamps,
        "statuses": statuses,
        "witness_sha256": witness_sha256,
        **stats,
        "protect_count": protect_count,
        "migrate_count": migrate_count,
    }
    case = TrafficCase(**provisional, case_sha256=hash_json(provisional))
    case.validate()
    return case
