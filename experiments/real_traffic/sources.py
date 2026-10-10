"""Strict readers for the public Abilene route and traffic-matrix files."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

SOURCE_ROOT = "https://www.cs.utexas.edu/~yzhang/research/AbileneTM"
SOURCE_URLS = {
    "routing": f"{SOURCE_ROOT}/A",
    "demands": f"{SOURCE_ROOT}/demands",
    "links": f"{SOURCE_ROOT}/links",
    "readme": f"{SOURCE_ROOT}/readme.txt",
}
SNAPSHOTS_PER_WEEK = 12 * 24 * 7
MATRIX_WIDTH = 144
FIELDS_PER_DEMAND = 5


class TraceFormatError(ValueError):
    """A downloaded source violates the disclosed data contract."""


def traffic_url(week: str) -> str:
    if (type(week) is not str or len(week) != 3 or week[0] != "X"
            or not week[1:].isdigit() or not 1 <= int(week[1:]) <= 24):
        raise ValueError("week must be X01 through X24")
    return f"{SOURCE_ROOT}/{week}.gz"


def hash_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class Demand:
    index: int
    source: str
    target: str


@dataclass(frozen=True)
class Link:
    index: int
    source: str
    target: str
    kind: int


@dataclass(frozen=True)
class ConflictGraph:
    demands: tuple[Demand, ...]
    routes: tuple[tuple[int, ...], ...]
    edges: tuple[tuple[int, int], ...]
    adjacency: tuple[frozenset[int], ...]
    source_sha256: str

    @property
    def n(self) -> int:
        return len(self.demands)


@dataclass(frozen=True)
class TrafficWeek:
    snapshots: tuple[tuple[Decimal, ...], ...]
    source_sha256: str

    @property
    def count(self) -> int:
        return len(self.snapshots)


def _records(text: str) -> Iterator[tuple[int, str]]:
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if line and not line.startswith("#"):
            yield lineno, line


def parse_demands(text: str, *, expected_count: int = MATRIX_WIDTH) -> tuple[Demand, ...]:
    if type(expected_count) is not int or expected_count < 1:
        raise ValueError("expected_count must be a positive exact integer")
    output: list[Demand] = []
    for lineno, line in _records(text):
        fields = line.split()
        if len(fields) != 2 or "," not in fields[0]:
            raise TraceFormatError(f"bad demand row at {lineno}")
        source, target = fields[0].split(",", 1)
        try:
            index = int(fields[1])
        except ValueError as exc:
            raise TraceFormatError(f"bad demand index at {lineno}") from exc
        if index != len(output) + 1 or not source or not target:
            raise TraceFormatError(f"noncanonical demand row at {lineno}")
        output.append(Demand(index, source, target))
    if len(output) != expected_count:
        raise TraceFormatError(f"expected {expected_count} demands, received {len(output)}")
    if len({(d.source, d.target) for d in output}) != len(output):
        raise TraceFormatError("duplicate demand")
    return tuple(output)


def parse_links(text: str) -> tuple[Link, ...]:
    output: list[Link] = []
    for lineno, line in _records(text):
        fields = line.split()
        if len(fields) != 3 or "," not in fields[0]:
            raise TraceFormatError(f"bad link row at {lineno}")
        source, target = fields[0].split(",", 1)
        try:
            index, kind = int(fields[1]), int(fields[2])
        except ValueError as exc:
            raise TraceFormatError(f"bad link numbers at {lineno}") from exc
        if index != len(output) + 1 or kind not in (0, 1, 2):
            raise TraceFormatError(f"noncanonical link row at {lineno}")
        if ("*" in (source, target)) != (kind != 0):
            raise TraceFormatError(f"link type differs at {lineno}")
        output.append(Link(index, source, target, kind))
    if not output or len({(x.source, x.target) for x in output}) != len(output):
        raise TraceFormatError("empty or duplicate link bank")
    return tuple(output)


def parse_routes(text: str, demands: Sequence[Demand],
                 links: Sequence[Link]) -> tuple[tuple[int, ...], ...]:
    by_demand = {d.index: d for d in demands}
    by_link = {link.index: link for link in links}
    routes: list[set[int]] = [set() for _ in demands]
    seen: set[tuple[int, int]] = set()
    for lineno, line in _records(text):
        fields = line.split()
        if len(fields) != 5:
            raise TraceFormatError(f"bad routing row at {lineno}")
        link_name, demand_name = fields[:2]
        try:
            link_index, demand_index = int(fields[2]), int(fields[3])
            fraction = Decimal(fields[4])
        except (ValueError, InvalidOperation) as exc:
            raise TraceFormatError(f"bad routing value at {lineno}") from exc
        if not fraction.is_finite() or fraction <= 0 or fraction > 1:
            raise TraceFormatError(f"routing fraction outside (0,1] at {lineno}")
        if link_index not in by_link or demand_index not in by_demand:
            raise TraceFormatError(f"routing index outside inventory at {lineno}")
        link, demand = by_link[link_index], by_demand[demand_index]
        if link_name != f"{link.source},{link.target}":
            raise TraceFormatError(f"routing link name differs at {lineno}")
        if demand_name != f"{demand.source},{demand.target}":
            raise TraceFormatError(f"routing demand name differs at {lineno}")
        key = (link_index, demand_index)
        if key in seen:
            raise TraceFormatError(f"duplicate routing row at {lineno}")
        seen.add(key)
        if link.kind == 0:
            routes[demand_index - 1].add(link_index)
    for demand, route in zip(demands, routes):
        if demand.source == demand.target:
            if route:
                raise TraceFormatError("loop demand uses an internal link")
        elif not route:
            raise TraceFormatError(f"demand {demand.index} has no internal route")
    return tuple(tuple(sorted(route)) for route in routes)


def build_conflict_graph(demands: Sequence[Demand], routes: Sequence[Sequence[int]],
                         *, source_bytes: Iterable[bytes] = ()) -> ConflictGraph:
    if len(demands) != len(routes):
        raise TraceFormatError("route inventory differs from demands")
    keep = [i for i, demand in enumerate(demands) if demand.source != demand.target]
    selected_demands = tuple(demands[i] for i in keep)
    selected_routes = tuple(tuple(routes[i]) for i in keep)
    if any(not route or len(set(route)) != len(route) for route in selected_routes):
        raise TraceFormatError("non-loop route is empty or repeated")
    edges: list[tuple[int, int]] = []
    adjacency = [set() for _ in keep]
    route_sets = [set(route) for route in selected_routes]
    for left in range(len(keep)):
        for right in range(left + 1, len(keep)):
            if route_sets[left].intersection(route_sets[right]):
                edges.append((left, right))
                adjacency[left].add(right)
                adjacency[right].add(left)
    digest = hashlib.sha256()
    for block in source_bytes:
        digest.update(len(block).to_bytes(8, "big"))
        digest.update(block)
    return ConflictGraph(
        selected_demands,
        selected_routes,
        tuple(edges),
        tuple(frozenset(row) for row in adjacency),
        digest.hexdigest(),
    )


def parse_week(data: bytes, demands: Sequence[Demand], *,
               expected_snapshots: int = SNAPSHOTS_PER_WEEK,
               fields_per_demand: int = FIELDS_PER_DEMAND) -> TrafficWeek:
    if not demands:
        raise TraceFormatError("traffic parser requires a demand inventory")
    if type(expected_snapshots) is not int or expected_snapshots < 1:
        raise ValueError("expected_snapshots must be a positive exact integer")
    if type(fields_per_demand) is not int or fields_per_demand < 1:
        raise ValueError("fields_per_demand must be a positive exact integer")
    digest = hashlib.sha256(data).hexdigest()
    keep = [d.index - 1 for d in demands if d.source != d.target]
    snapshots: list[tuple[Decimal, ...]] = []
    try:
        stream = gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb")
        text = io.TextIOWrapper(stream, encoding="ascii", errors="strict", newline="")
        for lineno, raw in enumerate(text, start=1):
            fields = raw.split()
            expected_fields = len(demands) * fields_per_demand
            if len(fields) != expected_fields:
                raise TraceFormatError(f"traffic row {lineno} has {len(fields)} fields")
            real: list[Decimal] = []
            for index in keep:
                try:
                    value = Decimal(fields[fields_per_demand * index])
                except InvalidOperation as exc:
                    raise TraceFormatError(f"bad traffic value at row {lineno}") from exc
                if not value.is_finite() or value < 0:
                    raise TraceFormatError(f"nonfinite or negative traffic at row {lineno}")
                # Source unit is 100 bytes per five minutes; retain exact Mbit/s.
                real.append(value * Decimal(8) / Decimal(3_000_000))
            snapshots.append(tuple(real))
    except (OSError, EOFError, UnicodeError) as exc:
        raise TraceFormatError("invalid or truncated gzip source") from exc
    if len(snapshots) != expected_snapshots:
        raise TraceFormatError(f"expected {expected_snapshots} snapshots, received {len(snapshots)}")
    return TrafficWeek(tuple(snapshots), digest)


def mean_traffic(week: TrafficWeek) -> tuple[Decimal, ...]:
    if not week.snapshots:
        raise ValueError("empty traffic week")
    width = len(week.snapshots[0])
    totals = [Decimal(0)] * width
    for row in week.snapshots:
        if len(row) != width:
            raise TraceFormatError("ragged traffic week")
        for index, value in enumerate(row):
            totals[index] += value
    scale = Decimal(len(week.snapshots))
    return tuple(value / scale for value in totals)
