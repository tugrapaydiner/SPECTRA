"""Development mixed-result transfer cases after the first admission failure.

Graph identities remain explicit. Case construction is answer-free: a deterministic
witness proves every SAT row and a trapped retained edge proves every UNSAT row.
No solver answer or benchmark timing is consulted.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import random

from experiments.critical_clique_coloring.core import parse_dimacs_col

from .binary_list import BinaryListProblem
from .cases import Case, canonical, digest, load_case, write_case
from .witness_colouring import prepare_witness_topology

SCHEMA = "spectra.fresh_transfer.v2"
UPSTREAM_REPOSITORY = "marijnheule/clicolcom"
UPSTREAM_COMMIT = "4932048642da2144f387961b595112277afff82f"
MASTER_SEED = 0x6D4B1F29A7C3E850
QUERY_COUNT = 1024
EXPECTED = tuple(1 if row % 2 == 0 else 0 for row in range(QUERY_COUNT))
GRAPH_INVENTORY = {
    "DSJC500.5.col": {
        "family": "random-dimacs",
        "bytes": 599069,
        "git_blob_sha1": "5732c76326e835fa9dd61fb5c4251d93d9a34648",
    },
    "flat300_20_0.col": {
        "family": "flat-planted",
        "bytes": 198295,
        "git_blob_sha1": "3e37b08bfe90ed4dd57654ac959ef8c758729b1c",
    },
    "le450_25c.col": {
        "family": "leighton",
        "bytes": 165335,
        "git_blob_sha1": "615254414a3ef5784d0fc130573f2c14cbf7bba0",
    },
    "miles1500.col": {
        "family": "geometric",
        "bytes": 43473,
        "git_blob_sha1": "937ba541922a0e2ef4d1e9bf49cef49271a8e909",
    },
    "queen16_16.col": {
        "family": "queen",
        "bytes": 58010,
        "git_blob_sha1": "d1e1057d1e70fbea9a1318a0bf53c7321ec62b4c",
    },
    "school1.col": {
        "family": "school-scheduling",
        "bytes": 180775,
        "git_blob_sha1": "10c1165916a0edfb5941af2c5c914b1b6199ff28",
    },
}


def _git_blob_sha1(payload: bytes) -> str:
    return hashlib.sha1(
        f"blob {len(payload)}\0".encode("ascii") + payload
    ).hexdigest()


def _seed_for(name: str) -> int:
    suffix = int.from_bytes(
        hashlib.sha256(name.encode("utf-8")).digest()[:8], "big"
    )
    return MASTER_SEED ^ suffix


def _expected_payload(name: str) -> bytes:
    return canonical({
        "schema": SCHEMA,
        "graph": name,
        "query_count": QUERY_COUNT,
        "sat": EXPECTED.count(1),
        "unsat": EXPECTED.count(0),
        "statuses": EXPECTED,
    })


def build_fresh_case(source: Path, destination: Path) -> dict:
    """Build one answer-free development case and return its generation receipt."""
    source = Path(source).resolve(strict=True)
    name = source.name
    if name not in GRAPH_INVENTORY:
        raise ValueError("graph is outside the transfer inventory")
    expected_graph = GRAPH_INVENTORY[name]
    raw = source.read_bytes()
    if len(raw) != expected_graph["bytes"]:
        raise ValueError(f"{name}: byte length differs from frozen metadata")
    blob_sha1 = _git_blob_sha1(raw)
    if blob_sha1 != expected_graph["git_blob_sha1"]:
        raise ValueError(f"{name}: Git-blob identity differs from frozen metadata")

    original = parse_dimacs_col(source, name=name)
    if original.n < 9:
        raise ValueError(f"{name}: fewer than nine vertices")
    if len(original.edges) < QUERY_COUNT // 2:
        raise ValueError(f"{name}: fewer than 512 edges")

    prepared = prepare_witness_topology(original)
    graph = prepared.graph
    colouring = prepared.colouring
    if len(graph.edges) < QUERY_COUNT // 2:
        raise ValueError(f"{name}: prepared topology has fewer than 512 edges")

    colour_count = max(colouring, default=-1) + 1
    if not 1 <= colour_count <= 63:
        raise AssertionError(f"{name}: witness colour count is outside [1, 63]")

    trap_colour = colour_count
    palette = colour_count + 1
    trap_mask = 1 << trap_colour
    masks = tuple((1 << colour) | trap_mask for colour in colouring)
    problem = BinaryListProblem.create(graph.n, palette, graph.edges, masks)

    seed = _seed_for(name)
    rng = random.Random(seed)
    edges = list(graph.edges)
    vertices = list(range(graph.n))
    rng.shuffle(edges)
    rng.shuffle(vertices)
    tag_vertices = tuple(vertices[:9])

    queries = []
    seen = set()
    for row in range(QUERY_COUNT):
        if row == 0:
            query = ()
        elif row % 2 == 0:
            # The 511 nonzero even rows encode their index as a unique nonempty
            # subset of nine witness pins. The retained colouring proves SAT.
            code = row // 2
            query = tuple(sorted(
                (tag_vertices[bit], 1 << colouring[tag_vertices[bit]])
                for bit in range(9)
                if (code >> bit) & 1
            ))
        else:
            # Each odd row chooses a distinct retained edge and pins both
            # endpoints to the shared trap colour. This proves UNSAT without a
            # solver. Index-coded witness pins vary width without changing proof.
            code = row // 2
            left, right = edges[code]
            restrictions = {left: trap_mask, right: trap_mask}
            for bit in range(6):
                if (code >> bit) & 1:
                    vertex = tag_vertices[bit]
                    if vertex not in restrictions:
                        restrictions[vertex] = 1 << colouring[vertex]
            query = tuple(sorted(restrictions.items()))
        if query in seen:
            raise ValueError(f"{name}: duplicate deterministic query at row {row}")
        seen.add(query)
        queries.append(query)

    # Verify the construction directly from disclosed witnesses, without a solver.
    edge_set = set(graph.edges)
    for row, query in enumerate(queries):
        if EXPECTED[row]:
            for vertex, allowed in query:
                if not ((allowed >> colouring[vertex]) & 1):
                    raise AssertionError(f"{name}: SAT row rejects the witness")
        else:
            pinned_trap = {
                vertex for vertex, allowed in query if allowed == trap_mask
            }
            if not any(
                (left, right) in edge_set
                for left in pinned_trap
                for right in pinned_trap
                if left < right
            ):
                raise AssertionError(f"{name}: UNSAT row lacks a trapped edge")

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    topology = prepared.receipt
    provenance = (
        "exposed transfer development; upstream="
        + UPSTREAM_REPOSITORY + "@" + UPSTREAM_COMMIT
        + "; graph_git_blob_sha1=" + blob_sha1
        + "; original_graph_sha256=" + original.sha256
        + "; prepared_topology_sha256=" + graph.sha256
        + "; topology_mode=" + topology["topology_mode"]
        + "; deterministic structural witness portfolio plus one trap colour"
        + "; no solver filtering"
    )
    case = Case(
        name,
        "fresh-transfer-development:"
        + expected_graph["family"] + ":" + graph.sha256,
        provenance,
        problem,
        tuple(queries),
    )
    case_path = destination / "case.json"
    expected_path = destination / "expected.json"
    write_case(case_path, case)
    expected_path.write_bytes(_expected_payload(name))
    loaded, case_sha256 = load_case(case_path)
    if loaded != case:
        raise AssertionError("transfer case did not round-trip")

    receipt = {
        "schema": SCHEMA,
        "classification": "exposed development after failed prospective opening",
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "graph": name,
        "family": expected_graph["family"],
        "graph_bytes": len(raw),
        "graph_git_blob_sha1": blob_sha1,
        "original_graph_sha256": original.sha256,
        "prepared_topology_sha256": graph.sha256,
        "vertices": graph.n,
        "original_edges": len(original.edges),
        "retained_edges": len(graph.edges),
        "dropped_edges": len(original.edges) - len(graph.edges),
        "topology_mode": topology["topology_mode"],
        "degree_cap": topology["degree_cap"],
        "maximum_retained_degree": topology["maximum_retained_degree"],
        "full_graph_best_colours": topology["full_graph_best_colours"],
        "full_graph_best_method": topology["full_graph_best_method"],
        "full_graph_best_trial": topology["full_graph_best_trial"],
        "witness_colours": topology["witness_colours"],
        "witness_method": topology["selected_method"],
        "witness_trial": topology["selected_trial"],
        "witness_restarts": topology["restarts"],
        "witness_sha256": topology["witness_sha256"],
        "palette": palette,
        "trap_colour": trap_colour,
        "generator_seed": seed,
        "tag_vertices": tag_vertices,
        "query_count": QUERY_COUNT,
        "sat": EXPECTED.count(1),
        "unsat": EXPECTED.count(0),
        "expected_sha256": digest(expected_path.read_bytes()),
        "case_sha256": case_sha256,
        "query_bank_sha256": digest(canonical(tuple(queries))),
        "answer_filtering": False,
    }
    (destination / "generation.json").write_bytes(canonical(receipt))
    return receipt


def open_fresh_transfer(graph_directory: Path, destination: Path) -> dict:
    """Build all six exposed development cases with no replacement."""
    graph_directory = Path(graph_directory).resolve(strict=True)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    receipts = {}
    for name in GRAPH_INVENTORY:
        receipts[name] = build_fresh_case(
            graph_directory / name,
            destination / name,
        )
    manifest = {
        "schema": SCHEMA + ".inventory",
        "classification": "exposed development after failed prospective opening",
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "master_seed": MASTER_SEED,
        "query_count": QUERY_COUNT,
        "expected_status_sha256": digest(canonical(EXPECTED)),
        "graphs": receipts,
    }
    (destination / "inventory.json").write_bytes(canonical(manifest))
    return manifest
