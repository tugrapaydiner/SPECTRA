#!/usr/bin/env python3
"""Audit WAP graph identities, optimum witnesses, and critical-clique certificates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from core import build_critical_clique_quotient, parse_dimacs_col, verify_clique


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    metadata_bytes = args.metadata.read_bytes()
    inventory_bytes = args.inventory.read_bytes()
    metadata = json.loads(metadata_bytes)
    inventory = json.loads(inventory_bytes)
    rows = []
    for name, optimum in sorted(metadata["instances"].items()):
        graph_path = args.data_dir / name
        raw = graph_path.read_bytes()
        record = inventory["graphs"][name]
        if len(raw) != int(record["bytes"]):
            raise AssertionError(f"byte count mismatch for {name}")
        if git_blob_sha1(raw) != record["git_blob_sha1"]:
            raise AssertionError(f"Git blob identity mismatch for {name}")
        graph = parse_dimacs_col(graph_path, name=name)
        color_count = int(optimum["chromatic_number"])
        clique = tuple(int(vertex) - 1 for vertex in optimum["maximum_clique_1based"])
        verify_clique(graph, clique, expected_size=color_count)
        quotient = build_critical_clique_quotient(graph)
        clique_set = set(clique)
        cut_classes = []
        for vertex in clique:
            members = set(quotient.classes[quotient.class_of[vertex]])
            if not members <= clique_set:
                cut_classes.append(quotient.class_of[vertex])
        if cut_classes:
            raise AssertionError(f"maximum clique cuts critical cliques in {name}: {sorted(set(cut_classes))}")
        rows.append(
            {
                "name": name,
                "role": record["role"],
                "graph_sha256": graph.sha256,
                "graph_git_blob_sha1": git_blob_sha1(raw),
                "vertices": graph.n,
                "edges": graph.m,
                "chromatic_number": color_count,
                "maximum_clique_size": len(clique),
                "quotient_vertices": quotient.n,
                "quotient_edges": quotient.m,
                "vertices_removed": quotient.vertices_removed,
                "nontrivial_classes": sum(weight > 1 for weight in quotient.weights),
                "largest_class": max(quotient.weights),
                "critical_clique_certificate_sha256": quotient.certificate_sha256,
            }
        )

    result = {
        "schema": "spectra.wap.critical_clique_audit.v1",
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "instances": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
