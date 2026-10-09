from __future__ import annotations
from itertools import combinations, product
from pathlib import Path
import hashlib
import json
import os
import random
from dataclasses import replace

import pytest

from experiments.critical_clique_coloring.core import Graph
from experiments.wap_support.workload import (
    binary_colors, choice_literal, compact_binary_clauses,
    deterministic_dsatur, model_to_labels, restrictions_to_assumptions,
    rotating_triple_masks, verify_labels, WorkloadCase, _sha_json,
)
from experiments.wap_support.controls import ARMS, run_session
from spectra.cnf.quotient_query import QuotientRuntime, build_quotient_runtime


def graph(n, edges):
    adjacency=[set() for _ in range(n)]
    for a,b in edges:adjacency[a].add(b);adjacency[b].add(a)
    return Graph("synthetic",Path("synthetic"),n,tuple(edges),
                 tuple(frozenset(x) for x in adjacency),"0"*64)


def sat_formula(n, clauses, assumptions=()):
    for values in product((False,True), repeat=n):
        if any(values[abs(x)-1] != (x>0) for x in assumptions):continue
        if all(any(values[abs(x)-1] == (x>0) for x in clause) for clause in clauses):
            yield values


def test_all_five_vertex_graphs_preserve_a_proper_dsatur_model():
    bank=tuple(combinations(range(5),2))
    for mask in range(1<<len(bank)):
        edges=tuple(edge for i,edge in enumerate(bank) if mask>>i&1)
        g=graph(5,edges);colors=deterministic_dsatur(g);k,masks=rotating_triple_masks(colors)
        labels=bytes(colors)
        assert verify_labels(g,k,masks,(),labels)
        clauses=compact_binary_clauses(g,masks)
        assumptions=restrictions_to_assumptions(tuple((v,1<<colors[v]) for v in range(5)),masks)
        assert any(sat_formula(5,clauses,assumptions))


def test_compact_formula_matches_original_list_relation_randomly():
    rng=random.Random(441901)
    for _ in range(500):
        n=rng.randrange(1,8)
        edges=tuple(e for e in combinations(range(n),2) if rng.randrange(3)==0)
        g=graph(n,edges);colors=deterministic_dsatur(g);k,masks=rotating_triple_masks(colors)
        clauses=compact_binary_clauses(g,masks)
        formula_models=set()
        for values in sat_formula(n,clauses):
            signed=[i+1 if value else -(i+1) for i,value in enumerate(values)]
            labels=model_to_labels(signed,masks)
            assert verify_labels(g,k,masks,(),labels)
            formula_models.add(labels)
        brute=set()
        choices=[binary_colors(mask) for mask in masks]
        for labels in product(*choices):
            b=bytes(labels)
            if verify_labels(g,k,masks,(),b):brute.add(b)
        assert formula_models==brute


def test_rotating_triples_expose_more_than_same_list_parity():
    g=graph(3,((0,1),(1,2),(0,2)))
    colors=(0,1,2);k,masks=rotating_triple_masks(colors)
    assert len(set(masks))==3 and verify_labels(g,k,masks,(),bytes(colors))
    assert {binary_colors(m) for m in masks}=={(0,1),(1,2),(0,2)}

@pytest.mark.parametrize("bad",[0,1,3,7,-1,True,1.0])
def test_binary_domain_contract_rejects_nonpairs(bad):
    if bad==3:
        assert binary_colors(bad)==(0,1)
    else:
        with pytest.raises(ValueError):binary_colors(bad)


def test_assumption_mapping_is_exact():
    masks=(3,6,5)
    query=((0,1),(1,4),(2,1))
    assert restrictions_to_assumptions(query,masks)==[-1,2,-3]
    assert choice_literal(0,1,masks)==1 and choice_literal(0,0,masks)==-1


@pytest.fixture(scope="session")
def support_runtime(tmp_path_factory):
    override = os.environ.get("SPECTRA_QUOTIENT_LIBRARY")
    path = (Path(override).resolve() if override
            else build_quotient_runtime(tmp_path_factory.mktemp("wap-support-native")))
    return QuotientRuntime(path)


def _small_case():
    queries = tuple(
        tuple((vertex, 1 << ((index >> vertex) & 1)) for vertex in range(8))
        for index in range(8)
    )
    provisional = {
        "schema": "spectra.wap_support.case.v1",
        "graph_name": "small.col",
        "graph_sha256": "0" * 64,
        "graph_git_blob_sha1": "0" * 40,
        "n": 8,
        "edges": (),
        "dsatur_coloring": (0,) * 8,
        "dsatur_colors": 1,
        "palette": 3,
        "masks": (3,) * 8,
        "queries": queries,
        "generator_attempts": 8,
        "generator_rejections": 0,
        "generator_unique_model_hashes": 8,
        "generator_seed": 0,
        "query_width": 8,
        "query_count": 8,
        "clauses": 0,
    }
    case = WorkloadCase(**provisional, case_sha256=_sha_json(provisional))
    case.validate()
    return case


def test_every_complete_session_arm_returns_full_original_valid_outputs(support_runtime):
    case = _small_case()
    g = graph(case.n, case.edges)
    for arm in ARMS:
        result = run_session(case, support_runtime, arm, order_name="shuffle", order_seed=71)
        assert result.status == "COMPLETE"
        assert len(result.output_bytes) == case.n * case.query_count
        assert result.output_sha256 == hashlib.sha256(result.output_bytes).hexdigest()
        for position, query_index in enumerate(
                __import__("experiments.wap_support.controls", fromlist=["order_indices"]).order_indices(
                    case.query_count, "shuffle", 71)):
            labels = result.output_bytes[position * case.n:(position + 1) * case.n]
            assert verify_labels(g, case.palette, case.masks, case.queries[query_index], labels)


def test_case_validation_rejects_noncanonical_edges_and_bad_lists():
    case = _small_case()
    with pytest.raises(ValueError, match="canonical"):
        replace(case, edges=((1, 0),), case_sha256="0" * 64).validate()
    with pytest.raises(ValueError, match="binary list"):
        replace(case, masks=(7,) + case.masks[1:], case_sha256="0" * 64).validate()


def test_case_validation_rejects_unsorted_or_out_of_list_query():
    case = _small_case()
    queries = list(case.queries)
    queries[0] = tuple(reversed(queries[0]))
    provisional = case.canonical_without_hash()
    provisional["queries"] = tuple(queries)
    bad = WorkloadCase(**provisional, case_sha256=_sha_json(provisional))
    with pytest.raises(ValueError, match="query restriction"):
        bad.validate()


def test_matrix_preserves_timeout_as_hashed_negative_evidence(tmp_path, monkeypatch):
    import subprocess
    from experiments.wap_support import run as runner

    cases = tmp_path / "cases"
    cases.mkdir()
    _small_case().to_json(cases / "small.json")
    runtime = tmp_path / "runtime.so"
    runtime.write_bytes(b"runtime")
    output = tmp_path / "evidence"
    monkeypatch.setattr(runner, "schedule",
                        lambda _cases: [("small.json", "scc", "forward", 17)])
    monkeypatch.setattr(runner, "_machine", lambda: {
        "platform": "test", "python": "test", "cpu_model": "test",
        "affinity": [0], "pid": 1, "address_space_limit_bytes": runner.ADDRESS_SPACE_BYTES,
        "threads": 1, "vmrss_kib": 1, "vmhwm_kib": 1,
    })

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["worker"], runner.SESSION_DEADLINE_SECONDS,
                                        output="partial out", stderr="partial err")

    monkeypatch.setattr(runner.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="timed out"):
        runner.run_matrix(cases, runtime, output)
    failure = json.loads((output / "sessions/0000-small-scc-forward/failure.json").read_text())
    assert failure["status"] == "TIMEOUT"
    row = json.loads((output / "sessions.jsonl").read_text())
    assert row["status"] == "TIMEOUT"
    assert json.loads((output / "progress.json").read_text())["complete"] is False
    assert (output / "MANIFEST.json").is_file()
    analyse_manifest = __import__("experiments.wap_support.analyse",
                                  fromlist=["validate_manifest"]).validate_manifest
    analyse_manifest(output)
