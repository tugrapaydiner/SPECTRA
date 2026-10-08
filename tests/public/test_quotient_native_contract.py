"""Native ABI, certificate, checker, and ownership contracts.

These tests target the boundary that the application benchmark depends on.  They
intentionally avoid timing assertions: correctness, immutability, and accounting
geometry are release contracts; performance is evidence produced separately.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os

import pytest

from spectra.cnf.arcfree_support import ArcFreeSupportTable
from spectra.cnf.quotient_query import QuotientRuntime, build_quotient_runtime


@pytest.fixture(scope="session")
def runtime(tmp_path_factory):
    path = os.environ.get("SPECTRA_QUOTIENT_LIBRARY")
    if path is None:
        path = build_quotient_runtime(tmp_path_factory.mktemp("quotient-native-contract"))
    return QuotientRuntime(path)


def _lift_side(certificate: dict, side: int) -> bytes:
    labels = bytearray(certificate["n"])
    for vertex, node in enumerate(certificate["node"]):
        if certificate["wide"][vertex]:
            colour = side
        else:
            colour = (
                certificate["colour0"][vertex]
                if side == 0
                else certificate["colour1"][vertex]
            )
        labels[vertex] = colour
    return bytes(labels)


def test_abi_certificate_and_independent_lifts(runtime):
    edges = ((0, 1), (1, 2), (2, 3))
    masks = (3, 3, 3, 3)
    with runtime.prepare(4, 2, edges, masks=masks, mode="scc") as prepared:
        certificate = prepared.certificate()
        assert certificate["schema"] == "spectra.quotient.certificate.v1"
        assert certificate["n"] == 4 and certificate["k"] == 2
        assert certificate["impossible"] is False
        assert len(certificate["node"]) == 4
        assert len(certificate["colour0"]) == 4
        assert len(certificate["colour1"]) == 4
        assert len(certificate["wide"]) == 4
        assert len(certificate["offsets"]) == sum(
            domain.bit_count() for domain in certificate["palettes"]
        ) + 1
        assert certificate["arcs"] == []
        assert all(value == 0 for value in certificate["offsets"])
        for side in (0, 1):
            answer = _lift_side(certificate, side)
            assert prepared.check(answer)

        # Exported Python containers are copies.  Mutating them cannot alter the
        # prepared native index or its subsequent answers.
        certificate["node"][0] = 10_000
        certificate["initial"][0] = 0
        result = prepared.solve()
        assert result.status == "SAT_VERIFIED"
        assert prepared.check(result.labels)


def test_observer_owns_original_inputs_and_checks_every_constraint(runtime):
    edges = ((0, 1), (1, 2))
    masks = (0b011, 0b110, 0b101)
    checker = runtime.checker(3, 3, edges, masks)
    assert checker.payload_bytes > 0
    valid = bytes((0, 1, 2))
    assert checker.check(valid)
    assert not checker.check(bytes((1, 1, 2)))
    assert not checker.check(bytes((2, 1, 2)))
    assert not checker.check(valid, ((0, 0b010),))
    assert checker.check(valid, ((0, 0b001), (2, 0b100)))
    assert checker.cache_match((bytes((1, 2, 0)), valid), ((2, 0b100),))
    assert not checker.cache_match((bytes((1, 2, 0)),), ((2, 0b100),))
    assert not checker.cache_match((b"bad",), ())
    checker.close()
    with pytest.raises(RuntimeError):
        checker.check(valid)
    with pytest.raises(RuntimeError):
        checker.cache_match((valid,))


def test_one_byte_results_do_not_poison_python_byte_singletons(runtime):
    with ArcFreeSupportTable(runtime, 1, 2, (), masks=(3,)) as table:
        for _ in range(300):
            assert table.solve(((0, 1),)).labels == b"\x00"
            assert table.solve(((0, 2),)).labels == b"\x01"
            assert table.solve(()).labels == b"\x00"
    # These literals were historically vulnerable when the extension wrote into
    # CPython-owned one-byte objects after construction.
    assert bytes([0]) == b"\x00"
    assert bytes([1]) == b"\x01"
    assert tuple(bytes([i]) for i in range(4)) == (b"\x00", b"\x01", b"\x02", b"\x03")


def test_native_support_matches_search_and_is_thread_safe(runtime):
    edges = tuple((i, i + 1) for i in range(15))
    masks = tuple(3 for _ in range(16))
    table = ArcFreeSupportTable(runtime, 16, 2, edges, masks=masks)
    ordinary = runtime.prepare(16, 2, edges, masks=masks, mode="scc")
    queries = tuple(
        ((vertex, 1 << ((vertex + phase) & 1)),)
        for phase in range(2)
        for vertex in range(16)
    )

    def run(query):
        support = table.solve(query)
        search = ordinary.solve(query)
        return (
            support.status,
            support.labels,
            search.status,
            ordinary.check(support.labels, query) if support.labels else False,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(run, queries * 8))
    assert all(a == c == "SAT_VERIFIED" and d for a, _, c, d in rows)
    for query in queries:
        values = {labels for a, labels, _, _ in (run(query) for __ in range(5)) if a == "SAT_VERIFIED"}
        assert len(values) == 1
    ordinary.close()
    table.close()


def test_support_refuses_residual_arcs_and_resource_underflow(runtime):
    # The overlapping palettes create a one-way implication after SCC reduction.
    prepared = runtime.prepare(2, 3, ((0, 1),), masks=(3, 6), mode="scc")
    assert prepared.info["quotient_arcs"] > 0
    with pytest.raises(ValueError, match="arc-free"):
        prepared.solve_support(())
    prepared.close()

    with pytest.raises(MemoryError):
        runtime.checker(20, 2, (), tuple(3 for _ in range(20)), max_bytes=1)
    with pytest.raises((MemoryError, ValueError)):
        runtime.prepare(20, 2, (), masks=tuple(3 for _ in range(20)), max_build_bytes=1)


def test_strict_native_types_and_closed_handles(runtime):
    with pytest.raises(ValueError):
        runtime.prepare(True, 2, (), masks=(), mode="scc")
    with pytest.raises(ValueError):
        runtime.prepare(1, True, (), masks=(), mode="scc")
    with pytest.raises(ValueError):
        runtime.prepare(1, 2, [], masks=(), mode="scc")
    with pytest.raises(ValueError):
        runtime.prepare(1, 2, (), masks=[], mode="scc")

    prepared = runtime.prepare(1, 2, (), masks=(3,), mode="scc")
    prepared.close()
    prepared.close()
    with pytest.raises(RuntimeError):
        prepared.info
    with pytest.raises(RuntimeError):
        prepared.certificate()
    with pytest.raises(RuntimeError):
        prepared.solve_support(())
    with pytest.raises(RuntimeError):
        prepared.solve()
