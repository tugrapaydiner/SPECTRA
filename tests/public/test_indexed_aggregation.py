"""Exact, version-independent aggregation contracts for the balanced benchmark."""
import importlib.util
from pathlib import Path
import pytest

SPEC = importlib.util.spec_from_file_location("indexed_aggregation_benchmark", Path(__file__).resolve().parents[2]/"scripts/bench_indexed_search.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def test_balanced_integer_aggregation_avoids_version_specific_float_sums(monkeypatch):
    from fractions import Fraction
    monkeypatch.setitem(bench.CONFIG, 'bootstrap_repeats', 32)
    table = {}
    for i in range(4):
        table[(str(i), 4096, 'reference')] = [9007199254740993 + i, 1 + i, 2 + i]
        table[(str(i), 4096, 'indexed_cold')] = [6007199254740993 + i, 2 + i, 7 + i]
    ids = [str(i) for i in range(4)]
    report = bench.ratios_by_formula(ids, table, 4096, 'indexed_cold', 91327)
    before = sum(sum(table[(c, 4096, 'reference')]) for c in ids)
    after = sum(sum(table[(c, 4096, 'indexed_cold')]) for c in ids)
    assert report['mean_ratio'] == float(Fraction(after, before))
    assert report['reference_mean_ms'] == float(Fraction(before, 12_000_000))
    assert report['candidate_mean_ms'] == float(Fraction(after, 12_000_000))


def test_unbalanced_integer_aggregation_rejected():
    table = {('a', 4096, 'reference'): [1, 2], ('a', 4096, 'indexed_cold'): [1]}
    with pytest.raises(ValueError, match='balanced'):
        bench.ratios_by_formula(['a'], table, 4096, 'indexed_cold', 91327)
