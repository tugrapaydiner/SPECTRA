"""Fixed-width SVM interfaces must reject excess input without consuming it all."""
from __future__ import annotations

import pytest

from spectra.svm import _input, verify_certificate


def guarded_values(value, limit):
    """An endless source represented without hanging a broken implementation."""
    for _ in range(limit):
        yield value
    raise AssertionError('input consumed beyond its bounded look-ahead')


@pytest.mark.parametrize('classes', [2, 3, 16, 128])
def test_certificate_stops_after_one_excess_outcome(classes):
    count = classes * (classes - 1) // 2
    assert not verify_certificate(classes, 0, guarded_values(0, count + 1))


def test_legacy_input_stops_after_one_excess_feature():
    with pytest.raises(ValueError, match='sixteen features'):
        _input(guarded_values(0., 17))


@pytest.mark.parametrize('count', [0, 1, 15, 17])
def test_legacy_input_requires_exactly_sixteen_features(count):
    with pytest.raises(ValueError, match='sixteen features'):
        _input(iter([0.] * count))


def test_valid_input_and_certificate_iterators_keep_their_semantics():
    row = [i / 16 for i in range(16)]
    assert list(_input(iter(row))) == row
    assert verify_certificate(3, 0, iter([0, 0, -1]))
    assert not verify_certificate(3, 1, iter([0, 0, -1]))
    assert not verify_certificate(3, 0, iter([0, 0]))


def test_invalid_geometry_does_not_consume_certificate():
    assert not verify_certificate(129, 0, guarded_values(0, 0))
    assert not verify_certificate(3, -1, guarded_values(0, 0))
