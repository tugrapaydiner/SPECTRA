"""Tiny custom development fixtures only; never generate the 48 panel inputs."""
import random
import pytest

from .binary_list import BinaryListProblem
from .cases import Case
from .transfer_panel import FAMILIES, topology, make_case, gmean, seed_for


@pytest.mark.parametrize('family', FAMILIES)
def test_tiny_generators_are_deterministic_canonical_and_bounded(family):
    # n=8 is not a panel size; random seed is not a panel seed.
    a=topology(family,8,random.Random(731))
    b=topology(family,8,random.Random(731))
    assert a==b==tuple(sorted(set(a)))
    assert all(0<=u<v<8 for u,v in a)


@pytest.mark.parametrize('family', FAMILIES)
def test_tiny_cases_round_trip_without_native_backend(family, monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('generator attempted to solve')
    monkeypatch.setattr('experiments.compiled_query.exact.ExactRuntime',forbidden)
    case=make_case(family,2,count=64,n=8)
    assert Case.decode(case.payload())==case
    assert len(case.queries)==64
    assert len(case.queries[0])==1
    assert case.queries[28]==()
    assert any(mask==0 for _,mask in case.queries[30])
    assert any(mask>>63 for _,mask in case.queries[22])


@pytest.mark.parametrize('family,n', [('unknown',8),('random_tree',7),('random_tree',9)])
def test_invalid_topology_request_is_rejected(family,n):
    with pytest.raises(ValueError):
        topology(family,n,random.Random(1))


def test_ratios_and_seed_definitions():
    assert gmean([.25,4])==1
    assert seed_for('a')==seed_for('a')!=seed_for('b')


def test_case_generation_rejects_boolean_count():
    with pytest.raises(ValueError):
        make_case('random_tree',0,count=True,n=8)
