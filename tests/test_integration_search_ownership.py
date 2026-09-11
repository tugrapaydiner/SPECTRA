"""Adversarial callback fixtures; no scientific holdout is generated here."""
from itertools import product

import pytest

from eval.verified_search import BudgetedVerifiedSearch, CheckedIncumbent, ValueContract, ValueTarget


SHA = "a" * 64


def engine(transition, *, initial=lambda: [], decode=lambda state: state,
           checker=lambda answer: False, value=lambda state: .5):
    return BudgetedVerifiedSearch(initial=initial, actions=(0, 1), transition=transition,
        decode=decode, checker=checker, value=value,
        contract=ValueContract(ValueTarget.QUALITY, SHA, "b"*64, "fixture.v1"),
        model_sha256=SHA, transition_id="fixture.v1")


def test_in_place_transition_does_not_corrupt_sibling_or_caller_state():
    initial = []
    visited = []
    def transition(state, action):
        state.append(action)
        visited.append(tuple(state))
        return state
    result = engine(transition, initial=lambda: initial, checker=lambda a: a == [1]).solve(
        max_transitions=2, max_depth=1)
    assert result.valid and result.answer == [1]
    assert visited == [(0,), (1,)]
    assert initial == []
    assert result.work["state_ownership"] == "defensive_snapshots_v1"
    assert result.work["state_snapshots"] > 0


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_reused_transition_scratch_cannot_change_queued_nodes(depth):
    scratch = []
    visited = []
    def transition(state, action):
        scratch[:] = state + [action]
        visited.append(tuple(scratch))
        return scratch
    count = sum(2**d for d in range(1, depth+1))
    result = engine(transition).solve(max_transitions=count+1, max_depth=depth)
    expected = {p for d in range(1, depth+1) for p in product((0, 1), repeat=d)}
    assert len(visited) == count and set(visited) == expected
    assert result.work["transitions"] == result.work["checks"] == count
    assert result.work["stop_reason"] == "tree_exhausted"


def test_mutating_decode_and_value_receive_no_stored_node_alias():
    seen = []
    def transition(state, action):
        seen.append(tuple(state))
        return state+[action]
    def decode(state):
        state.append(8)
        return state
    def value(state):
        state.append(9)
        return .5
    engine(transition, decode=decode, value=value).solve(max_transitions=6, max_depth=2)
    assert not any(8 in state or 9 in state for state in seen)


def test_checker_retaining_argument_cannot_later_mutate_incumbent():
    borrowed = []
    def checker(answer):
        borrowed.append(answer)
        return answer["id"] == 1
    incumbent = CheckedIncumbent(checker)
    incumbent.offer({"id": 1}, 0., (1,))
    borrowed[0]["id"] = 0
    assert incumbent.snapshot().answer == {"id": 1}
    incumbent.offer({"id": 0}, 1., (0,))
    assert incumbent.snapshot().valid and incumbent.snapshot().answer == {"id": 1}


@pytest.mark.parametrize("field", ["transition_id", "state_schema"])
@pytest.mark.parametrize("value", [1, True, [], None, ""])
def test_value_contract_identifiers_are_nonempty_strings(field, value):
    args = dict(target=ValueTarget.QUALITY, model_sha256=SHA, evaluator_sha256="b"*64,
                transition_id="fixture.v1", state_schema="xyz")
    args[field] = value
    with pytest.raises(ValueError):
        ValueContract(**args)
