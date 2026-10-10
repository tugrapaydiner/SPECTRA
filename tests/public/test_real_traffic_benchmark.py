from experiments.real_traffic.benchmark import minfill_order, model_to_labels


def test_minfill_order_is_a_complete_deterministic_permutation() -> None:
    edges = ((0, 1), (1, 2), (2, 3), (0, 3))
    first = minfill_order(4, edges)
    second = minfill_order(4, edges)
    assert first == second
    assert sorted(first) == [0, 1, 2, 3]


def test_native_model_mapping_keeps_original_colour_meaning() -> None:
    masks = (0b0011, 0b0110, 0b1000)
    labels = model_to_labels((1, -2, 3), masks)
    assert labels == (1, 1, 3)
