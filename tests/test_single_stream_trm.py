import torch

from model.single_stream_trm import InputConditionedSingleStreamTRM


def make_model(seed: int) -> InputConditionedSingleStreamTRM:
    torch.manual_seed(int(seed))
    return InputConditionedSingleStreamTRM(
        dim=96,
        num_tokens=5,
        seq_len=16,
        n_layers=1,
        T=1,
        N_sup=4,
        heads=4,
        alpha_y=0.1,
        max_grid_size=8,
    ).cpu()


def test_single_stream_first_step_is_input_conditioned_and_z_stays_zero():
    model = InputConditionedSingleStreamTRM(
        dim=16, num_tokens=5, seq_len=16, n_layers=1, T=1, N_sup=1,
        heads=4, alpha_y=0.1, max_grid_size=8,
    )
    # Make the recurrence algebra transparent: the update is exactly x_emb+y.
    model.f = lambda h: h
    x1 = torch.tensor([[1,0,0,4,0,4,1,0,0,1,4,0,4,0,0,1]])
    x2 = torch.tensor([[2,0,0,3,0,3,2,0,0,2,3,0,3,0,0,2]])
    s1 = model.init_execution_state(x1, height=4, width=4)
    s2 = model.init_execution_state(x2, height=4, width=4)
    z1 = s1.z.clone(); z2 = s2.z.clone()
    o1 = model.run_execution_step(s1)
    o2 = model.run_execution_step(s2)
    assert not torch.equal(o1["y"], o2["y"])
    assert torch.equal(o1["z"], z1)
    assert torch.equal(o2["z"], z2)


def test_single_stream_executes_one_block_application_per_supervision_step():
    torch.manual_seed(0)
    model = InputConditionedSingleStreamTRM(
        dim=16, num_tokens=5, seq_len=16, n_layers=1, T=1, N_sup=3,
        heads=4, alpha_y=0.1, max_grid_size=8,
    ).eval()
    x = torch.randint(0, 5, (1, 16))
    with torch.inference_mode():
        _, steps = model(x, height=4, width=4)
    assert len(steps) == 3
    assert steps[0]["work_cumulative"]["block_applications"] == 1
    assert steps[1]["work_cumulative"]["block_applications"] == 2
    assert steps[2]["work_cumulative"]["block_applications"] == 3
    assert steps[2]["work_cumulative"]["recursive_cycles"] == 3


def test_single_stream_unused_z_parameters_are_not_optimizer_owned():
    model = make_model(1401)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert "alpha_z" not in trainable
    assert not any(name.startswith("norm_z.") for name in trainable)
    assert "alpha_y" in trainable


def test_single_stream_semantics_are_explicit():
    model = make_model(2402)
    assert model.n == 0
    assert model.SEMANTICS == "input_conditioned_single_stream_v1"
