from pathlib import Path
import dataclasses
import pytest
import torch
from deploy.m10_artifact import export_cpu_artifact, load_cpu_artifact
from deploy.m10_runtime import CPURecursiveRuntime
from model.trm import TRM
from spectra.inference import predict_final


def runtime(tmp_path: Path, seed: int, depth: int, *, n: int = 1, T: int = 1):
    torch.manual_seed(seed)
    model = TRM(dim=16, num_tokens=5, seq_len=16, n_layers=1, n=n, T=T,
                N_sup=depth, heads=4, max_grid_size=8, ternary=True, act8=True).cpu().eval()
    path = tmp_path / f'artifact-{seed}-{depth}-{n}-{T}.pt'
    export_cpu_artifact(model, path, height=4, width=4, box=2,
        source_checkpoint_sha256='synthetic-contract', source_checkpoint_tensor_sha256='synthetic-contract',
        training_seed=seed, training_step=0, data_provenance={'contract_only': True}, export_git_sha='contract')
    return CPURecursiveRuntime(load_cpu_artifact(path))


@pytest.mark.parametrize('seed', [11, 62])
@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('depth', [1, 4, 16])
def test_final_prediction_bitwise_equal_to_full_diagnostic_trace(tmp_path, seed, batch, depth):
    engine = runtime(tmp_path, seed, depth)
    torch.manual_seed(seed+819)
    x = torch.randint(0, 5, (batch, 16))
    trace = engine.forward(x)
    predicted = predict_final(engine, x)
    assert torch.equal(trace.logits, predicted.logits)
    assert torch.equal(trace.answer, predicted.answer)
    assert torch.equal(trace.step_outputs[-1]['halt_logit'], predicted.halt_logit)
    assert len(trace.step_outputs) == depth
    assert predicted.work['retained_recurrent_steps'] == 0
    assert predicted.work['halt_head_fp32_calls'] == 1
    assert predicted.work['native_calls_by_layer']['out_head'] == 1
    assert predicted.work['native_linear_calls'] == trace.work['native_linear_calls']-depth+1
    assert predicted.work['native_call_count_matches_architecture']
    assert predicted.work['a8_quantization_calls'] == trace.work['a8_quantization_calls']
    assert set(dataclasses.asdict(predicted)) == {'logits', 'answer', 'halt_logit', 'work'}
    # The optional API must not change subsequent historical execution or telemetry.
    repeated = engine.forward(x)
    assert torch.equal(repeated.logits, trace.logits)
    assert repeated.work == trace.work


@pytest.mark.parametrize('n,T', [(1, 3), (2, 1), (2, 2)])
def test_nested_recurrence_order_exact(tmp_path, n, T):
    engine = runtime(tmp_path, 9, 3, n=n, T=T)
    x = torch.arange(16).remainder(5)[None, :]
    trace = engine.forward(x)
    prediction = predict_final(engine, x)
    assert torch.equal(trace.logits, prediction.logits)
    assert torch.equal(trace.answer, prediction.answer)
    assert prediction.work['native_call_count_matches_architecture']


def test_bad_inputs_and_subclasses_not_silently_supported(tmp_path):
    engine = runtime(tmp_path, 8, 2)
    with pytest.raises(TypeError):
        predict_final(engine, torch.zeros(1, 16))
    with pytest.raises(ValueError):
        predict_final(engine, torch.zeros(1, 15, dtype=torch.long))
    with pytest.raises(TypeError):
        predict_final(None, torch.zeros(1, 16, dtype=torch.long))
    class Derived(CPURecursiveRuntime):
        pass
    with pytest.raises(TypeError):
        predict_final(Derived(engine.artifact), torch.zeros(1, 16, dtype=torch.long))
