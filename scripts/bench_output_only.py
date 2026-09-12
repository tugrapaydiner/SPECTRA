"""Bounded CPU API cost and retained-tensor comparison, not a quality benchmark."""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from deploy.m10_artifact import export_cpu_artifact, load_cpu_artifact
from deploy.m10_runtime import CPURecursiveRuntime
from model.trm import TRM
from spectra.inference import predict_final

CONFIG = dict(dims=[16, 64], depths=[1, 4, 16], batches=[1, 4], seed=91426, rounds=5)


def retained_bytes(obj):
    """Actual unique CPU tensor storage reachable from the returned API object."""
    storages = {}
    def visit(item):
        if isinstance(item, torch.Tensor):
            store = item.untyped_storage()
            storages[(str(item.device), store.data_ptr())] = store.nbytes()
        elif dataclasses.is_dataclass(item):
            for field in dataclasses.fields(item):
                visit(getattr(item, field.name))
        elif isinstance(item, dict):
            for value in item.values(): visit(value)
        elif isinstance(item, (tuple, list)):
            for value in item: visit(value)
    visit(obj)
    return sum(storages.values())


def digest(tensor):
    raw = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    write(out/'config.json', CONFIG)
    source_paths = ['scripts/bench_output_only.py', 'spectra/inference.py',
                    'deploy/m10_runtime.py', 'deploy/m10_artifact.py',
                    'deploy/m10_native.py', 'deploy/m10_dense_extension.cpp']
    write(out/'source_sha256.json', {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths})
    write(out/'environment.json', dict(python=sys.version, platform=platform.platform(),
                                     torch=torch.__version__, threads=torch.get_num_threads()))
    records, cells = [], {}
    with (out/'rows.jsonl').open('x') as stream:
        for dim in CONFIG['dims']:
            for depth in CONFIG['depths']:
                seed = CONFIG['seed']+dim+depth
                torch.manual_seed(seed)
                model = TRM(dim=dim, num_tokens=5, seq_len=16, n_layers=1, n=1, T=1,
                            N_sup=depth, heads=4, max_grid_size=8, ternary=True, act8=True).cpu().eval()
                artifact = out/f'artifact-{dim}-{depth}.pt'
                export_cpu_artifact(model, artifact, height=4, width=4, box=2,
                    source_checkpoint_sha256='synthetic-api-cost', source_checkpoint_tensor_sha256='synthetic-api-cost',
                    training_seed=seed, training_step=0, data_provenance={'untrained_contract_artifact': True},
                    export_git_sha='see-source-sha256')
                del model
                engine = CPURecursiveRuntime(load_cpu_artifact(artifact))
                for batch in CONFIG['batches']:
                    torch.manual_seed(seed+batch)
                    x = torch.randint(0, 5, (batch, 16))
                    # Native compilation and both warmups are outside the timed API.
                    baseline = engine.forward(x)
                    candidate = predict_final(engine, x)
                    assert torch.equal(baseline.logits, candidate.logits)
                    assert torch.equal(baseline.answer, candidate.answer)
                    assert torch.equal(baseline.step_outputs[-1]['halt_logit'], candidate.halt_logit)
                    expected = (digest(baseline.logits), digest(baseline.answer), digest(baseline.step_outputs[-1]['halt_logit']))
                    storage = dict(trace=retained_bytes(baseline), final=retained_bytes(candidate))
                    del baseline, candidate
                    case = f'd{dim}:depth{depth}:batch{batch}'
                    times = {'trace': [], 'final': []}
                    for round_id in range(CONFIG['rounds']):
                        order = ['trace', 'final'] if (round_id+batch+depth) % 2 else ['final', 'trace']
                        for mode in order:
                            start = time.perf_counter_ns()
                            result = engine.forward(x) if mode == 'trace' else predict_final(engine, x)
                            elapsed = time.perf_counter_ns()-start
                            halt = result.step_outputs[-1]['halt_logit'] if mode == 'trace' else result.halt_logit
                            actual = (digest(result.logits), digest(result.answer), digest(halt))
                            if actual != expected: raise AssertionError('final numerical output changed')
                            if retained_bytes(result) != storage[mode]: raise AssertionError('storage changed across rounds')
                            record = dict(case=case, mode=mode, round=round_id, wall_ns=elapsed,
                                logits_sha256=actual[0], answer_sha256=actual[1], halt_sha256=actual[2],
                                retained_tensor_bytes=storage[mode], native_calls=result.work['native_linear_calls'],
                                halt_calls=result.work['halt_head_fp32_calls'])
                            records.append(record); stream.write(json.dumps(record)+'\n');stream.flush()
                            times[mode].append(elapsed)
                            del result, halt
                    cells[case] = dict(mean_ms={k: statistics.mean(v)/1e6 for k,v in times.items()},
                        mean_ratio=statistics.mean(times['final'])/statistics.mean(times['trace']),
                        retained_tensor_bytes=storage, retained_ratio=storage['final']/storage['trace'])
                    print(case, cells[case], flush=True)
    report = dict(schema='spectra.output_only_api.v1', exact_final_outputs=True,
                  measured_calls=len(records), independent_artifacts=6, cells=cells,
                  quality_benchmark=False, untrained_artifacts=True,
                  timing_scope='warm CPU API call, excluding input generation and independent comparison',
                  memory_scope='unique tensor storage reachable from returned object; NOT peak execution memory or RSS',
                  claim='output retention and head work; no broad speedup or learned-quality claim')
    write(out/'summary.json', report)
    write(out/'SHA256.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.out), indent=2))
