#!/usr/bin/env python3
"""Capture a bounded CPU numerical trace; diagnostic completion is not replay PASS.

Only the first already-consumed M17 development pool/core is inspected. The historical
hash is reported, never rewritten or waived. The strict full replay is separate.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

# Apply exactly the existing profile before the first torch/numpy import.
PROFILE = {"ATEN_CPU_CAPABILITY": "avx2", "MKL_ENABLE_INSTRUCTIONS": "AVX2",
           "ONEDNN_MAX_CPU_ISA": "AVX2", "DNNL_MAX_CPU_ISA": "AVX2"}
DISPATCH_CHOICES = ("original", "sse42", "compatible")

def configure_dispatch(choice: str) -> None:
    if choice not in DISPATCH_CHOICES:
        raise ValueError("unknown MKL dispatch diagnostic")
    if "torch" in sys.modules or "numpy" in sys.modules:
        raise RuntimeError("dispatch must be selected before importing numerical libraries")
    os.environ.update(PROFILE)
    os.environ.pop("MKL_CBWR", None)
    if choice == "sse42":
        os.environ["MKL_ENABLE_INSTRUCTIONS"] = "SSE4_2"
    elif choice == "compatible":
        os.environ["MKL_CBWR"] = "COMPATIBLE"

if __name__ == "__main__":
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("--mkl-dispatch", choices=DISPATCH_CHOICES, default="original")
    selection, _ = early.parse_known_args()
    configure_dispatch(selection.mkl_dispatch)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from scripts.verify_fixed_pool_replay import (Evidence, AcceptedM16, M17_INVENTORY_SHA,
    M17_ZIP_SHA, FAMILY_SPECS, frozen_inputs, load_sources, read_bound_archive,
    reconstruct_pool, source_identity, tensor_digest, write_json)
from scripts.m17_models import candidate_pool


def capture_first_cycle(core, inputs, spec) -> dict[str, torch.Tensor]:
    """Read-only hooks, with a no-hook execution equality check."""
    result = {}
    counts = {}
    handles = []
    def hook(name):
        def collect(module, args, output):
            ordinal = counts.get(name, 0)
            counts[name] = ordinal + 1
            if isinstance(output, tuple):
                output = output[0]
            if isinstance(output, torch.Tensor):
                result[f"{name or 'core'}_{ordinal}"] = output.detach().cpu().clone()
        return collect
    with torch.no_grad():
        x = inputs[:32]
        embedded = core.token_embed(x) + core.encode_positions(x, spec.height, spec.width)
        initial = torch.zeros_like(embedded)
        y0, z0 = core.recursive_cycle(embedded, initial, initial)
        try:
            for name, module in core.named_modules():
                handles.append(module.register_forward_hook(hook(name)))
            e = core.token_embed(x) + core.encode_positions(x, spec.height, spec.width)
            y1, z1 = core.recursive_cycle(e, torch.zeros_like(e), torch.zeros_like(e))
        finally:
            for handle in handles:
                handle.remove()
        if not all(torch.equal(a, b) for a, b in ((e, embedded), (y1, y0), (z1, z0))):
            raise ValueError("diagnostic hooks changed numerical execution")
        result.update(input=x.clone(), embedded=embedded, y=y0, z=z0)
    return result


@torch.no_grad()
def capture_attention_stages(core, trace):
    """Expose the first native-MHA contractions without changing its execution.

    Reconstruct the same packed QKV/softmax path and REQUIRE exact final equality
    to the unmodified module output on this host before retaining intermediates.
    This localizes vendor BLAS differences; it does not substitute a new kernel.
    """
    import torch.nn.functional as F
    inp = trace['blocks.0.norm1_0']
    attn = core.blocks[0].attn
    b, n, d = inp.shape
    heads = attn.num_heads
    qkv = F.linear(inp, attn.in_proj_weight)
    q, k, v = torch._transform_bias_rescale_qkv(qkv, attn.in_proj_bias, heads)
    qk = torch.bmm(q.reshape(-1, n, d//heads),
                   k.reshape(-1, n, d//heads).transpose(-2, -1)).reshape(b, heads, n, n)
    probability = torch.softmax(qk, dim=-1)
    context = torch.bmm(probability.reshape(-1, n, n),
                        v.reshape(-1, n, d//heads)).view(b, heads, n, d//heads)
    output = F.linear(context.transpose(1, 2).reshape(b, n, d),
                      attn.out_proj.weight, attn.out_proj.bias)
    if not torch.equal(output, trace['blocks.0.attn_0']):
        raise ValueError('decomposed attention differs from the native path on this host')
    return {'mha_' + key: value.detach().clone() for key, value in
            dict(qkv=qkv, q=q, k=k, v=v, qk=qk, probability=probability,
                 context=context, output=output).items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mkl-dispatch', choices=DISPATCH_CHOICES, default='original')
    parser.add_argument('--family', choices=tuple(FAMILY_SPECS), default='sudoku_shift')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError('requires CPU-only torch')
    h = Evidence(Path('results/m16/sources'))
    a = AcceptedM16(Path('results/m16/runs/accepted-34522192590.tar.gz'), h)
    m = read_bound_archive(Path('results/m17/runs/34534009702-33e548d3e0f52d0664c00929316b2198d5fcffe4.tar.gz'),
                          inventory_sha=M17_INVENTORY_SHA, zip_sha=M17_ZIP_SHA)
    spec, seed, _ = FAMILY_SPECS[args.family]
    prefix = f'experiment/{args.family}/'
    with tempfile.TemporaryDirectory() as td:
        sources = load_sources(args.family, m, h, a, Path(td))
        core_seed = min(sources)
        core, _ = sources[core_seed]
        x, ids, manifest_sha = frozen_inputs(m, prefix, seed, spec)
        p, _ = reconstruct_pool(core, x, spec)
        q = candidate_pool(core, x, spec)
        if tensor_digest(p) != tensor_digest(q):
            raise ValueError('independent and production pools differ on this host')
        trace = capture_first_cycle(core, x, spec)
        trace.update(capture_attention_stages(core, trace))
    observed = tensor_digest(p)
    expected = json.loads(m[prefix+'summary.json'])['development_fixed_pool']['pool_tensor_sha256_by_core'][str(core_seed)]
    report = {'status': 'DIAGNOSTIC_COMPLETE_NOT_ACCEPTANCE', 'source': source_identity(),
        'mkl_dispatch_diagnostic': args.mkl_dispatch,
        'pool': f'{args.family}/development/{core_seed}', 'manifest_sha256': manifest_sha,
        'expected_pool_hash': expected, 'observed_pool_hash': observed,
        'historical_hash_match': observed == expected, 'independent_production_equal': True,
        'trace_no_hook_equal': True, 'trace_sha256': tensor_digest(trace),
        'cpuinfo': Path('/proc/cpuinfo').read_text().split('\n\n')[0],
        'python': sys.version, 'torch': str(torch.__version__), 'numpy': np.__version__,
        'torch_build': torch.__config__.show(),
        'dispatch_environment': {k: os.environ.get(k) for k in (*PROFILE, 'MKL_CBWR')},
        'fields': {k: {'shape': list(v.shape), 'sha256': tensor_digest({k:v})} for k, v in trace.items()}}
    np.savez_compressed(args.out/'first_cycle.npz', **{k:v.numpy() for k,v in trace.items()})
    write_json(args.out/'diagnostic.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('source', 'cpuinfo', 'fields', 'torch_build')}, indent=2))
    return 0  # Completed diagnosis; the strict verifier still enforces the hash.

if __name__ == '__main__':
    raise SystemExit(main())
