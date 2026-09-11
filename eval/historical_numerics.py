"""Opt-in M17 CPU replay arithmetic, not a new trained inference algorithm.

The frozen 32-example pools were produced by AMD MKL. An ISA cap alone does
not select that reduction order on Intel. This reproduces the observed orders
explicitly, while STILL requiring all original full-pool hashes to match.
No expected tensor or answer enters these operators. Core state is not changed.
The scope is the two frozen M17 core geometries, PyTorch 2.10 CPU, x86 AVX2/FMA.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import hashlib
from pathlib import Path
import platform
import types

import torch

from eval.checkable_tasks import require_core

PROFILE = 'm17-fp32-ordered-v1'
FLAGS = ['-O3', '-std=c++17', '-mavx2', '-mfma', '-ffp-contract=off', '-fno-fast-math']
SOURCE = Path(__file__).resolve().parents[1]/'deploy/replay_ordered_extension.cpp'


@lru_cache(maxsize=1)
def extension():
    if platform.machine().lower() not in {'x86_64','amd64'} or platform.system() != 'Linux':
        raise RuntimeError('historical ordered replay requires Linux x86-64 AVX2/FMA')
    flags = next((line.split(':',1)[1].split() for line in Path('/proc/cpuinfo').read_text().splitlines()
                  if line.startswith('flags')), [])
    if not {'avx2','fma'} <= set(flags):
        raise RuntimeError('historical ordered replay requires AVX2 and FMA')
    if str(torch.__version__) != '2.10.0+cpu':
        raise RuntimeError('historical ordered replay is bound to torch 2.10.0+cpu')
    from torch.utils.cpp_extension import load
    return load(name='spectra_replay_ordered_v1', sources=[str(SOURCE)],
                extra_cflags=FLAGS, verbose=False)


def identity():
    return {'profile': PROFILE, 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'compile_flags': list(FLAGS), 'torch': str(torch.__version__),
            'scope': 'frozen M17 B=32 CPU core replay only; no latency or universal portability claim',
            'expected_outputs_used': False,
            'sudoku_ff2': 'k=256, two increasing-k FMA blocks of 128; bias added between blocks',
            'maze_attention': 'increasing-k FP32 FMA for both attention contractions',
            'maze_linears': 'increasing-k FP32 FMA; one full-K block then bias for core projections/FFN'}


def _replay_input(x, shape):
    if (not isinstance(x,torch.Tensor) or tuple(x.shape)!=shape or x.dtype!=torch.float32
        or x.device.type!='cpu' or x.requires_grad or torch.is_grad_enabled()):
        raise ValueError('ordered replay requires its frozen B=32 geometry under no_grad/inference_mode')
    if not bool(torch.isfinite(x).all()):
        raise ValueError('ordered replay input must be finite')


def _linear_forward(self,x):
    _replay_input(x,(32,16,256))
    return extension().blocked_linear(x.reshape(-1,256).contiguous(),
        self.weight.T.contiguous(),self.bias,128).reshape(32,16,64)


def _full_linear(x, weight, bias=None):
    k=x.shape[-1];n=weight.shape[0]
    if bias is None:
        bias=torch.zeros(n,dtype=torch.float32,device='cpu')
    return extension().blocked_linear(x.reshape(-1,k).contiguous(),
            weight.T.contiguous(),bias,k).reshape(*x.shape[:-1],n)


def _maze_ff_forward(self,x):
    _replay_input(x,(32,121,self.in_features))
    return _full_linear(x,self.weight,self.bias)


def _attention_forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                       attn_mask=None, average_attn_weights=True, is_causal=False):
    _replay_input(query,(32,121,48))
    if (key is not query or value is not query or key_padding_mask is not None or
        attn_mask is not None or need_weights is not False or is_causal is not False):
        raise ValueError('ordered replay supports only unmasked, weight-free self attention')
    b,n,d=query.shape;h=self.num_heads
    qkv=_full_linear(query,self.in_proj_weight)
    q,k,v=torch._transform_bias_rescale_qkv(qkv,self.in_proj_bias,h)
    qk=extension().ordered_bmm(q.reshape(-1,n,d//h).contiguous(),
            k.reshape(-1,n,d//h).transpose(-2,-1).contiguous()).view(b,h,n,n)
    probability=torch.softmax(qk,dim=-1)
    context=extension().ordered_bmm(probability.reshape(-1,n,n).contiguous(),
            v.reshape(-1,n,d//h).contiguous()).view(b,h,n,d//h)
    return _full_linear(context.transpose(1,2).reshape(b,n,d),self.out_proj.weight,self.out_proj.bias),None


@contextmanager
def ordered_core(core,spec):
    """Temporarily override instance methods; restore them even on failure.

    No global Torch monkeypatch, parameter mutation, new checkpoint, changed
    labels, geometry selection from outcomes, or loosening of hash contracts.
    Do not share this core concurrently with another execution during the scope.
    """
    require_core(core,spec)
    if len(core.blocks)!=1 or (core.dim,core.seq_len) not in {(64,16),(48,121)}:
        raise ValueError('unsupported historical core geometry')
    block=core.blocks[0]
    if (not isinstance(block.attn,torch.nn.MultiheadAttention) or block.attn.num_heads!=4
        or block.attn.dropout!=0 or not block.attn.batch_first or block.attn.in_proj_weight is None
        or block.attn.in_proj_bias is None or block.attn.bias_k is not None
        or block.attn.bias_v is not None or block.attn.add_zero_attn):
        raise ValueError('unsupported historical attention contract')
    overrides=([(block.ff[2],_linear_forward)] if core.dim==64 else
               [(block.attn,_attention_forward),(block.ff[0],_maze_ff_forward),
                (block.ff[2],_maze_ff_forward)])
    for module,forward in overrides:
        if forward is _linear_forward and (not isinstance(module,torch.nn.Linear) or module.in_features!=256
                             or module.out_features!=64 or module.bias is None):
            raise ValueError('unsupported historical FF2 contract')
        if forward is _maze_ff_forward and (not isinstance(module,torch.nn.Linear) or module.bias is None
                            or (module.in_features,module.out_features) not in {(48,192),(192,48)}):
            raise ValueError('unsupported historical maze FF contract')
        if 'forward' in module.__dict__:
            raise ValueError('refusing to replace an already-customized replay module')
    extension()
    installed=[]
    try:
        for module,forward in overrides:
            module.forward=types.MethodType(forward,module)
            installed.append(module)
        yield identity()
    finally:
        for module in installed:
            del module.forward
