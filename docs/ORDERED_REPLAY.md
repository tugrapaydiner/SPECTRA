# Frozen M17 replay: explicit CPU reduction order

## The actual blocker

The `historical-avx2` ISA caps do not make MKL choose identical arithmetic on
Intel and AMD. A controlled same-host investigation retained the original
failures instead of selecting a favorable runner or updating expected hashes.

PR20 CPU run `34560492307`, Python 3.13, Intel Xeon Platinum 8573C:
artifact `10184209157`, ZIP SHA256
`f3eb49caf788f4618e9149dfe1798529f752e55aa575d92709ea8cbd64850ce4`.
The original, SSE4.2-capped and CNR-compatible probes all complete, but none
matches the two historical development pool hashes on that host. The old strict
replay and the separate SSE4.2 strict replay both fail. These failures are real.

AMD EPYC 9V74 comparison: run `34560186245`, Python 3.11, artifact `10184052028`,
ZIP SHA256 `f979e72a21dcaacee6b63eeb040586d3d8cc433b9308ff923e023c958973085b`.
Original and SSE4.2-capped execution match the old hashes; CNR-compatible does
not. A separate retained AMD EPYC 7763 trace agrees with the 9V74 trace. This is
an observed bounded comparison, not a claim that every CPU has been tested.

Intel documents that `MKL_ENABLE_INSTRUCTIONS` only controls Intel processors;
CNR COMPATIBLE has a different numerical contract. Neither setting alone is a
justification for silently accepting a changed historical tensor.

References: Intel oneMKL `mkl_enable_instructions` and “Getting Started with
Conditional Numerical Reproducibility”; PyTorch “Numerical accuracy”.

## Localized, arithmetic explanation

Read-only module hooks are checked against an unhooked execution, then removed.
The packed native attention decomposition is independently required to reproduce
the unmodified attention output exactly on the host before its intermediates
are retained. Both independent and original pool constructors agree on each
host; the divergence is not caused by the independent reconstruction algorithm.

For Sudoku core 1401's first supervision half-cycle, embedding, first RMSNorm,
packed QKV, both attention contractions, output projection, first FFN linear,
and GELU all agree. The second FFN linear differs at 27,871 of 32,768 positions,
with maximum absolute error 1.9073486328125e-6. Its historical K=256 arithmetic is:

```
p0 = FMA(x[0], w[0], ... FMA in increasing k through 127 ...)
p1 = FMA(x[128], w[128], ... increasing k through 255 ...)
answer = float32(float32(p0 + bias) + p1)
```

Both partial accumulators start at positive zero. Bias is added after the FIRST
128-term partial sum, not after summing all products. A scalar libm `fmaf` oracle
reproduces all 32,768 original first-layer outputs exactly with this order;
putting bias after both partial sums does not.

For maze core 1701, the first divergence is in QK: 64 differences, all at the
last row/column, maximum 1.1920928955078125e-7. The first-cycle softmax happens to
round identically despite these differences. The subsequent probability-times-V
contraction differs at 1,260 entries (row 84), maximum 7.152557373046875e-7. Both
historical contractions are reproduced by increasing-k FP32 fused accumulation.
These are trace-specific counts, not a theorem that all later drift is confined
to those coordinates.

## The explicit reproduction profile

`--cpu-profile historical-ordered` retains the old AVX2 Torch/oneDNN/MKL dispatch
caps and additionally uses `eval.historical_numerics.ordered_core` for the two
frozen core geometries. A small C++ AVX2/FMA extension fixes only the affected
reductions. Vector lanes compute independent output entries; k is never
reassociated. Compiler flags prohibit fast-math and implicit contraction.

The context overrides instance methods only. It does not globally monkeypatch
Torch, mutate parameters, alter checkpoints or pass an expected tensor/answer
into an operator. It rejects training, unknown geometries, batches other than
32, masked/cross attention, and preexisting custom forwards. Original instance
methods are restored on both normal exit and exceptions. It must not be used
concurrently on a shared core. The remaining numerical library operations and
private packed-QKV primitive are explicitly bound to PyTorch 2.10.0+cpu.

This profile is **for faithful replay**, not a faster runtime, not the B=1
geometric restart solver, and not a universal bitwise portability guarantee.
Historical latency numbers are not transferred to it. The old `historical-avx2`
and `host` modes remain available to expose the original drift.

## Acceptance stays strict

Every one of the six original pool hashes must match. Candidate/continuation
labels, pool membership, selected indices, per-family decisions, and checkpoint
identities must remain exact. The existing evaluator-score tolerance is not
increased; the successful local explicit-order run has all 49,152 scores exactly
equal, as well as all 16,384 reconstructed candidate states. No failed scientific
gate is promoted. No new confirmation is generated or used for model selection.

The extension has independent scalar `fmaf` oracle tests, bias-order
counterexamples, vector-tail and malformed-input tests. Full frozen replay,
not those primitive tests alone, is the integration acceptance criterion.

Local replay passed before publication. Cross-host CI must validate the final
source before merge; no passing ancestor or favorable runner substitutes for
that requirement.
