# Conditional-compute contract and explicit statistical limitations

This experimental system stores two classifiers. It does not compress the SVM,
improve each constituent model, or provide a proof that its labels are correct.
The exact input schema, prototype classifier, original SVC class order and routing
policy are validated. Both classifiers contribute to deployment memory.

## Decision rule

Let f be the fixed 256-prototype classifier, g the selected SVC, and m(x) the gap
between the largest and second-largest prototype class scores. With threshold t,
return f(x) for m(x)>=t and otherwise g(x). An absent threshold explicitly invokes
g on every row WITHOUT computing f. A zero threshold accepts all finite fast gaps.
There is one Python-to-native call per chunk. Raw-code validation, the prototype
stage, confidence calculation, normalization and needed SVM work occur in that call.
A confidence-blind hash gate is an ablation, never the promoted policy.

The native prototype uses the existing exact-order integer-signature/product-table
function. The stronger stage reuses the prior finite/adaptive SVC engine, including
its existing conditional IEEE/libm error assumptions and exact fallback. Here all
native SVC outputs are checked against the selected fitted sklearn model. No new
universal libm, cross-platform, or exact-real arithmetic guarantee is inferred.

## What calibration could guarantee under additional assumptions

For a threshold t define H_t(x,y)=1{m(x)>=t, f(x)!=y, g(x)=y} and the corresponding
beneficial shortcut B_t=1{m(x)>=t, f(x)=y, g(x)!=y}. Then, identically,

    R(cascade_t) - R(g) = E[H_t] - E[B_t] <= E[H_t].

This is relative added harm, NOT the total error rate, error conditional on taking
a shortcut, or an individual prediction guarantee. The stronger model can itself
be wrong. Risk can decrease even when some harmful shortcuts occur.

Conditional on fixed models, an independent identically distributed calibration
sample from the target population makes each fixed H_t count binomial. A one-sided
Clopper-Pearson upper bound at failure probability .05/13 for each of13 fixed
thresholds gives simultaneous coverage by the union bound. Thus selecting among
those thresholds after examining calibration is allowed under those assumptions.
The .005/.01/.02 sensitivity budgets share the same simultaneous bounds; no new
thresholds are searched. If no threshold qualifies, always-strong has identically
zero relative added harm. These bounds are not simultaneous across all tasks.

**The benchmark does not establish the IID assumptions.** Exact-feature grouping
and stratification change sampling, contributor/scene information is unavailable
per row, and the official test may represent different writers or conditions.
The calibration split is independent of model fitting in the no-data-overlap
sense, which is weaker than an IID sampling theorem. These tests were also used
in earlier SPECTRA research. Consequently the reported bounds are calibration
rules/diagnostics, not a guaranteed safety budget for these deployments.

In fact Pendigits calibrates to accept all inputs, while its official-test added
harm is59/3498=1.6867%, above the nominal1% budget. The measured relative accuracy
loss versus strong is47/3498=1.3436 points after12 beneficial shortcuts. This is
retained evidence against an unconditional or distribution-shift claim. No stricter
sensitivity policy is substituted as primary after seeing that outcome.

## Input, lifetime and resource contract

Accepts writable contiguous uint8 buffers of the original feature width and domain.
Booleans/floating input reinterpretation, clipping and snapping are not performed.
At most65536 rows and8000000 raw elements per call. Validate every code before
writing labels. Native pointers and supplied libraries must be trusted; CRC and
SHA256 are identities, not signatures or protection against a malicious runtime.

The Python wrapper holds a buffer export and the existing native owner lease
through the call. Same-session operations serialize; reentrant closure is deferred
until the call exits. Callers must not mutate a buffer concurrently. Each call
allocates temporary input/score storage and returns fresh labels. No service-level
threading, free-threaded Python, worst-case RSS or latency guarantee is made.

`load_deployment` checks the policy hash and BOTH model hashes before loading, to
prevent accidental policy/model mixing in a trusted stable directory. It cannot
establish the calibration data's representativeness. Model loading is deliberately
not included in the warm timing. The complete model pair remains resident even
when most rows take the cheap route.

## Prior art

Cascaded classification, confidence routing and calibration are established ideas.
Learn-then-Test provides relevant multiple-testing-based risk-calibration precedent:
https://arxiv.org/abs/2110.01052
Prototype classifiers are also established, including ProtoNN:
https://proceedings.mlr.press/v70/gupta17a.html
Neither is measured as a directly implemented competitor here. This work is an
integration and a bounded benchmark, not a claim to have invented those concepts.
