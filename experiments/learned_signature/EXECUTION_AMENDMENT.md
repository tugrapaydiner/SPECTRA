# Execution-only refinement after the complete first timing run

All eighteen trained models remain byte-identical to FROZEN_MODELS.json. Final
evaluation is already open; no classifier refit or family selection is allowed.
The first complete native matrix (714 jobs) is retained. Its learned-table runtime
is slower than the earlier finite runtime on these new models; it is not promoted
as an unqualified speed win.

Optimize execution, not model quality: cache the folded kernel value for a
single-profile model (rather than reloading its table on each coefficient term),
keep the cache check outside the expensive distance routine, and evaluate weighted
integer distances with 16-bit SIMD operations where the exact range permits it.
For unit-weight domains capped at127, investigate integer norm-plus-dot distance
using non-saturating unsigned/signed byte multiply-add: each pair product sum is
at most2*127^2=32258<32767. Integer norms/signatures fit the already-enforced table
range; this is not a floating-point cancellation-prone distance rewrite.

Compare final margins bit-for-bit against the already independently checked
v2 margin files before timing. Scalar/portable execution remains a control. Keep
all first-run jobs and source/binary identities. A second complete matrix will
include the prior v2 native implementation as a same-model control, not select
favorable old/new cells. No new training, quality gate, default or release change.
The failed primary learned-quality gate remains failed regardless of speed.
