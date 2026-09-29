# Complete cost experiment, fixed after quality evaluation

All eighteen models have been frozen and evaluated. No change to trained weights,
selection, projection rule, kernel definition or native numerical source is permitted
from these results. The full-NCA primary is worse than the diagonal control on all
three official tests. The local covariance secondary reaches3929/4000 on Letter;
its no-label-neighbor control reaches the same count. Both are worse on the other
tasks. This is not evidence that supervised labels alone explain the Letter gain.

Fourteen arms, all three tasks, batches1/32/256, seven shuffled repetitions (882
cells). Six newly fitted families, local scalar/exhaustive/direct-exp diagnostics,
forced generic projection for the diagonal control, prior uniform/diagonal metric
executors and their original frozen linear/MLP controls. Recompile the six native
linear/MLP controls and previous metric executor on this host before timing. Every
arm receives original uint8 feature codes and returns fresh labels. The candidate
pays for feature projection. All expected outputs are model-specific; direct-exp
uses its separately observed expected outputs even if it differs from the product.

The diagonal/uniform representation has an exact model-only diagonal recognition
path; do not handicap those baselines with avoidable duplicate projection work.
No neural-framework dispatch or query-kernel precomputation is hidden outside the
request timer. Preparation, compilation, feature extraction before supplied integer
codes and model loading are excluded and separately scoped. There is no newly
compiled m2cgen or upstream LIBSVM comparison for these changed product kernels.

Keep all882 cells, regressions and both failed/passed point requirements. The
original primary requires >=0.5 points and <=1.25x diagonal cost on at leasttwo
tasks. Secondary results do not substitute for that requirement. No producer
throughput, energy, tail latency or new independent accuracy claim follows.
