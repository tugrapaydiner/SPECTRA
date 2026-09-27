# Shared/generic SVM continuation: prospective acceptance scope

Base: PR30 head 1fe050075277a4e7c060ef876517d0534f448eb6, exact tree
1ad6734bbc3fa935841047231ee02540a92ec21a. Frozen before performance measurements.
This is a local prospective protocol, not independently preregistered research.

## Questions and fixed candidates

1. Can one immutable prepared model serve independent workers without reloading or
   duplicating support vectors, coefficients, dictionaries and centroids?
2. Can the same execution support 1..4096 input features and integer/string class
   labels while retaining SPC SVM01 and the original Session API?
3. Does the first cost-aware scheduler pay for itself? It chooses the highest
   possible-win pivot, breaks ties by centroid distance and index, then chooses
   its unqueried incident comparison with minimum `missing_kernels*(d+16)+terms`.
   It checks the unchanged vote predicate after every edge. Cost-scan work is
   explicitly recorded. This is a heuristic, not query/cost optimality.

No training or model selection on the retained two datasets. Synthetic fitted SVCs
are compatibility tests, NOT benchmarks claiming broader learned capability.
The prototype is opt-in; promotion requires >=1.10x improvement over the existing
strengthened scheduler on BOTH retained tasks, same warmed native callable scope.
All regressions and full-data fidelity checks remain. Failed promotion leaves the
existing scheduler as default. No coefficient, exp, vote or tolerance relaxation.

## Measurements

Two frozen prior models: Pendigits direct and Letter exact tables. First512 retained
inputs/task, 19 repeats, fixed randomized order seed20260926. Separate scopes:
public single-call API (conversion and returned class included), native complete
512-row batch divided by512, and public512-row batch. Model/file parsing and build
excluded from warm timing. Reused input/output buffers in native scope disclosed;
public calls return fresh results. Compare original Session, shared strengthened
scheduler and shared cost-aware prototype; original and new prepared model storage
and worker scratch counts shown for1,2,8,32 workers, not labelled peak RSS.

Before every timed output is accepted, check all classes against original
exhaustive predictions. No concurrent compilation/test/training during timings.
Report median complete-batch duration per row, every repeat and work counters.
No production-tail or universal performance guarantee from this small sample.
Model load/setup and multiple worker creation separately timed, at5 repeats.

## Correctness and packaging

Check all7498 retained inputs; original and shared single/batch/certificate paths.
Synthetic dimensions1,3,16,17,64,257 and maximum4096, binary/multiclass labels,
bounded high-cardinality fallback, signed zero, float32/float64 API boundaries,
close ownership, concurrent workers, CRC/geometry/metadata corruptions, invalid
buffers/settings, ordinary schedules and cost-aware opt-in. The guarantee is
relative to computed pair signs; the certificate does not certify true labels or
exact real-valued exponential arithmetic. Thread safety requires separate worker
state or one worker lock. Numerical-library/OS portability is not inferred.

Run actual sdist-built installed wheel in a numerical-framework-free venv; native
code must compile from shipped files. Add tests to exact-head CI. Keep previous
four numerical source files unchanged; new generic arithmetic separately compared.
