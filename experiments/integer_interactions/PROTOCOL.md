# Integer interaction metric — prospective development protocol

Base local e0edf87eabe9ab546a44b907aa87b1e90a389aa9 / published PR38
4a47618306940962f1b15a71f7879ada5b60cf43. No old source or result is replaced.
The previous diagonal metric improves one exposed task and fails its 0.5-point
promotion gate on all three. This continuation investigates missing cross-feature
geometry, not another speed-only evaluation of those same predictions.

## Candidate and mechanism
Learn a full linear neighborhood-probability metric using only fitting labels,
with trace normalization, an identity regularizer and positive identity floor.
Quantize the learned linear map to signed small integers BEFORE SVM fitting, and
append an integer identity block so the raw features are not collapsed. Training
and deployment use exactly the same integer embedding. A matched diagonal map,
uniform map and supervised within-class whitening control isolate interactions,
quantization and the need for iterative optimization. No teacher or pretrained net.

The candidate kernel is explicitly a product of two tables: for exact integer
squared distance S and a fixed power-of-two B, H[S//B]*L[S%B]. H[k]=exp(-a*k*B),
L[k]=exp(-a*k). This is a NEW floating-point kernel definition used in fitting,
not a silently substituted bitwise evaluation of an older libm exponential.
In exact real arithmetic it is the usual PSD RBF on the integer embedding.
Finite precision has ordinary kernel roundoff; label fidelity and full pair
scores are checked against an independent scalar decoder of the new function.
The two tables avoid storage proportional to the largest squared distance.
Projection is charged per request; preparation and complete model bytes separately.

## Development and data
Use only original TRAIN partitions of Letter, Pendigits and Satellite initially.
All official test partitions have been consumed by earlier research and are not
fresh confirmation. Same-feature groups are kept together. First pilot: one
fixed grouped split (seed611), up to4000 fitting rows /1500 validation rows,
C10 and gamma2/8 (Satellite8/32), ALL arms retained. It is legitimate to reject
this candidate from this pilot; any revision must be recorded before full search.
NCA anchors512/gallery2048 maximum, same samples across diagonal/full, no self
matches. L-BFGS-B at most100 iterations, identity penalty0.02, floor0.05.
Initial integer scale8, coefficients bounded to[-31,31], identity block appended.

After pilot freeze a complete selection specification before executing it:
three grouped splits seeds611/977/1543, equal C/gamma grids per candidate,
then final fit on all original training rows with seed20260928. Models, source,
validation outcomes and controls are locked before the one final evaluation.
No tuning after evaluation. Every original task, arm, split and failure stays.
Report the extra fitting cost, not a claim of equal training budgets merely
because model selection has the same number of cells.

## Acceptance and interpretation
Success target: >=0.5-point gain versus the newly matched diagonal control on
at least two tasks and <=1.25x complete native prediction cost. Do not redefine
this gate after results; report uniform and strongest historical references too.
Passing an exploratory exposed-data gate is not independent confirmation or
broad intelligence. Training remains quadratic-cost kernel fitting, CPU only.
Benchmark raw-code to fresh outputs, all rows, batches1/32/256, randomized full
jobs, seven repeats, one core. Include original scalar exp, product-table
scalar/exhaustive and compiled paths plus a trained full-float metric control
where supported. Native controls do not inherit a Python-framework handicap.
New unrelated public data is allowed only under a separately frozen acquisition
and split protocol. Do not call repartitioned exposed data a new independent test.

Metric learning, factorized RBFs, Mahalanobis metrics and integer linear algebra
are established prior art. No first-invention or release/default claim is implied.
