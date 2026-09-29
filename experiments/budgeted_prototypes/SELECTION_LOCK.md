# Complete selection lock — September 29, 2026 UTC

Six training-only pilot panels are retained, including unsuccessful affine heads,
variable-width responses, normalized responses, finer quantization and late-iterate
averaging. None used a new official test prediction. The strongest Letter pilot
was the simpler constant-head local-metric model with1024 prototypes, gamma8,
quarter-grid4, integer mass4 per feature, no response normalization or averaging.
Do not attribute improvement from the1e-4 to1e-6 head penalty to local geometry.

## Fixed candidate matrix

Tasks: original Letter, Pendigits, Satellite training partitions and newly acquired
original OptDigits training partition. All previous tasks are exposed research
benchmarks. The OptDigits official test remains unparsed. Exact duplicate feature
rows are grouped before selection. Use two first-of-five grouped splits with
seeds611/977, fit cap6000, validation cap2000. There is no third repeated split
secretly selected after results. Each family gets P256/512/1024 and gamma2/8,
except Satellite gamma8/32. Families fixed, moving-centers, local-centers-and-metric
are all kept:144 complete selection fits. Pick pooled validation accuracy, then
smaller P, then listed gamma. Every selected family is refitted on all original
training data with seed20260929, then the output head is refitted on exact product
kernel features. No final-test early stopping, selection or repeated retries.

All prototype runs use200 epochs, batch256, Adam initial head lr.03/geometry lr.01,
head penalty1e-6, local-metric identity penalty.001, centers quarter-grid4,
positive per-prototype integer weights of total4*d, fixed global gamma. Geometry
uses the documented straight-through proxy during training, not the exact table's
derivative. Quantized geometry is frozen before exact-feature float64 head fitting
(L-BFGS-B150 iterations). Cap exits/warnings are reported, not called convergence.
No affine output, variable gamma, response normalization or averaging is promoted.

## Controls

Each task and split also evaluates RBF SVC on q/maximum: C1/10/100,
gamma.5/2/8 (OptDigits.125/.5/2),72 fits. Final models use the same normalized
inputs, not label-aware feature engineering. LinearSVC C10 is a fixed lower bound.
A properly scaled MLP control uses StandardScaler fitted on its fit rows, then
ReLU128x128 or256x256, alpha1e-5/1e-3, Adam lr.002, batch128,
max_iter400, early_stopping10% internal, patience30,32 selection fits. Search
opportunities and actual training cost differ across families and are disclosed.
The MLP's own early-stopping partition lies inside the training role.
There are four tasks times six final families (three prototype, SVC, MLP, linear)
=24 final models. Use the actual manifest count, not an inferred test count.

## Final evaluation and timing

Save chosen settings and hash all24 final models/source before parsing OptDigits
official test or predicting any official test in this round. The historical
Letter/Pendigits/Satellite tests remain exploratory, not fresh confirmation.
Evaluate every selected family once, including regressions. Also preserve previously
reported stronger SVM results as historical references, not matched current timing.
Primary gate stays >=.5 points vs matched fixed prototypes on two tasks, <=1.25x
native raw-code-to-label cost. The practical frontier against new SVC/MLP controls
is a separate criterion. A passed internal gate cannot imply superiority to ProtoNN
or all compact classifiers; unmeasured prior systems remain explicit.

Warm native timing includes input validation/normalization or scaling, all kernels,
head layers and new labels. Exclude loading/compilation/table preparation and measure
them separately. Three batch sizes1/32/256, seven shuffled whole-dataset repetitions,
one pinned CPU, no concurrent fitting. Mean whole-job cost is not service tail
latency. All timed outputs must match their own frozen model. Runtime order and
FP64 score observer are checked independently. No production default or release.
