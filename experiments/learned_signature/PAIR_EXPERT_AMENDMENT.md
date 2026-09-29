# Development amendment: a shared distance bank with pair-specific kernels

Written after the completed first training-only split (seed1401), before any
final-test predictions in this continuation. The diagonal metric underperforms;
centered alignment mostly ties the tuned single kernel. Those arms remain intact.

Add a pair-specific kernel family, a standard one-versus-one model-selection idea
implemented with shared integer signatures. The inference kernel bank contains
at most seven global radial scales; each pair stores one scale index and its own
learned SVM coefficients. Distances to shared support vectors are computed once
per request and reused across different pair kernels. No trained whole-model
ensemble is hidden outside the timer.

Selection is NESTED inside the fitting portion of each existing development split:
80/20 stratified inner partition (seed+31). Tune the 21 global C/gamma cells on
inner fitting/validation to obtain a common reference configuration. For every
class pair, tune the SAME 21 cells on that pair's inner fitting rows. Change from
the global configuration only when its inner validation gains at least two correct
predictions; otherwise retain the common configuration. Ties prefer fewer supports,
then the original grid order. Refit the selected pair on the whole outer fitting
portion. Outer validation rows do NOT select individual pair kernels.

Compare complete outer-validation multiclass decisions against all original arms.
Keep every inner trial and final pair mapping, total CPU/wall fit cost, unique
support count and coefficient count. This uses MORE fitting than one SVM: its
training cost is not claimed equal. Final model selection/fitting is frozen before
both original, previously consumed test sets are opened. No new independent test
claim and no alteration to the original failed/successful gate.

For the final whole-development model, use the same nested inner procedure with
seed1401 and all official training rows; do not use outer-test results or different
rules per dataset. The candidate can improve quality while increasing support or
storage cost; measure the actual tradeoff in the same native engine. Compiled
lookup banking is established execution organization, not a claim of being first
to use heterogeneous pairwise kernels or multiclass model selection.
