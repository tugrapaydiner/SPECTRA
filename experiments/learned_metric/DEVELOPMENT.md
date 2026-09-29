# Development chronology

The first pilot used only seed611 training/validation and fixed C10/gamma2.
The smooth large-margin objective collapsed several Letter feature weights to0:
1783/2000 versus uniform1878/2000. Pendigits improved only1495→1496/1499.
The entire six-fit result and source commit are retained, not promoted.

Before running the complete selection sweep or reading evaluation predictions,
replace the candidate with an NCA-style probability objective on a fixed training
gallery, using positive diagonal weights bounded .5..4 and fixed total mass.
This learns from the complete sampled neighbor probability rather than the first
near-impostor triplets and does not drop original features. The exact integer
weights, not the relaxed values, define the classifier trained afterward.
Neighborhood component analysis is established prior art, not claimed invented.
Reference: Goldberger et al., NeurIPS2004.

All old aggregate outcomes from the interrupted pre-delivery continuation are
excluded from numerical evidence because its model/implementation was unavailable.
These benchmarks were consumed before; no new independent holdout claim is made.

The complete162-fit selection sweep finished before final fitting. Letter selected
C10/gamma8 for all three arms; the NCA validation mean exceeded uniform by0.7833
percentage points. Pendigits selected different equal-budget settings and had no
NCA validation gain. All six nonlinear and four simpler models were saved in
FINAL_LOCK.json before evaluation. Native linear/MLP export and an original-source
binary64 observer were then fixed before model evaluation. The observer evaluates
repeated integer coordinates with unchanged original RBF arithmetic: it shares no
learned table/signature code and compares every natural pair margin, not only labels.

Evaluation found3911/4000 vs3928/4000 on Letter and3435/3498 vs3432/3498 on Pendigits
(uniform vsNCA). No fit or model selection follows those predictions. The initial
0.5-point quality threshold is NOT changed to make the0.425-point Letter gain pass.
Complete-call timing now fixes eleven arms, all test rows, three chunk sizes and
seven shuffled repetitions. The historical model has its own saved expected labels;
its smaller original training set/settings are not the matched learning baseline.
Native controls compute all their layers from original codes, and sklearn is charged
for constructing query kernels within each timed invocation.
