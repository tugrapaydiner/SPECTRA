# Wider-kernel control correction before official test opening

All248 original training-only selection fits completed. No new-model official test
prediction or OptDigits test parsing has happened. On the64-dimensional optical
features, fixed prototypes chose the lowest available gamma2 but achieved only
94.77% pooled validation accuracy, versus98.95% for local metrics and99.48% for
SVC. The SVC grid included gamma.125; the prototype grid did not. This creates a
plausible width-range handicap, so it must not be marketed as a learning advantage.

Add gamma.125 and.5 to EVERY optical prototype family at EVERY original budget
P256/512/1024, on BOTH original grouped splits611/977:36 additional fits. Nothing
is dropped or rerun selectively; preserve the initial248 and all36 added cells.
Training code, epochs, regularization, quantization and other tasks are unchanged.
Reselect optical fixed/centers/local using all12 choices per family. Tie rules
remain smaller prototype count, then the original gamma order2/8 followed by.125/.5.
All other families retain their previously selected settings. Record an amended
selection lock before the24 final refits, then freeze every final artifact before
any official evaluation. No test-driven correction or claim of equal tuning cost
between model families. The original primary quality/cost gate is unchanged.

This amendment improves the comparison, not evidence of novelty. Constant pixels
and variable bandwidth can affect the meaning of a local metric, so interpret any
remaining gain against the strengthened fixed and moving-center controls.
