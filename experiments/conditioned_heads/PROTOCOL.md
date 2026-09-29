# Exact-feature head conditioning: prospective experiment

Parent c5af3032e1f9dd8908e922836dd7c06d4222b876 (PR40). The final CI archive is present
and byte-verified. The previous local auditor copy, complete prototype evidence,
and 24 prototype model files were not in recoverable attachments or Library
search results. Hence this is a NEW run, not verification or continued fitting of
those previous model bytes. The historical mismatch remains unclassified.

Question: does the final exact-feature softmax fit stop before exploiting its
learned features because the features are highly correlated? Test invertible
centered Cholesky conditioning, with the original L2 head penalty transformed
correctly. It changes optimization coordinates, not the mathematical objective,
prototype geometry, head shape, model format, or inference algorithm. No training
whitening transform is shipped: fold it back into the original head/bias.

Pilot uses training roles only: original Letter/Pendigits/Satellite train and
OptDigits original train, fixed duplicate-grouped split seed611. At most4000 fitting
and1500 validation rows. P256, gamma8 (Satellite32, OptDigits0.5), local prototype
learner, quarter4/units4, 80 epochs, reg1e-6, seed611, no final head. Compare the
original150-iteration L-BFGS head fit, a longer ordinary raw-coordinate fit300,
and a Cholesky-conditioned fit150. Each starts from the SAME geometry and head.
Also compare additional150 iterations from the original head versus conditioned
150 iterations from that exact common start. Report all conditions/objectives,
original-coordinate gradients, CPU/wall time, full validation counts and solver
terminations. No validation is read by the head optimizer. Include preprocessing
cost in training cost and numerical pullback checks.

All four official tests were already consumed in PR40. They stay unopened by this
round until the complete continuation matrix and all final artifacts are fixed.
No fresh-holdout claim. No teacher, pretrained model or GPU. The pilot may reject
the hypothesis; record complete runs and changes before further model evaluation.

A final run, if justified, will be locked separately before execution. Original
prototype/native files remain unchanged. Preconditioning is established numerical
optimization, not a first-invention claim or a guarantee of better generalization.
Positive result must survive the longer ordinary optimizer, not just a150-iteration
straw baseline. No high-90s score assigned from test counts.

Every NEW result binds the exact executed script/source hashes and artifact bytes.
Evidence packs include actual models and a standalone verifier; summary JSON alone
must not be called a reproduced experiment. Prior audit source mismatch is recorded
as unresolved unless the actual missing local version is recovered.

Implementation-stage control (before first pilot): also include centered diagonal
standardization at150 iterations. It uses the same objective and starting logits,
so Cholesky must earn its cost against a cheap feature-scale-only alternative.
