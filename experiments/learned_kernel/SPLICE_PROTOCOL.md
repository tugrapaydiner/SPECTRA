# Second-domain confirmation: unchanged learned metric, 2026-09-28

Semeion results are already opened and RETAINED: baseline750/797, selected metric748/797; this is a failed quality-improvement confirmation. Letter's historically consumed test improves3880/4000 to3894/4000; Pendigits selected baseline stays3423/3498. These outcomes are not replaced by this experiment.

Apply the SAME fitted-method/search program to official UCI69 Splice-junction Gene Sequences, before accessing its rows or labels. This adds one independent application domain, not a retry of the Semeion split. The program/function, regularizations, gamma factors, C values, training-only metric anchors, integer2*d budget, uniform/aligned alternatives and linear/MLP controls are unchanged. Retain every result even if worse.

Encode each of the60 positions as a one-hot over fixed A,C,G,T,D,N,S,R (480 binary features). Preserve ambiguous symbols as their own category, not inferred labels. Reject any symbol outside this declared alphabet. Instance names and biological theory are NOT features. Positive integer metric weights can emphasize features but are learned solely from fitting labels.

Use outer50% development/50% test, stratified random_state20260928; inner80%fit/20%validation random_state20260929. Check exact duplicate sequences before fitting. If duplicates exist, keep each exact-sequence group together using StratifiedGroupKFold2 (outer) and5 (inner), shuffle=True and the same seeds, first fold as test/validation. Retain duplicate/conflicting-label records. Report actual sizes and split indices. These partitions are NOT gene-family-, species- or homology-disjoint and do not establish biological/clinical applicability.

Use training validation only for choices, freeze all five kernel model bytes and both classifier-control checkpoints before any final-test prediction. No retraining on fit+validation, error-based retuning, changed regularization, hidden data augmentation or test-selected ensemble. Select richer family only on strictly greater validation accuracy than the strongest selected single kernel; ties keep the simpler baseline. Record all frozen controls on test, not only the selected winner. No new hypothesis or experiment will be selected after this test within this continuation.

Report raw correct counts, confusion, paired changes/intervals, all search CPU costs and same-function native direct/compiled timing. The quality gate requires a positive held-out selected-vs-baseline result and paired interval excluding zero; either failed task remains failed. Small-MLP and linear controls are mandatory context; their own numerical/runtime contract must be separated. A global intelligence, world-first, state-of-the-art or medical-use claim does not follow.

Official source: https://archive.ics.uci.edu/dataset/69/molecular-biology-splice-junction-gene-sequences ; CC BY4.0, DOI10.24432/C5M888. Data acquisition and source hash are separate from trained results.
