# Adaptive follow-up declared after initial validation, before its execution

The first native pilot collected 192 training formulas, 768 attempted snapshots,
591 unsolved snapshots, 8,211 candidate labels and 16,422 continuation runs. Only
208 unsolved snapshots contain unequal empirical candidate labels. It trained
three full-residual and three coarse scorers, then evaluated all six declared
arms on 32 validation formulas. No development evaluation had run at this point.

Initial validation failed to show a learned benefit. At the 4,096-move diagnostic,
probSAT-style search solved 34.375% of the repeated model/search cases; the full
learned scorer solved 30.2083%, with mean complete times 2.746 and 4.406 ms.
These are local development measurements, not a universal algorithm comparison.

The follow-up tests two hypotheses rather than erasing this result:
1. An absolute-success loss mostly spends its gradient on differences BETWEEN
   states. Learn within-state outcome ordering from the same retained training
   labels using a paired logistic loss. Uniform weighting per informative state
   prevents states with more candidates dominating the loss. Tied labels yield
   no ordering information; every tied state's original record is retained.
2. Invoking a patch model every 32 moves may cost more than it saves. Compare 32
   with 128 explicitly, preserving a no-model control and low-cost patch controls.

The native solver, candidate generator, feature definition, training data,
training seeds, number of updates and full-cost timing boundary remain unchanged.
Only the ranking loss and call interval are varied. Compare:
`probsat`, `walksat`, `random128`, `greedy128`, `bce32`, `bce128`, `rank32`,
`rank128`, and `coarse_rank128`. The primary proposed candidate is rank128; it is
not selected retrospectively from the development results. Select the strongest
nonlearned control on validation separately in each distribution/size stratum.

All follow-up outcomes are adaptively developed. Record exact source/model hashes
before development execution. Do not rename this development split confirmation.
The original source and validation observations stay frozen and independently
replayable. No legacy file or original scientific threshold is modified.
