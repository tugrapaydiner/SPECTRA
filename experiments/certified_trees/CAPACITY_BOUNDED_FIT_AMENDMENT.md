# Capacity-study bounded-fit amendment

The threaded execution amendment preserved the locked candidate grid. During the
first full-grid attempt, Letter seed611 completed the control and the first two
depth-7 candidates. The next depth-8 fit did not finish within the 240-second tool
window and produced no candidate record.

To make the COMPLETE matrix reproducible and prevent one candidate from consuming
an unbounded amount of compute, rerun the entire grid in fresh child processes with:
- thread_count=8;
- a fixed 30-second wall limit PER FIT;
- the same data, grouped splits, random seeds and hyperparameters.

A timed-out fit is a retained TIMEOUT result and is not eligible for selection.
No candidate receives an extended retry after its accuracy is known. The earlier
partial attempt remains a separate diagnostic and is not pooled with the bounded
run. Selection tie-breaks among successfully completed candidates remain exactly
those in CAPACITY_PROTOCOL.md.

This amendment is recorded before the bounded complete run and before any official
test prediction from these newly trained models.
