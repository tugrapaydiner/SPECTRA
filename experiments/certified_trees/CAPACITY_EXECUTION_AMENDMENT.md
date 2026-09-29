# Capacity-study execution amendment

The prospective candidate/model grid in CAPACITY_PROTOCOL.md is unchanged.
A single-thread Letter control fit (256x6, seed611) took about 12.4 seconds.
Before the complete candidate matrix, the same control was rerun with
thread_count=8 and produced the identical 3200-row validation prediction SHA256
d6b05551bfa2881b1fbef2925d495329129ba54565513e7dd656dab3470b2920
and the same 2982/3200 correct count, while completing materially faster.

Use CatBoost CPU thread_count=8 for the COMPLETE locked grid and final refits.
No candidate hyperparameter, split, tie-break, test boundary, accuracy gate or
execution benchmark changes. Training wall/CPU costs are reported with this
thread count and are not compared as if they were single-thread training costs.
The compact inference benchmark remains single-core.

This amendment is about making the predeclared experiment executable in the
current environment; it was recorded after one control equivalence check and one
deeper-candidate timing sanity check, before the complete 72-fit selection grid.
The observed sanity candidate is not removed or privileged; it is rerun normally.
