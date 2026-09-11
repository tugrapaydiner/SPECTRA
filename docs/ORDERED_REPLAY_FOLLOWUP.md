# Follow-through: all maze core matrix reductions

This supplements [the original trace diagnosis](ORDERED_REPLAY.md).

## Cross-host follow-through: fixing the first divergence was not sufficient

At commit `b15962e558aaa2d382357730556fc16004318cff`, CPU run
`34562438770` on Intel passed all four Sudoku pool hashes, then failed the maze
1701 hash (`f2d34157...` versus the unchanged `fbc29af0...`). This rejects the
assumption that the first-cycle attention discrepancy was the only cause.
The failure artifact is `10184864050`, ZIP SHA256
`2ead2ee3d30cd9d0512b1a557f12669acdcdf0f3d3e3e4d23cc8e5240d09ff1a`.

The completed profile also makes the maze core's remaining linear contractions
explicit: packed QKV and output projections (K=48), FF1 (K=48), and FF2 (K=192).
Each uses one increasing-k FMA sum with bias added afterwards. On the unchanged
AMD reference, direct comparisons over the first four depths and every action
agree exactly for 22,302,720 QKV values, 7,434,240 attention projection values,
29,736,960 FF1 values and 7,434,240 FF2 values. The tested 96/128 blocking
alternatives do not reproduce FF2. These are internal value counts, not new
independent tasks or new scientific confirmation.

The output answer head remains the original operator: this profile must still
pass the independently decoded labels and every original full-pool hash. The
per-instance context now restores all three maze module overrides on exceptions
and rejects a preexisting override before installing any. The completed local
full replay reproduces every original pool hash and all 49,152 scores exactly.
Final CPU CI must still pass on the actual merge candidate.
