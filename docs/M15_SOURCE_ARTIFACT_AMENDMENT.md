# M15 Source-Artifact Amendment — Exact Accepted M14 Checkpoints

**Status: committed after a source-materialization failure and before any M15 data generation, auxiliary fitting, development result, or confirmation result.**

The first M15 workflow attempt (`34286178758`) stopped inside `reconstruct_sources` on the first primary core. No M15 verifier, action policy, VQ model, development analysis, confirmation set, or distribution-shift set was generated. The failure therefore exposed no scientific M15 result.

The attempted deterministic retraining reproduced the frozen M14 recipe but did not reproduce the accepted tensor state under the M15 process/thread environment. The expected accepted tensor hash remains unchanged; it is **not** relaxed to the newly observed retraining hash.

M15 now uses a stronger source-identity procedure:

1. download the exact final accepted M14 Actions artifact `10077794916` from run `34280670349`;
2. require artifact ZIP SHA-256 `5729932600743b2cef19d9e0b9baf112ec75b627552af4d30a089266994a2f37`;
3. extract only the frozen source checkpoints needed by M15;
4. require accepted serialized checkpoint SHA-256 values:
   - seed 1401: `d5d4769726e3e45822e84a947dd4e8cb007a35374a8a40ae61a786c4c2ba55a4`
   - seed 2402: `b43f111af13bc8f7b667c9b7e97558b2b9743f522945193a7a625db77ea194ff`;
5. strict-load each checkpoint with the existing M14 loader;
6. require the already-preregistered tensor-state SHA-256 values:
   - seed 1401: `4cf4c93ec9d3bd688850394685924cb23d0762a8716eb3e2f42e42befd249f08`
   - seed 2402: `00a84312a4fba02e31b1954bddffe10b5933e01eaaae4226eb068490a3f97c0d`;
7. freeze every core parameter before any M15 target generation or fit.

The optional matched dim-48 FP/ternary context pair is omitted unless its exact accepted source artifact is independently supplied and hash-verified. This does not remove the mandatory M15 precision test: FP32 versus INT8 search-state storage is still run on the exact accepted FP64 cores.

This amendment changes **only source checkpoint materialization**. It does not change the M15 mechanism hypotheses, disconfirmation criteria, auxiliary training recipes, data hierarchy, verifier strengths, learned/unguided action directions, search rollout/depth budgets, terminal guard, beta grid/selection rule, VQ measurements, H1 thresholds, confirmation seed, lower-clue shift, or acceptance rule.

The failed source-reconstruction attempt is retained as provenance rather than erased.