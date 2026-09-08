# M15 Protocol Clarification — Checkpoint Materialization

**Committed before any M15 auxiliary training or M15 result.**

`M15_PROTOCOL.md` names the exact M14 source-checkpoint file and tensor-state hashes. To avoid committing duplicate binary checkpoints into the Git tree, M15 materializes the source models by rerunning the already-frozen M14 training recipe/data/seeds and requires the reconstructed **tensor-state SHA-256** to equal the accepted M14 tensor-state hash before any M15 auxiliary fit or evaluation proceeds.

The original M14 serialized-file SHA-256 remains provenance for the accepted artifact, but byte-identical `.pt` reproduction is not required because the serialized payload includes run/Git metadata and serialization details unrelated to parameter identity. No M15 result is accepted unless the exact expected tensor state is recovered.

Primary required tensor-state identities:

- FP64 recursive seed 1401: `4cf4c93ec9d3bd688850394685924cb23d0762a8716eb3e2f42e42befd249f08`
- FP64 recursive seed 2402: `00a84312a4fba02e31b1954bddffe10b5933e01eaaae4226eb068490a3f97c0d`

The matched dim-48 FP/ternary context check is attempted only if its accepted tensor states can likewise be reconstructed exactly; failure to reproduce those contextual states removes that optional context ablation rather than substituting a different checkpoint. The primary precision ablation (FP32 versus INT8 search-state storage on the same accepted FP64 cores) remains mandatory.

This clarification changes no hypothesis, threshold, data split, search budget, verifier recipe, action recipe, or confirmation rule.