# SPECTRA Architecture — Current Evidence Boundary

The original long implementation blueprint is preserved as `ARCHITECTURE_LEGACY.md`. It is design history, not a current evidence ledger. Current claims are governed by milestone protocols/acceptance gates and `CLAIMS.md`.

## Recurrent core and control
SPECTRA uses explicit recurrent state `(x,y,z)` with shared computation. M11 established physical early exit and faithful active-query sparsity with full K/V context. M12 established a real router/halter actor-critic training path, but its learned-control pilot collapsed; no learned-control superiority is claimed.

## Search/value semantics
M07's grounded verifier estimates **one-cycle structural-improvement probability** on its declared `(x,y,z)` distribution. It is not an eventual-solve or absolute-state-quality oracle. M15 shows that using this target-specific probability as an absolute MCTS value can cause systematic regressions even while the verifier remains predictive of its declared target. Search values must match the quantity search is intended to optimize. Exact symbolic validity remains terminal authority where an exact checker exists.

## Ensemble disagreement
Finite-member disagreement is an empirical heuristic. Without calibration/coverage evidence it does not identify OOD states, certify epistemic uncertainty, or guarantee that an LCB objective avoids proxy overoptimization. M15's validation-selected LCB partially reduces regressions but remains far below the accepted no-search M14 system.

## Precision and VQ
INT8 recurrent/search-state storage and ternary machinery have explicit correctness contracts. In M15, replacing INT8 search-state storage with FP32 leaves the observed learned-search regressions unchanged, so INT8 state distortion is not the primary mechanism in that experiment.

`LatentVQ` performs nearest-code projection. Finite codebook membership and local projection error do **not** imply a depth-independent bound relative to an unquantized recursive trajectory. M15 measures FP/INT8/VQ trajectory divergence, code use, answer agreement, and task success through depth 8. No topology, rank, isometry, or reasoning-preservation theorem is claimed.

## Stability regularizers
`model/spectral.py` contains semi-orthogonal initialization, a VICReg-style variance/covariance objective, and a stochastic Jacobian norm-preservation penalty. These are training objectives/diagnostics, not proofs of global full rank, independence, dynamical isometry, or preserved reasoning.

## Measurement and accepted task result
M13 is authoritative for performance/energy semantics: unavailable counters stay unavailable, CPU-package energy is not GPU/whole-system energy, complete-solve timing includes decode/verification, memory scopes are separated, and `1 MAC = 2 arithmetic operations`.

M14 establishes a narrow generated-4x4-Sudoku result: a trained dim-64 FP recursive model with reference-free semantic-validity early exit beats a larger trained single-pass FP baseline in strict validity at comparable **median** complete-solve latency on independent confirmation. Mean/tail latency are higher, and the exact symbolic solver remains faster and perfect.

M15 does not expand that capability claim. It explains why the tested learned-verifier latent search can make the accepted M14 system worse, retains difficult/shifted failures, and narrows unsupported VQ/uncertainty/stability theory.
