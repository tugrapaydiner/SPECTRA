# M17 — Harder inputs and a second task: fixed-pool target alignment

This protocol is frozen before generating M17 data or training its maze models.
M16 established a faithful CPU implementation improvement, not search superiority.
M17 tests a separate claim: replacing an improvement-event target with an absolute
quality target improves candidate selection outside the original easy Sudoku study.

## Source identity
Use M16 scientific run 34522192590, artifact 10170185765, archive SHA256
`4e002847e476f5f30226395329ca4f28ed9d61467a079c4e58aae4694fff0adc`.
Its scientific steps and 372 fast tests passed; only its first persistence step
failed because the repository ignores `runs/`. Keep that outcome and recover the
same bytes; do not rerun to obtain a favorable source. Sudoku core and auxiliary
checkpoint hashes come from this fixed archive and are verified before use.

## Families and data
1. Sudoku shift: reuse the two accepted M14 FP64 cores and all three frozen M16
   target estimators without additional training. Generate unique 4x4 Sudoku with
   4-5 clues. Development seed 2026091703 has 0/64/128 train/validation/development
   rows. Confirmation seed 2026091704 has 256 rows. Exclude every supplied M14/M15
   manifest, both M16 manifests, and earlier M17 content/groups.
2. Maze: generated perfect 11x11 mazes, unique shortest paths, fixed corner
   endpoints, no augmentation, existing exact candidate_success checker, PATH only
   on OPEN cells and all other tokens restored from the input. Generation seed
   2026091701 has 1024/128/128 train/validation/development rows. Confirmation seed
   2026091702 has 256 rows. All exact/group fingerprints are disjoint across the
   declared hierarchy. This is not a claim of symmetry-family independence.

Maze cores: seeds 1701/2702, FP32 TRM dim48, one shared block, four heads, n=T=1,
N_sup=4, max_grid_size16, alpha_y=alpha_z=0.1. Train from scratch on CPU for exactly
1800 AdamW updates, batch32, lr0.001, weight_decay0.01, clip1.0. Supervise only OPEN
cells with weighted per-step cross entropy [0.1,0.2,0.3,0.4]; use the final checkpoint,
not best development seed/checkpoint. Log validation at steps1,600,1200,1800 without
selection. Freeze cores before auxiliary fitting. CPU training compute is measured.

Maze target models use the same TypedStateValue architecture and initialization
within each core, dim48, one block, four heads, max_grid_size16, act_bits8, include_y.
Use only the first256 training puzzles. Each has a common 16-state pool: four
identity-spine depths and four actions, identity plus three normalized Gaussian
latent directions of norm0.5, generator seed16111. Label those same states with
current strict validity, the repository's reference-free maze_score, and one-cycle
improvement of that same score. Three separate fits:180 steps, batch64, lr0.002,
weight_decay0.01, clip1.0. Initialization and sampled minibatches are identical
within a core (seed core_seed+17000). Checkpoints bind task/decode/target/core/data
identity. No reference solution enters score construction or search.

## Fixed-pool study
Same states and action order for all selectors. Retain each pool's predicted
scores, exact semantic flags, depth/action, and selection. Primary intervention is
learned current structural quality versus learned one-cycle improvement. Contexts:
learned current validity, identity depth1, identity depth4, uniform-random selection
(analytic expected success on the fixed pool), and exact first-valid selection.
Also report how an exact improvement-event oracle ranks these candidates: a perfect
predictor of the wrong event is not necessarily a correct-answer ranker.

A family opens confirmation only when development pool coverage is at least20%,
quality improves returned success by at least5 percentage points, and the lower
bound of a 95% crossed seed/common-example bootstrap for that paired effect is
positive (2000 replicates, seed17091). Freeze source/code/checkpoints and write the
selection rule before generating that family's confirmation. Same gate applies
there. Both families must pass to call this a two-family confirmed alignment result.
Otherwise retain the failed family and do not substitute easier tasks or new seeds.
The terminal-probability selector is contextual, not a post-hoc replacement primary.

## Closed-loop controls, always reported on development
All methods receive the input only. Charge embedding, every transition, decode,
exact check, evaluator call, and controller overhead. Compare native/reference
semantic exit at K4, identity continuation up to24 cycles with online checking,
checked best-first search with max_depth4, charged identity prefix4 and total24
transitions using quality/terminal predictors, a uniform-priority checked-search
control, and the exact symbolic solver (Sudoku MRV or maze BFS). This is best-first
search, not MCTS. Use randomized method order (seed17092), three timing repetitions
and per-instance/seed round medians; return all raw rows, mean/median/p95, cold setup
scope, missing energy status, task success and executed work. A better fixed-pool
selector does not establish faster complete solves or a better search frontier.

Search superiority is a separate descriptive outcome and is not required to report
a selector-alignment result. If no new solves are found, say so. If the exact solver
is faster or perfect, say so. No learned-search superiority can be declared from a
positive comparison with the deliberately incompatible improvement selector alone.

## Limits
CPU only. No API/GPU teacher, paid hardware, unverifiable energy claim, confirmation
retry, or post-confirmation tuning. Two task families with separately trained models
are a replication across tasks, not zero-shot weight transfer. This study alone does
not establish a general algorithmic advance, L7 qualification, or a100/100 project.
