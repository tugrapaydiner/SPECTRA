# M16 prior-work boundary

Literature check: September 10, 2026 (Toronto). This is a targeted primary-source audit, not a claim of exhaustive novelty clearance.

| Prior work | Relevant overlap | Consequence for SPECTRA |
|---|---|---|
| [Tiny Recursive Models](https://arxiv.org/abs/2510.04871) | Compact shared recurrent computation and learned halting | Recursion or a small parameter count is not the contribution. |
| [Probabilistic Tiny Recursive Model](https://arxiv.org/abs/2605.19943), May 2026 | Stochastic latent trajectories and selection with the existing Q head | Latent exploration plus selection cannot be described as a new field invented here. |
| [Rational Metareasoning for Large Language Models](https://arxiv.org/abs/2410.05563) | Value-of-computation objectives balance reasoning cost and accuracy | Cost-aware reasoning and continuation-value targets have established precedent. |
| [AERA](https://arxiv.org/abs/2608.27964), August 2026 | Estimates future recovery opportunities rather than treating present confidence as correctness | The distinction between present evidence and future computation value is not by itself novel. |
| [Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](https://arxiv.org/abs/2502.11880) | Optimized ternary inference kernels and mixed-precision matrix multiplication | A ternary kernel must compete on declared workloads; its existence or speedup over an unoptimized scalar reference is insufficient. |

The bounded M16 contribution is an inspectable, matched-target intervention in this specific frozen recurrent-search system: identical candidate pools and backbone fitting schedules, explicit current-versus-future evaluator semantics, exact ancestral holdout checks, a valid-incumbent retention contract, and honest complete-solve CPU comparisons. The results determine whether that package warrants a publication; software correctness alone does not.

M16 does not claim to reproduce the numerical results of these papers. Different model families, training datasets, hardware, search budgets and benchmark difficulties prevent direct comparison. In particular, generated 4x4 Sudoku is not Sudoku-Extreme, Pencil Puzzle Bench, ARC-AGI or Maze-Hard.

A serious subsequent positive result requires harder-family evidence, strong competitors, portable measurements and independent reproduction. Merely naming additional components, presenting future-success probabilities as current correctness, or treating a task-specific exact checker as learned reasoning would not establish that result.
