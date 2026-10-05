# Structured-search development declaration — 2026-10-05

Question: do unit propagation and reversible search decisions improve SPECTRA's
ability to solve externally sourced structured tasks under a bounded search
budget? The current focused policy improved planted random SAT but leaves a
transfer gap. Begin by measuring its failures on the first ten public top95 Sudoku
puzzles, in upstream file order. Candidate: opt-in dependency-free watched-literal
DPLL, iterative decision trail, shortest unresolved clause branching, no learning.

External source: Peter Norvig's pytudes, commit
`bfcdac7e2c74f44a9c2ff36217a8dd834ad6b73b`, `py/sudoku-top95.txt`; retain attribution,
raw bytes and MIT license. Direct control: unmodified upstream `py/sudoku.py`.
Native control: Glucose4 through pinned python-sat 1.9.dev15, with counters and
budget overshoot visible. Include existing focused and deductive solvers.

Rows 1–10 are development; rows 11–95 are held out from local solver execution
until candidate/encoder/runner/analysis/protocol publication. Public puzzles are
not claimed to be hidden, novel, representative of industrial SAT or independent
of an author's prior exposure. Do not tune after the held-out run. Original
SPECTRA Sudoku4/maze models, sealed inputs and earlier evidence remain untouched.

Use a fresh experiment module and new optional solver module, leaving existing
evidence-bound runtime files unchanged. Do not alter public exports used by the
previous frozen study: expose the new solver from its own module. Measure encoding,
solve and independent original-grid checking as part of complete task cost; also
report solve-only and encoding cost separately. Strong controls include native SAT
and direct-domain constraint propagation/search, even if they dominate the candidate.

Initial DPLL development budget: 2,048 branch attempts. Existing focused/deductive
arms: 2,048 flips, seed 17. Native: 2,000 conflicts requested. These are different
work units; compare full measured cost without calling them equal compute. Retain
every development attempt and failure. Freeze the final candidate and operating
point before measuring the remaining 85 inputs.
