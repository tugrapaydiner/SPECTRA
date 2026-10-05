# Research engineering case study

SPECTRA studies a practical question: **how much verified task success is obtained
for the complete CPU cost of a reasoning system?** Its strongest contributions
are executable experiments, exact state maintenance and explicit deployment
contracts. The bounded CNF work is classical search; the separate historical
neural and SVM tracks keep their own evidence and limitations.

## Start with one complete investigation

The [focused-search study](../experiments/focused_evaluation/RESULTS.md) follows a
single question from implementation to a frozen comparison and a measured
decision. Read it before the older, broader research archive.

| Engineering question | Inspect | What a reviewer can check |
|---|---|---|
| Can a different move policy improve verified solving? | [Frozen protocol](../experiments/focused_evaluation/PROTOCOL.md) | Candidate selected before fresh inputs; explicit admission conditions |
| Is the effect bookkeeping or search behavior? | [Focused implementation](../spectra/cnf/focused.py) and [indexed control](../spectra/cnf/indexed.py) | Dense-pool and policy ablations; no changes to historical defaults |
| Are successes real? | [Independent checker and arm adapters](../experiments/focused_evaluation/study.py) | Full Boolean assignments checked against original signed clauses |
| Does the cost include setup and verification? | [Runner](../experiments/focused_evaluation/run.py) | Whole-call wall/CPU timing; separate memory workers and native setup/destruction |
| Is the conclusion supported by the observations? | [Analysis](../experiments/focused_evaluation/analyse.py) | Complete inventory, paired formulas, uncertainty, unfavorable controls and missing-data rejection |
| Does installed code work without the checkout? | [Installed probe](../scripts/check_focused_installation.py) | Dependency-free wheel, exhaustive small fixtures and CLI solve/check |

## The design decisions

**Exact incremental state.** Each clause keeps a true-literal count and an XOR
of its true variable identifiers. An XOR identifies the sole supporter only
when the count is one. Maintaining break counts through touched incidences avoids
rescanning all clauses for every candidate flip. Literal-rescan tests check the
cache after each mutation, including repeated literals and tautologies.

**Data structure and trajectory are coupled.** The indexed solver uses a ranked
set to reproduce the historical clause-selection order. Swap-delete sampling is
constant-time but changes that order. Therefore a dense pool must be evaluated
as a different stochastic trajectory even with the same variable weights; a
faster data structure does not establish greater solve quality. The `poly` arm
isolates this combined bookkeeping/order change from the age-policy change.

**A small, inspectable policy.** The selected policy ranks clause variables by
break count and last-flip age, with a fixed coin rule to avoid repeatedly choosing
the youngest nonzero-break variable. It is a break-only age heuristic, not the
full original Novelty algorithm. Dense `minbreak` is a simpler policy control.
All local-search arms share a total flip budget. Native conflicts are a different
unit, so the native solver is reported with complete costs and observed counters.

**Two kinds of reproducibility.** Retained byte/witness verification can be run
without optional packages and does not regenerate historical latency. Rerunning
the frozen implementation on the retained inputs measures a new host/run.
Neither repeated seeds nor timing rounds create new independent formulas.

**Failure is an experimental outcome.** The earlier focused run's raw evidence
was lost; its summary stays explicitly unverified. The fresh study uses new
declared inputs and cannot retroactively repair that evidence. Native UNSAT
answers are labeled as reports because no proof checker is attached. UNKNOWN
cases stay in cost denominators. Historical negative neural/SVM results remain
in the [research review](RESEARCH_REVIEW_20260911.md) and
[native comparison](../experiments/native_baselines/RESULTS.md).

## Reviewer walkthrough

1. Read the study's question, predeclared gate, complete result table and limits.
2. Run its standard-library archive verifier. It checks raw evidence and recomputes
   the report rather than trusting a screenshot or README number.
3. Trace one solve through `PreparedCNF`, `_BreakState`, the dense pool and the
   independent original-clause check. Explain which invariants permit each update.
4. Compare pooled results with every family/size cell, the policy controls and
   the native baseline. Explain why seed/round replication is clustered by formula.
5. Build the sdist/wheel and run the installed probe outside the checkout.

Development provenance: the new implementation, experiment harness and
documentation were produced with AI assistance. The executable artifacts and
retained raw evidence are available for independent technical review.

## Scope as an AI research engineering sample

The [structured-task follow-up](../experiments/structured_search/RESULTS.md)
addresses the external-input gap with public 9×9 Sudoku. A new optional watched
DPLL implementation solves 85/85 evaluation tasks, versus focused's 0/85 and
deductive's 1/85. Mean complete wall cost falls from focused's 64.53 ms to 40.54 ms.
The independent native SAT and original direct-domain controls solve all 85 faster
(16.11 and 6.09 ms). The follow-up retains development attempts, source publication
before evaluation, full-cost timing, separate memory observations and 340 exact
Python replays. It demonstrates a bounded algorithmic capability addition;
it does not imply neural learning or competitive general-purpose SAT performance.

This investigation demonstrates hypothesis definition, algorithm implementation,
measurement, failure analysis, regression tests and reproducible delivery on a
small CPU workload. These are relevant parts of research engineering: current
[OpenAI evaluation roles](https://openai.com/careers/research-engineer-frontier-evals-and-environments-san-francisco/)
and [Anthropic evaluation roles](https://job-boards.greenhouse.io/anthropic/jobs/5198255008-62)
emphasize experimental design, trustworthy measurement and engineering execution
(reviewed 2026-10-05). Those roles also involve model/production work beyond this
study; this mapping is an interpretation, not an employer assessment of SPECTRA.

The next substantive gaps are diversity beyond this one external task family,
demonstrated training/generalization improvements, profiling beyond this CPU host
and evidence of production-scale operation. These studies supply none of those by implication.
Any follow-up should declare a new question and fresh evaluation before tuning.
