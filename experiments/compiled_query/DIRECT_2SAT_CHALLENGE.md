# Direct 2-SAT challenge

**Branch:** `direct-2sat-challenge`  
**Base:** Part 3 commit `65dc9ff1f7c32e3528e1635af58a4f761ef514af`  
**Current stage:** independent engine and challenge matrix implemented; results pending.

## Why this challenge exists

SPECTRA has beaten persistent MiniCard, persistent CaDiCaL and reduced CUDD on its
narrow repeated-query WAP workload. That is meaningful, but the admitted exact
problem has one or two colours per original vertex and therefore has a direct
2-SAT formulation. The strongest missing comparison is a purpose-built 2-SAT
service with the same ready-input, answer, proof, checking and accounting
contract.

The existing SPECTRA `raw` mode is not independent enough to close this question:
it shares the package parser, native source, proof machinery, observer and result
transport. This challenge uses a separately written C++ library and a plain C ABI.
It imports no SPECTRA native solver code and consumes only the public `SBC1` bytes.

A negative result is useful. If the independent service is faster, that result
must remain visible and SPECTRA must improve rather than weaken the comparison.

## Matched contract

Both systems receive the same immutable `SBC1` packet and must return `SCQ2`.
Every admitted completed query returns either:

- a complete original-vertex colouring; or
- an `SBP1` contradiction proof replayable from original inputs.

The challenger includes its own independently implemented packet parser, solver,
proof constructor, output serializer and original-input observer. Timed calls
include construction, all decisions, complete labels or proofs, internal checking,
serialization and disposal. Build time and later Python replay are disclosed
separately for both systems.

No timeout, allocation failure, malformed input or unsupported platform may be
converted into UNSAT or omitted from a result table.

## Challenger engines

The independent service computes the original implication graph and SCC
condensation from scratch.

- **closure:** when bounded by the caller's memory cap, precompute component-DAG
  reachability as dense machine-word rows. A query unions the rows of its forced
  roots and tests complementary components.
- **dag:** otherwise propagate over the condensation DAG with query-local state.

SAT assignments are reconstructed in original colour space. UNSAT queries produce
fresh original-literal paths. A second implementation reparses the packet and
checks every returned label, edge, restriction and proof hop before output leaves
the library.

This is intended to be a strong competitor. The dense closure is expected to be
especially dangerous on the highly compressed WAP instances.

## Development sequence

1. Exhaustively check every three-vertex graph, every one/two-colour domain over
   three colours and every per-vertex restriction.
2. Differentially test random mixed SAT/UNSAT banks against brute force and the
   public SPECTRA exact service.
3. Run Linux x86-64, native ARM64, UBSan and ASan+UBSan contracts.
4. Measure exposed synthetic cases at 1, 8, 64 and 1,024 queries using
   position-balanced schedules and calibrated batches.
5. Challenge the already exposed WAP05/06 banks with complete first-use costs.
6. Only after implementation and thresholds are frozen, open genuinely unseen
   mixed-result topology/workload banks.

The synthetic and WAP05/06 measurements are development diagnostics. They cannot
be promoted into new confirmation evidence.

## What would count as a competitive win

The eventual prospective study must predeclare source identities, packet banks,
hardware classes, resource limits, schedule, repetitions and effect-size gates.
SPECTRA may claim a direct 2-SAT win only if it is faster under the full matched
contract while every decision and proof replays successfully. One-query and long-
bank results must be reported separately; a sustained-query win cannot be
advertised as universal latency leadership.

This challenge does not establish general SAT, ordinary graph-colouring,
asymptotic or production superiority.