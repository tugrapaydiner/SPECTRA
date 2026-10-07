# Development checkpoint 02 — stronger controls, not promotion

The reversible-state candidate now has exact XOR unit identification, optional active
or heap priority structures, bounded least-conflicting-value scans, and static-degree
MRV tie-breaking. These are established mechanisms, not invented techniques. The
original backend, defaults, previous results and sealed observations remain unchanged.

The state payload now charges assumption conversion. Native Python varargs use explicit
unsigned-long-long casts. Build payload counts the actual reusable validation/cursor
bank. These are bounds on counted native payload, not Python objects, allocator spare
capacity, output storage, process RSS or wall time. Native ABI 3 rejects older binaries.

Local checks: 524 public tests PASS, including 57 sparse contracts; all 57 sparse
contracts PASS under UBSan. Those repetitions are not extra unique tests. Exhaustive
small systems, randomized assumptions, exact priority-structure path agreement,
original witness checking and resource/ownership tests are included. No new installed
wheel or external-host validation is claimed by this development checkpoint.

## What development learned

All 229 old PR54 cases are exposed development/compatibility, never fresh confirmation.
The first full API experiment retains 3,435 observations; the priority-structure study
11,221; the degree/value-policy study 9,618. Active sets/heaps often regress; extra
complexity is not promoted merely because it is implemented. A single planted graph
caused a long original search path. Static-degree ordering greatly reduces that path.

The new one-call Python extension must not claim an algorithmic win merely by replacing
PySAT's many Python calls with one bulk call. A separate pinned, UNMODIFIED MiniCard
core at 79776615ddc8803dd86803ea6f00838fc74349b1 was compiled with a bulk tuple adapter.
Its original OPB regression program passes 352,620 assignment checks under ASan+UBSan.
Fourteen adapter contracts pass, including exhaustive formulas and random assumptions.
An initial adapter confused identical bank object identities with semantic roles;
one regression failed before the fix. The core was not changed, and no timing from
that invalid adapter is used. All original sources, failure and fixed tests are retained.

The complete seven-arm bulk-control study retains 11,221 rows (229 cases, seven rounds).
It times original constraint tuples through fresh preparation, solving, witness creation,
original-constraint checking and disposal. Raw task encoding is OUTSIDE this API timer;
independent domain checking follows it. This is not the complete raw-domain application
boundary and cannot replace PR54's full-pipeline result.

Mean microseconds, same complete matrix:

| Panel | SPECTRA first | SPECTRA degree | SPECTRA degree+LCV | Bulk MiniCard | PySAT MiniCard |
|---|---:|---:|---:|---:|---:|
| Exposed Sudoku |459.490|463.222|457.599|571.335|862.087|
| Exposed planted graphs |176.194|81.248|106.557|128.613|229.324|
| Exposed uniform graphs |219.624|79.819|91.841|115.251|215.590|

All arms solve all Sudoku/planted inputs. Uniform graphs retain the same 22 SAT tasks;
other native results are unverified UNSAT reports versus candidate UNKNOWN. Repeated
rounds are not independent problems. Slow outliers are retained. These are descriptive
development results without confirmation intervals or threshold promotion. Bulk
MiniCard substantially shrinks the apparent advantage. The demanding 2x gate is not
established. Degree+LCV has a large observed wall-time outlier; do not hide it by choosing
its median. No new confirmation inventory has been generated or opened.

## Recovered controls and remaining work

The original Knuth DLX1 and SSXCC sources and compiled executables were acquired.
The third acquisition job reached a valid first solution but failed because the old
C90 main's unspecified fall-through status was nonzero. This is not a solver error
or SPECTRA win. A documented entrypoint compatibility wrapper must preserve explicit
errors and original search before using these controls. OR-Tools 9.15.6755 and
DRAT-trim 2e3b2dc0ecf938addbd779d42877b6ed69d9a985 are available for stronger comparisons
and proof investigation, not silently counted as tested baselines.

The native-control archive omitted one hidden source entry. It was recovered from
the separately retained original tarball, then all 223 manifest entries verified.
The prior dependency acquisition failures remain. Next: complete strong-control
adapters, reduce actual execution/verification cost, and test residual structure
without opening confirmation before the whole tested system and analysis are frozen.
