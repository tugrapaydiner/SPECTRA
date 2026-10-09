# SPECTRA

### Certified repeated-query compilation on CPU

SPECTRA is a CPU-first research toolkit. Its flagship research result is a
**certified original-address compiler for repeated list-colouring support
queries**: compile a fixed conflict graph once, then answer many restrictions
addressed to the original vertices while returning and independently checking a
complete original-graph witness.

On five prospectively unopened WAP optical-conflict graphs under the frozen
Python 3.13 contract, SPECTRA used **36.35% of persistent native MiniCard’s mean
complete 1,024-query session time** (approximately **2.75× lower latency**). The
exact whole-graph 95% ratio interval was **34.68–37.29%**, the p95 ratio was
**36.47%**, all **107,520 full answers** passed original graph/list/query checks,
and SPECTRA won on every holdout graph.

The complete 2× gate reproduced on two additional Python 3.13 server environments,
including Ubuntu 22 and Ubuntu 24. It did **not** reproduce under Python 3.11:
both Python 3.11 runs retained exact semantics and per-graph wins but reached
complete ratios of 0.529–0.534 because the frozen canonical evidence-decoding
front end added a large common cost. This failure is retained and the threshold is
not relaxed.

This is a narrow result—not a claim that SPECTRA beats SAT solvers or graph-colouring
systems generally, not an ordinary chromatic-number result, and not evidence for
historical “general reasoner,” MCTS, low-bit, or neural claims.

[Confirmed result](experiments/wap_support/HOLDOUT_RESULTS_20261009.md) ·
[Paper draft](experiments/wap_support/PAPER_DRAFT.md) ·
[Exact protocol](experiments/wap_support/PROTOCOL.md) ·
[Gate ledger](experiments/wap_support/FLAGSHIP_GATE_LEDGER.md) ·
[Novelty boundary](experiments/wap_support/NOVELTY_AND_CLAIM_BOUNDARY.md) ·
[Prior-art audit](experiments/wap_support/PRIOR_ART_AUDIT_20261009.md) ·
[Cross-environment results](experiments/wap_support/CROSS_ENVIRONMENT_REPRODUCTION_20261009.md) ·
[Certificate reproduction](experiments/wap_support/CERTIFICATE_REPRODUCTION_20261009.md) ·
[Replication guide](experiments/wap_support/REPLICATION.md)

**Package: Python 3.10+ · Flagship complete-call result: Python 3.13 · MIT-licensed project code · CPU-first · explicit verification boundaries**

## Flagship result

### Problem

A fixed graph represents conflicts between lightpaths. Each original vertex has a
small allowed-colour list. A sequence of queries further restricts arbitrary
original vertices. Every successful query must return a complete colouring that
satisfies all original edges, lists, and current restrictions.

### Contribution

SPECTRA:

1. builds the binary implication relation;
2. merges strongly connected, implication-equivalent choices;
3. retains an exact mapping from every original vertex and colour to the quotient;
4. answers restrictions expressed in the original address space;
5. lifts every answer back to a complete immutable original-vertex colouring;
6. checks every answer with a separately owned original-input observer; and
7. optionally exports an integer-only whole-relation certificate for an independent
   standard-library verifier.

Strongly connected components, equivalent-literal substitution, incremental SAT,
and offline/online knowledge compilation are established foundations. SPECTRA’s
claim is the exact end-to-end compiler/runtime/certificate contract and its frozen
complete-cost result—not invention of those individual techniques.

### Frozen holdout

| Quantity | Result |
|---|---:|
| Prospectively unopened graphs | 5 WAP-A graphs |
| Complete persistent sessions | 105 |
| Full answers audited | 107,520 |
| Queries per session | 1,024 |
| SPECTRA mean complete session | 569.290 ms |
| Native MiniCard mean complete session | 1,566.119 ms |
| Mean ratio | **0.363503** |
| Exact graph-clustered 95% interval | **[0.346789, 0.372944]** |
| p95 ratio | **0.364735** |
| Exact graph-clustered 95% interval | **[0.338157, 0.383226]** |
| Per-graph candidate wins | **5 / 5** |
| Frozen verdict | **PASS** |

Complete session time includes immutable case decoding and validation, fresh
formula/index and solver construction, all 1,024 queries, full answer
materialisation, independent original checking, diagnostics, and disposal.
Process launch/import is retained separately as cold wall time. Graph acquisition,
case generation, native build, evidence transport, and the optional independent
compiler-certificate audit are disclosed separately.

The availability lists and query restrictions are synthetic. The topologies are
public WAP optical-conflict graphs from the pinned upstream commit. Five graphs are
five independent clusters—not 5,120 independent query problems.

### Cross-environment boundary

| Environment | Complete ratio | Exact upper 95% bound | Frozen 0.50 gate |
|---|---:|---:|---|
| Original Ubuntu 24 / Python 3.13 | 0.363503 | 0.372944 | **PASS** |
| Ubuntu 24 / Python 3.13 reproduction | 0.359794 | 0.368473 | **PASS** |
| Ubuntu 22 / Python 3.13 diagnostic | 0.363912 | 0.373872 | **PASS** |
| Ubuntu 24 / Python 3.11 diagnostic | 0.534381 | 0.541937 | **FAIL** |
| Ubuntu 22 / Python 3.11 reproduction | 0.529004 | 0.536117 | **FAIL** |

Every execution completed all 105 sessions and audited all 107,520 answers. Exact
canonical compiler certificates reproduce byte for byte across tested Python 3.11
and 3.13 environments. The Python 3.11 failure is a complete-timing limitation,
not a different quotient or invalid output: the warm repeated-query session still
uses only 15.3–15.5% of MiniCard’s time, while frozen multi-megabyte JSON decoding,
deep validation, canonical reserialization, and hashing add a large common cost.
That cost remains in the published endpoint.

### Immutable evidence

```text
Freeze commit:
0af6fda36c60a110584e7db836ee5724e530fbd4

Freeze SHA-256:
31997d38cced5e5611f3f17dc6bc3929b21dcc92a6b7bc81a78db31aab949020

One-shot open commit:
b7bca7fb7f62fcb0a43e6b33705cb834cd64494c

GitHub Actions run:
37883619073

Artifact ID:
11595284861

Artifact ZIP SHA-256:
5410db4d5832544c3ec55b19c1449395020319516d20823038de66216b45527c
```

The artifact retains the frozen source archive, exact upstream graph bytes,
deterministic generated cases, schedule, all complete outputs, original-check
receipts, compiler certificates, build/environment metadata, analysis, and complete
SHA-256 manifests. A separate post-run audit verified every delivery-manifest entry
and independently recomputed the headline statistics.

## Reproduction

The one-shot run is the confirmation result. Later executions use exposed data and
are reproduction only; they cannot become a second confirmation or justify post-hoc
algorithm changes.

Use exact source commit:

```text
0af6fda36c60a110584e7db836ee5724e530fbd4
```

Acquire the five graph blobs listed in
`experiments/quotient_application/WAP_UPSTREAM_INVENTORY.json` from:

```text
marijnheule/clicolcom@4932048642da2144f387961b595112277afff82f
```

Then follow [REPLICATION.md](experiments/wap_support/REPLICATION.md). The results
branch retains the full Ubuntu 22/24 × Python 3.11/3.13 matrix. Project-owned CI is
cross-environment reproduction, not independent-team replication.

Independent replication is tracked in
[issue #55](https://github.com/tugrapaydiner/SPECTRA/issues/55). Negative or
conflicting replications are explicitly welcome.

## Exact API boundary

The experimental quotient runtime is opt-in. It does not compile during import and
does not silently fall back to another solver.

```python
from spectra.cnf.quotient_query import (
    QuotientRuntime,
    build_quotient_runtime,
)

library = build_quotient_runtime("native-quotient-build")
runtime = QuotientRuntime(library)

# edges use zero-based original vertex IDs.
# each mask has one bit per allowed colour.
prepared = runtime.prepare(
    n=4,
    k=3,
    edges=((0, 1), (1, 2), (2, 3)),
    masks=(0b011, 0b110, 0b101, 0b011),
    mode="scc",
)

try:
    result = prepared.solve(
        restrictions=((0, 0b001), (3, 0b010)),
        max_work=1_000_000,
    )
    if result.status == "SAT_VERIFIED":
        assert len(result.labels) == 4
finally:
    prepared.close()
```

A successful call returns a full original-vertex byte witness only after the
original observer accepts it. Exhaustion or contradiction returns an honest
non-solution status; it is not published as proof-certified UNSAT.

## What remains open

- independent reproduction by a person or team outside this project;
- authentic operator or simulator availability/restriction traces;
- external adoption in a maintained application;
- ordinary laptop and non-x86 measurements;
- external expert challenge of the prior-art audit;
- a separately frozen deployment-format study that removes research-evidence JSON
  overhead without rewriting the confirmed endpoint;
- formal machine-checked correspondence between native implementation and theorem.

These limitations constrain generality and impact. They do not change the frozen
PASS, but they determine how broadly the result may be advertised.

## Other project tracks

The repository also preserves bounded Boolean-search, CPU SVM, neural, low-bit,
compiler, and deployment experiments. Each has its own evidence and limitations.
**None inherits the WAP result.** In particular:

| Track | Retained value | Not established |
|---|---|---|
| Bounded CNF search | reproducible paths, reusable indexes, original-clause checks | general SAT superiority or UNSAT proofs |
| Native SVM execution | faithful same-model CPU execution and pipeline receipts | best task accuracy or universal speedups |
| Neural/low-bit history | retained hypotheses, models, failures, and controls | a flagship learned-reasoning result |
| Cover/domain experiments | exact structured execution and negative competitor results | universal graph-colouring advantage |

The latest published package release remains
[v0.7.1](https://github.com/tugrapaydiner/SPECTRA/releases/tag/v0.7.1). The WAP
flagship source is unreleased research until a separately audited release is made.

## Base toolkit quick start

```bash
git clone https://github.com/tugrapaydiner/SPECTRA.git
cd SPECTRA
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install .
python -m spectra doctor
python -m spectra cnf solve examples/tiny.cnf --seed 7 --max-flips 1024 --out answer.json
python -m spectra cnf check examples/tiny.cnf answer.json
```

The base installation does not require PyTorch or a compiler. Native runtimes are
explicit builds. Bounded search returns `SAT_VERIFIED` or `UNKNOWN`; process success
alone never means the problem was solved.

## Verify and develop

```bash
python -m pip install ".[test]"
python -m pytest tests/public --confcutdir=tests/public
python scripts/check_release_readiness.py
```

Historical slow retraining and large evidence replays remain separate. Test passes
are implementation evidence, not proof of generalization or support for every
platform/input. Read [SECURITY.md](SECURITY.md) before accepting external inputs,
native libraries, or checkpoints, and [CONTRIBUTING.md](CONTRIBUTING.md) before
changing numerical code or retained experiments.

Code is distributed under [MIT](LICENSE). Upstream datasets and third-party
components retain their own licenses and attribution requirements.
