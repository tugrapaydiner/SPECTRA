# First direct native 2-SAT challenge — exposed development result

**Date:** 2026-10-19
**Source:** `f36183dd3c859abfa6b2eccf859a3b2599875b3b`
**Workflow run:** `39339071060`
**Classification:** project-owned exposed development evidence, not a holdout and
not independent replication

## Verdict

The purpose-built direct 2-SAT challenger erased SPECTRA's general pure-service
advantage on this 28-cell panel. SPECTRA won important dense-compression cells, but
the challenger was faster overall. This is a failed broad competition result and a
successful diagnostic: it identifies concrete bottlenecks that must be removed
before a prospective field-leading claim is defensible.

No cells, failures, competitors or endpoints were removed after timing. The
verified-service endpoint is reported separately because Python proof/witness replay
can dominate and must not be used to conceal a native service loss.

## Aggregate results

Ratios below are `SPECTRA auto / challenger`, so values below 1 favour SPECTRA.
The explicit-oracle column compares SPECTRA auto with the fastest challenger mode
for each cell after the fact and is descriptive only.

| Environment and endpoint | Auto / challenger auto geomean | SPECTRA wins | Auto / challenger best explicit geomean | SPECTRA wins | Worst cell |
|---|---:|---:|---:|---:|---:|
| x86-64 pure service | **1.205127** | 8 / 28 | **1.212667** | 8 / 28 | **8.156227** |
| ARM64 pure service | **1.226596** | 8 / 28 | **1.235624** | 7 / 28 | **8.652765** |
| x86-64 verified service | **0.982597** | 12 / 28 | **0.988979** | 10 / 28 | **1.031475** |
| ARM64 verified service | **0.993398** | 11 / 28 | **0.994505** | 9 / 28 | **1.020761** |

The native service result means the direct challenger was approximately 17% faster
than SPECTRA auto on x86-64 and 18% faster on ARM64 by reciprocal geometric mean.
The verified endpoint is roughly tied because both systems then pay substantial
standard-library replay cost. That near tie is useful for end-to-end verification
engineering, but it is not a native-service victory.

## Largest losses

The most severe losses were concentrated rather than universal:

- base-UNSAT odd-cycle banks at 1,024 queries: about **7.96×** slower on x86-64
  and **8.42×** slower on ARM64;
- the same family at 64 queries: about **6.26×** and **5.01×** slower;
- mixed-palette 1,024-query cells: about **2.03×** and **1.77×** slower;
- high-pin 1,024-query cells: about **1.91×** and **1.69×** slower;
- long path/query-UNSAT cells: generally about **1.6–1.7×** slower; and
- small isolated high-colour cases: generally about **1.7–1.9×** slower.

The base-UNSAT result is the first optimization target: once graph construction has
already proved the immutable base impossible, repeatedly rebuilding and serializing
the same contradiction envelope is avoidable work.

## Strong cells retained

SPECTRA still showed substantial advantages on dense bipartite compression:

- approximately `0.18×` at one query;
- approximately `0.21×` at eight queries;
- approximately `0.25×` at 64 queries; and
- approximately `0.83×` on x86-64 and `0.99×` on ARM64 at 1,024 queries.

Several single-query path cases also favoured SPECTRA at approximately `0.84–0.92×`.
These wins show that the quotient engine has a real structural advantage, but the
automatic service does not yet convert that advantage into overall dominance.

## Correctness and portability

Before timing, the challenger passed exhaustive and randomized oracle agreement,
all three portable proof forms, malformed-input and receipt-tamper rejection,
scalar and tile64 kernels, and the repository's independent Python replay. The
matrix passed on:

- Linux x86-64, Python 3.11 and 3.13;
- native Linux ARM64, Python 3.13;
- Windows x64/MSVC;
- UBSan; and
- ASan+UBSan.

Every arm received the same immutable packet bytes and emitted a complete `SCQ2`
bank. Resource failure was never relabelled UNSAT.

## Evidence

- x86-64 artifact: `12228168310`
  - ZIP digest: `sha256:c513ce8d2e09a90aa7343c29b1f67ec498ea0101c17bf2193f94f7ecf6f826a1`
- ARM64 artifact: `12227359299`
  - ZIP digest: `sha256:d5eec3675a21f062288e69164c01a3d3a72a0d32869ed3e367df3c09746dffdc`

Both artifacts retain the source and tree identities, build receipts, environment,
raw per-repetition samples, position schedules, complete result JSON, logs and
SHA-256 manifests.

## Next engineering targets

1. Collapse base-UNSAT banks to one immutable contradiction computation and cheap
   repeated `SCQ2` emission.
2. Reduce proof construction and serialization cost for long path/query-UNSAT
   banks.
3. Improve mixed-palette and high-pin packet processing without weakening the
   original-input verifier.
4. Remove small-case fixed overhead while preserving dense-compression wins.
5. Re-run this exact exposed panel without changing its interpretation; only after
   a stable win, freeze a new multi-family prospective holdout.

## Claim boundary

This result does not show that the challenger is universally superior either. It
covers one project-owned implementation and one exposed synthetic panel. It does,
however, invalidate any current claim that SPECTRA generally beats the cheapest
purpose-built solver for its own admitted problem class. That claim remains open.
