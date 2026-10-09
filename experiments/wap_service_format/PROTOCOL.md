# Hash-bound WAP service format — exposed-data portability protocol

**Role:** post-confirmation deployment engineering on fully exposed inputs. This
cannot become a second confirmation study and does not change the frozen result at
`0af6fda36c60a110584e7db836ee5724e530fbd4`.

## Motivation

The frozen Python 3.13 result passed its complete-call gate on three server
environments. Both Python 3.11 reruns preserved every answer and certificate and
retained a 6.45–8.10x warm-session advantage, but missed the fixed 0.50 complete
ratio because both arms repeatedly paid a large research-evidence JSON front end.
That endpoint remains immutable. This study asks whether a deployment-oriented,
hash-bound prevalidated format removes that version-specific common overhead
without hiding solver work or weakening original-answer checking.

## New endpoint

An already canonical `WorkloadCase` is fully validated once before packing. The
packer emits a versioned data-only protocol-5 pickle container with:

- fixed magic/version/flags;
- vertex and query counts;
- exact payload length;
- payload SHA-256;
- the original canonical case SHA-256;
- a complete-file SHA-256 in a deterministic manifest.

The loader requires the manifest SHA-256 before decoding, verifies the header and
payload hash, uses a restricted unpickler that refuses global and persistent object
resolution, and checks immutable transport geometry. The format is for trusted,
hash-bound artifacts; it is not an arbitrary untrusted-pickle parser.

`run_prevalidated_session` skips only the repeated canonical evidence audit. It
still charges fresh formula/index construction, all 1,024 distinct restrictions,
complete immutable witness materialisation, the independently owned original
edge/list/query checker for every output, diagnostics, and disposal. Process
startup, imports, and native-library loading remain outside `complete_ns`, matching
the frozen endpoint. Packed file read, SHA-256 verification, restricted decoding,
and object construction are inside `complete_ns`.

Packing time is preprocessing and is reported separately. It is not subtracted
from a one-shot deployment claim; the supported contract is repeated sessions over
a previously installed and validated topology artifact.

## Fixed exposed-data matrix

Use the five already exposed WAP-A holdout cases exactly as published:
`wap02a`, `wap03a`, `wap04a`, `wap07a`, and `wap08a`. Do not generate, select, or
replace graphs or queries.

Arms:

1. exact SCC quotient (`scc`), primary candidate;
2. persistent native MiniCard (`minicard`), primary comparator;
3. no-contraction exact executor (`none`), representation ablation;
4. persistent CaDiCaL 1.9.5 (`cadical195`), secondary native solver.

Each arm runs forward, reverse, and the fixed shuffled order: 60 complete sessions
and 61,440 full answer checks per environment. PySAT remains pinned to
`1.9.dev15`. One CPU affinity, one numerical thread, a four-GiB address-space cap,
and a 60-second outer deadline remain fixed.

Run unchanged source on Ubuntu 24 with Python 3.11 and 3.13. A later ordinary-host
run is descriptive portability evidence, not a new holdout.

## Predeclared engineering gate

For each Python version independently:

- every packed artifact round-trips exactly to its canonical validated case;
- every session completes and every full answer passes the original checker;
- candidate mean complete-session cost is lower on every graph;
- upper exact whole-graph 95% mean-ratio bound `SCC/MiniCard <= 0.50`;
- upper exact whole-graph 95% p95-ratio bound `<= 1.10`.

Enumerate all `5^5 = 3,125` whole-graph resamples. Timing rounds and queries are not
independent problem samples. Failed calls remain terminal evidence; no selected
retry or threshold change is allowed.

Passing closes the tested Python-version/deployment-format gate only. It does not
create new confirmation, authentic operator traces, outside-team replication,
external adoption, non-x86 portability, or general reasoning superiority.

## Correctness and negative evidence

Public contracts must cover exact round trip, hash and header tampering, malicious
global-object pickle refusal, geometry mismatch, preservation of the validating
endpoint, canonical invalid-input refusal, and equality of canonical versus packed
solver outputs. Run focused contracts with the actual native runtime under UBSan.
Retain interrupted/resumed sessions and all prior Python 3.11 failures; do not
rewrite the frozen report.
