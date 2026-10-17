# Proof-carrying exact-query API

The current unreleased source tree exposes a versioned, installable API for exact
queries over a fixed simple conflict graph when every original vertex has one or
two allowed colours. It is available as `spectra.exact`.

This API is separate from bounded CNF search and from the older quotient runtime.
Every completed call returns either a complete original-vertex colouring or a
portable original-literal contradiction proof. There is no `UNKNOWN` outcome for
an admitted completed call, no silent fallback solver, and no compilation during
module import.

## Contract versions

| Layer | Version | Role |
|---|---:|---|
| Public Python API | `EXACT_API_VERSION = 1` | names, validation and result objects |
| Native extension ABI | `EXACT_NATIVE_ABI = 3` | CPython extension entry points |
| Query bank | `SBQ1` | immutable current restrictions for a prepared graph |
| Full session packet | `SBC1` | immutable graph, domains and request bank |
| Certified answer bank | `SCQ2` | full SAT labels or complete UNSAT proof per row |
| Contradiction proof | `SBP1` | original-literal paths checked without solver state |

The Python package version remains `0.7.1` until an actual package release is
made. The exact API version is its explicit contract version; source availability
on a branch is not a PyPI release.

## Explicit native build

Ordinary wheels remain compiler-free. Building the exact runtime is an explicit
local operation and requires GIL-enabled CPython plus a supported C++ compiler.
Linux uses `g++`; Windows x64 uses a configured MSVC developer environment.

```python
from pathlib import Path
from spectra.exact import build_exact_runtime, ExactRuntime

library = build_exact_runtime(Path("exact-native-build"))
runtime = ExactRuntime(library)
```

The builder publishes the extension only with an adjacent `build.json`. The
receipt binds the packaged C++ source hash, compiler command and version, Python
implementation and SOABI, API/ABI versions, and final binary hash. `ExactRuntime`
checks this receipt by default before loading executable code. Loading a separately
managed trusted extension without a receipt requires the explicit
`verify_receipt=False` override.

The receipt proves local identity, not compiler trust, source provenance beyond
the installed package, sandboxing, or absence of vulnerabilities.

## Original-address problem and persistent session

```python
from spectra.exact import ExactProblem, verify_answer

problem = ExactProblem.create(
    n=4,
    palette=3,
    edges=((0, 1), (1, 2), (2, 3)),
    masks=(0b011, 0b110, 0b101, 0b011),
)

with runtime.prepare(problem, mode="scc") as session:
    query = ((0, 0b001), (3, 0b010))
    answer = session.solve(query)
    assert answer.status in {"SAT_VERIFIED", "UNSAT_VERIFIED"}
    assert session.check(query, answer)
    assert verify_answer(
        problem, query, answer.status, answer.labels, answer.proof
    )
```

`ExactProblem.create` owns immutable, canonical inputs. Edges are converted to
sorted unique `(left, right)` pairs. Each domain mask must contain exactly one or
two colours from a palette of at most 64 colours.

Persistent modes are `raw`, `scc`, `sparse`, and `route`. They share the same
answer contract. `route` changes proof reconstruction, not the mathematical
problem. Persistent sessions own a solver and a separately constructed observer;
closing is idempotent, and subsequent use fails closed.

## One-call prepared session

`SBC1` is the deployment-oriented input form. It contains the complete fixed graph,
original domains, and request bank. The solver and observer each parse the bytes
independently. One call constructs both owners, answers every request, checks every
witness or proof, serializes `SCQ2`, and disposes native state.

```python
from spectra.exact import pack_session_packet, verify_session_output

queries = (
    (),
    ((0, 0b001),),
    ((0, 0b001), (1, 0b001)),
)
packet = pack_session_packet(problem, queries)
result = runtime.solve_session_packet(packet, mode="auto")
assert result.query_count == len(queries)
assert verify_session_output(packet, result.output)
```

The default `auto` mode uses only graph structure and request-bank length. It does
not inspect measured timings or cached answers. The result metadata names the selected engine. Explicit modes retain their meaning,
and callers can still force `raw`, `scc`, `sparse`, or `route` for controlled comparisons.

`pack_session_packet` accepts ordinary Python inputs. `pack_validated_session_packet`
is for already admitted immutable inputs and still rechecks exact types, bounds,
edge ordering and query geometry. The strict decoder rejects reversed, duplicate,
unsorted, truncated, overlong, or reserved-bit-modified packets rather than
repairing them.

## Independent replay and CLI

`verify_answer`, `verify_certified_output`, and `verify_session_output` use only
standard-library Python and original inputs. They do not load the native extension,
receive SCC identifiers, or trust the solver's previous decision.

```bash
spectra exact verify request.sbc1 answer.scq2
```

The command defaults to a 64 MiB limit for each input file and returns exit code
`0` for a fully verified bank, `1` for well-formed evidence that fails semantic
verification, and `2` for malformed input, I/O errors, or invalid settings.

An explicit build is also available:

```bash
spectra exact build exact-native-build
```

No command downloads a compiler or dependency.

## Resource and performance boundary

`max_bytes` is a conservative admission bound for native owners, scratch arrays,
proof envelopes, serialized output and the final immutable Python bytes copy. It
is not a total process-RSS limit, hard deadline, Python-object bound, or protection
against an untrusted compiler. Use operating-system process limits for hostile or
multi-tenant workloads.

Ready-`SBC1` timings begin after packet preparation. They include packet parsing and
hashing, native construction, solving, complete witnesses/proofs, independent
native checking, `SCQ2` construction, disposal, and output hashing. They exclude
JSON decoding, packet creation, compilation, process startup and offline Python
replay. Do not present them as first-use application latency.

The current performance evidence uses exposed synthetic development graphs and is
not an independent application benchmark. The installed API does not inherit the
frozen WAP result, create a general SAT claim, or establish algorithmic novelty.

## Tested boundary

The API contracts require:

- byte equality between the packaged and research native source in the repository;
- strict packet and output decoders;
- SAT and UNSAT proof replay from original inputs;
- build-receipt tamper rejection;
- persistent and one-call native execution;
- installed-wheel operation outside the checkout;
- Linux x86 and ARM64 builds, Windows x64/MSVC builds, and sanitizer checks in the
  dedicated workflow.

See [Security](../SECURITY.md), [Development](DEVELOPMENT.md), and the exact-query
research reports under `experiments/compiled_query/` for evidence and limitations.
