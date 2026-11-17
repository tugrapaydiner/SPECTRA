# Benchmarks

Workload generators, baseline solvers and benchmark runners. This directory is not
part of the installed package; run everything from a source checkout.

## WAP repeated-query workload

`benchmarks/wap` builds the list-colouring support workload behind the README
result. Graph topologies are the WAP optical-conflict instances from
[`marijnheule/clicolcom@4932048`](https://github.com/marijnheule/clicolcom/tree/4932048642da2144f387961b595112277afff82f);
`upstream_inventory.json` pins every graph by Git blob hash. Availability lists and
1,024-query sessions are generated deterministically from a fixed seed.

```python
from benchmarks.wap.workload import build_case

case, receipt = build_case("wap05a.col", graph_name="wap05a.col",
                           expected_git_blob_sha1="<blob sha from upstream_inventory.json>")
```

`benchmarks/wap/graph.py` contains the DIMACS `.col` reader, a colouring checker and
the certified critical-clique quotient used to derive colour lists.

## Exact queries against independent baselines

`benchmarks/exact` holds two independent solver implementations with shared
input/output and verification contracts. Their benchmark harnesses use
`spectra.exact` packet and verification helpers; both baselines emit the same
`SCQ2` answer format, so every answer is verified identically:

- `direct_twosat`: an independent native 2-SAT service with three strategies;
- `cadical_exact`: an adapter around the official CaDiCaL 3.0.1 library
  (set `CADICAL_SOURCE` to a checkout).

```bash
python -m benchmarks.exact.direct_twosat_benchmark \
    --spectra-library build/exact/_spectra_exact_binary*.so \
    --direct-library build/direct/libdirect_twosat.so --out results/direct --repeats 24
python -m benchmarks.exact.cadical_exact_benchmark \
    --spectra-library ... --cadical-library ... --cases cases/ --out results/cadical
```

Build the libraries with `spectra.exact.build_exact_runtime`,
`benchmarks.exact.direct_twosat.build_direct_twosat_runtime` and
`benchmarks.exact.cadical_exact.build_cadical_exact_runtime`. Repeats are
position-balanced across arms; reports give per-cell timings and geometric means.

## SAT

`benchmarks/sat/campaign.py` runs one formula per process for a declared inventory
and solver field, with memory, time and output limits (POSIX resource limits on
Linux, Job Objects on Windows). Every SAT model is rechecked against the original
formula and every UNSAT proof is replayed by a declared external checker; failures
are retained, never dropped.

```bash
python -m benchmarks.sat.campaign run --inventory inventory.json --solvers solvers.json \
    --out results/campaign --timeout 5 --memory-mib 4096 --checker /path/to/drat-trim
python -m benchmarks.sat.campaign_verify results/campaign --recheck-models
python -m benchmarks.sat.official_score scores.csv --out standings.json
```

`campaign_verify` re-verifies an archived campaign without executing any archived
command. `official_score` recomputes SAT Competition PAR-2 standings from the
published `scores.csv`.

## CNF local search

```bash
python benchmarks/cnf/indexed_search.py run --out results/indexed
python benchmarks/cnf/indexed_search.py verify --out results/indexed --replay
python benchmarks/cnf/compact_state.py run --out results/compact
```

Compare the reference, ranked and indexed search engines (and the compact versus
bitset repair state) on generated uniform and planted 3-SAT. Runs are verified and
replayable from their recorded sources and seeds.

## Two-plan migration

`benchmarks/traffic` builds migration cases from the public Abilene backbone
routing and traffic-matrix data: it reads the topology and demands, derives a
lightpath conflict graph, constructs current and target plans and generates
request banks.

## SVM fixtures

`benchmarks/svm_stream/corpus.py` rebuilds eighteen fixture models and their inputs
from public datasets with pinned scikit-learn 1.8.0 and fails unless every byte
matches `corpus_manifest.json`. `check_corpus.py` replays all of them through an
installed wheel; CI runs it on Linux, ARM64 and Windows.

```bash
pip install numpy==2.3.5 scipy==1.17.0 pandas==2.2.3 scikit-learn==1.8.0 \
    joblib==1.5.3 threadpoolctl==3.6.0
python benchmarks/svm_stream/corpus.py --out .svm-corpus
```
