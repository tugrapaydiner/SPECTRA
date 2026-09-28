# Single-lookup m2cgen review

This is an isolated export/compiler experiment, not a SPECTRA inference feature.
The production runtime, trained benchmark models, previous compact-emitter failures
and task-level negative results are unchanged. Start with [RESULTS.md](RESULTS.md)
and the prospective [PROTOCOL.md](PROTOCOL.md).

## Source and environments

The inspected upstream head is `9784632311986234032673cdbfd29fc4c5cb429d`.
The prior performance environment uses the original m2cgen0.10.0 wheel.
These are different source snapshots; each patched output is compared with its
OWN original, not assumed interchangeable. The helper pins the AST/interpreter
hash pairs, validates all five lookup sites, writes only a fresh separate source
copy, and retains a per-file patch receipt. It does not install, monkey-patch a
live interpreter or execute an upstream setup hook.

The reproduction packet supplies both original snapshots, MIT licensing, all
locally constructed compatibility fixtures, original seven benchmark model files,
prior generated C references and literal results. Pickles are trusted reproduction
inputs only: do not substitute untrusted model pickles. No generated code or
benchmark model is fitted during performance measurement.

```bash
python experiments/codegen_review/apply.py --source /original/m2cgen \
  --destination /fresh/single-review --variant single
PYTHONPATH=/fresh/single-review M2CGEN_ORIGINAL_SOURCE=/original/m2cgen \
  python -m pytest experiments/codegen_review/test_cache.py \
  experiments/codegen_review/test_apply.py experiments/codegen_review/test_audit.py
```

For each of `stock`, `guard`, `single`, create a separate `VARIANT-review` folder
under a common packages directory. `empty` is a diagnostic fourth alternative
used in the hashing probe, not the promoted fixed-model timing experiment.

```bash
python experiments/codegen_review/bench.py run --packages /packages \
  --models /trusted-frozen-models --out /new/run
python experiments/codegen_review/audit.py --run /new/run \
  --models /trusted-frozen-models --packages /packages \
  --script experiments/codegen_review/bench.py \
  --prior-sources /prior-generated-models --out /new/audit.json
```

`bench.py` requires Linux resource and CPU-affinity APIs. It restricts each child
to one numerical thread, one CPU, 120 seconds and a 4-GiB address-space limit.
That last value is not measured peak RSS. It runs all 57 predetermined attempts
without replacing failures. The run directory must be new. The generation timer
includes assembly and interpretation, but excludes imports, trusted model loading
and final output-file writing. Total child wall time and process high-water RSS
are recorded separately. This is code-export speed, not inference speed.

## Compatibility and hashing diagnostics

`panel.py prepare` constructs 19 small fixed synthetic fixtures (one deliberately
unsupported MLP); `panel.py evaluate` checks every exporter exposed by the pinned
package. Prefer the supplied same-byte fixtures when reproducing our comparison.
Each output file is retained with its identity, not just a success flag. Language
source equality is NOT compilation/execution on sixteen target platforms.

`hash_probe.py` instruments expression hashing for three fixed graph families;
it never supplies timed performance observations. Same-type cache misses retain
quadratic work. Removing impossible lookups and duplicate hit hashes does not
make every model export linear-time. Unknown stored key classes disable the
negative type shortcut; ordinary equality/hash still governs possible matches.

The private cache only supports well-formed stock AST semantics. Do not infer
semantics for malformed/custom children, hash side effects or modifying keys while
they are stored. Mutation between exports is tested with a fresh cache. This is
not a general replacement for a Python dictionary.

## Actual upstream test execution

`.github/workflows/codegen-review.yml` uses the original upstream Python3.10 test
environment and runs its identical non-end-to-end tests on separate stock, guard
and single-lookup copies. Generated-language end-to-end execution is excluded,
not silently passed. New focused tests run separately. The installed production
SPECTRA numerical tests are not represented by this upstream test count.

The patch draft against the pinned upstream head includes the five lookup changes,
cache implementation and focused tests. It has not been submitted to or reviewed
by the third-party maintainers. Automated tests inside this project are not outside
reproduction or community acceptance. A public upstream submission requires a
separate explicit decision by the repository owner.

## References

- Original source/tests and MIT license: https://github.com/BayesWitnesses/m2cgen
- Hash/equality contract: https://docs.python.org/3/reference/datamodel.html#object.__hash__
- Previous source and negative runtime findings: separately retained SPECTRA
  `experiments/codegen_scalability` local delivery at `3dbbacf`.
