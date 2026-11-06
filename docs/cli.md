# Command-line interface

`spectra` (or `python -m spectra`) uses only the standard library.

```bash
spectra --version
spectra doctor                                   # environment and packaged-source report
spectra cnf solve formula.cnf --backend indexed --seed 7 --max-flips 1024 --out answer.json
spectra cnf check formula.cnf answer.json
spectra exact build build/exact
spectra exact verify request.sbc1 answer.scq2
spectra exact selftest bundle/ && spectra exact audit bundle/
spectra exact compare linux-bundle/ windows-bundle/
spectra evidence run/ manifest.json
spectra svm run deployment --library build/svm/libspectra_svm.so --engine two_stage --input rows.jsonl --output out.jsonl
python -m spectra.sat.admission formula.cnf.gz
```

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success. For `cnf solve`, inspect `status`; `UNKNOWN` is a valid outcome. |
| `1` | Well-formed input that fails verification (unsatisfied witness, rejected answer bank). |
| `2` | Invalid input or settings, I/O error, or a refused overwrite. |

## Input handling

JSON inputs (witnesses, manifests) are read with a strict decoder that rejects
duplicate keys, `NaN`/`Infinity`, numeric overflow and invalid UTF-8. The default
limit is 16 MiB, set with `--max-json-bytes`; binary inputs to `exact verify` are
limited to 64 MiB (`--max-bytes`). Limits bound input bytes, not total memory or
time. Output paths are created exclusively and never overwritten.

`cnf solve` adds `formula_sha256` and `backend` to its record; `cnf check` rejects a
witness whose recorded `formula_sha256` does not match the formula.

## Evidence manifests

```python
from spectra.evidence import capture, verify, write_new

manifest = capture("run", ["config.json", "observations.jsonl"])
write_new("manifest.json", manifest)
assert verify("run", manifest, required=["config.json", "observations.jsonl"])["verified"]
```

A manifest (`spectra.evidence_freeze.v1`) records the SHA-256 and size of each
relative path. Verification rejects changed or missing files, path traversal and
symlinks. A matching manifest proves byte identity, not provenance.
