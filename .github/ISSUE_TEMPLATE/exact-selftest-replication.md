---
name: Exact self-test replication
about: Submit an independently generated exact self-test bundle and audit
title: "[Exact replication] "
labels: ""
assignees: ""
---

Thank you for checking SPECTRA on a machine outside the project. Please keep the
raw evidence available. A failure or mismatch is useful and should not be hidden.

## Independence

- Who controlled the machine and run?
- What relationship, if any, do you have to the SPECTRA project or its author?
- Was this run performed on project-owned infrastructure? **Yes / No**

## Source and installation

- Exact commit or release tag:
- Installation command or wheel identity:
- Did you modify the source or cases? **Yes / No**
- Predeclared deviations (`None` is valid):

## Environment

- Operating system and version:
- CPU architecture/model:
- Python version (`python -VV`):
- Compiler and version:
- Memory/process limits, container or VM details:

## Commands

Paste the exact commands used to generate and audit the bundle.

```text
spectra exact selftest ...
spectra exact audit ...
```

## Result

- Self-test exit code:
- Audit exit code:
- Audit `verified` value:
- Cases / queries:
- SAT / UNSAT counts:
- Selected modes by case:
- Selector reasons by case:
- Proof kinds:
- Source SHA-256:
- Native-library SHA-256:
- Semantic fingerprint, when compared:

## Evidence

- Link to the complete unmodified bundle archive:
- Final archive SHA-256:
- Link to `audit.json`:
- Link to `environment.txt`, `source.txt` and `deviations.md`:

Please confirm:

- [ ] The bundle contains the complete original self-test directory.
- [ ] I did not delete, replace or regenerate individual failed artifacts.
- [ ] The audit ran in a fresh process without a native module already loaded.
- [ ] I reviewed the bundle for local paths or personal information before sharing.
- [ ] Any deviation was stated above rather than chosen after seeing the result.

## Notes, failures or mismatches

Describe anything unexpected. Attach stderr and the partial bundle for failures;
do not submit only a screenshot or a hand-written summary.
