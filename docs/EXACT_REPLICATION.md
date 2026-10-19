# Independently reproducing the exact self-test

This guide is for people outside the SPECTRA project who want to check the
release-facing exact service on their own machine. It covers deterministic SAT /
UNSAT semantics, portable proofs, evidence retention and cross-platform agreement.
It is **not** the WAP performance replication protocol and does not establish a
speed claim.

## Independence and source choice

Use one exact public commit or release tag and state it before running the test.
Do not silently patch the source, change the cases, delete failures, replace the
native binary, or regenerate only part of a failed bundle.

A useful independent result should disclose:

- who controlled the machine and execution;
- any relationship to the SPECTRA project;
- the exact commit or release asset used;
- installation method and Python/compiler versions;
- every deviation decided before looking at the result; and
- whether the complete raw bundle is publicly available.

Project-owned GitHub runners are cross-platform verification, not independent
replication.

## Requirements

- GIL-enabled CPython on Linux x86-64 or ARM64, or Windows x64;
- a local C++17 compiler (`g++` on Linux or an x64 MSVC developer environment on
  Windows);
- enough space for a small native build and evidence directory; and
- no optional ML/scientific packages are required.

Run from a clean virtual environment or an isolated installed wheel. Record the
output of `python -VV`, the compiler version and the operating-system description.

## Generate the bundle

Choose a new destination that does not exist:

```bash
spectra exact selftest exact-selftest
```

A successful run creates exactly eight frozen files beneath that directory:

- `selftest.json`;
- `native/exact_binary.cpp`;
- `native/build.json`;
- one platform-specific native library;
- two `SBC1` request packets; and
- two `SCQ2` certified output banks.

`SHA256.json` binds those eight files but is not itself included in its inventory.
Do not edit, rename or recompress individual files before auditing them.

A failed run deliberately leaves partial evidence. Keep it if the failure is
material, and use a different destination for any rerun.

## Audit in a fresh process

Run the audit from a new interpreter or shell process:

```bash
spectra exact audit exact-selftest > audit.json
```

A valid audit reports:

- `verified: true`;
- two cases and five queries;
- two SAT and three UNSAT decisions;
- proof kinds `[1, 2, 3]`;
- eight manifest artifacts;
- `native_execution: false`; and
- `native_module_loaded: false`.

The audit refuses unbound files, missing files, changed hashes, unsafe paths,
receipt mismatches, changed selector receipts, invalid proofs, changed packets or
outputs, and an already-loaded native module.

## Retain and submit the evidence

Create one archive containing the unmodified `exact-selftest/` directory plus:

- `audit.json`;
- `environment.txt` with operating system, CPU architecture, Python and compiler;
- `source.txt` with the exact commit/tag and installation command;
- `deviations.md`, even when it only says `None`; and
- a SHA-256 for the final archive itself.

Review the bundle before publishing it. Build receipts contain local executable and
build paths, which may include a username or workspace name. Do not include tokens,
private repository URLs, unrelated files or other secrets.

Submit the result through the repository's **Exact self-test replication** issue
template. Failure reports are useful and should include the complete partial bundle
and stderr rather than only a screenshot.

## Compare with another bundle

Anyone with two or more complete bundles can compare them without loading native
code:

```bash
spectra exact compare first-bundle second-bundle > comparison.json
```

Platform, Python, compiler and native-library hashes may differ. The source,
packets, certified outputs, proof coverage, API/ABI and selector receipts must
match. A passing comparison emits one shared semantic fingerprint.

A comparison pass proves agreement for this deterministic self-test only. It does
not prove general solver correctness, performance leadership, production safety,
or external adoption.
