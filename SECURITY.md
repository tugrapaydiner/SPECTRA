# Security

SPECTRA is a local library and CLI, not a sandbox or a multi-tenant service.

## Inputs

- The installed package imports only the Python standard library and never
  loads pickles or executes code from input files.
- JSON inputs are read with a strict decoder (no duplicate keys, `NaN`, overflow
  or invalid UTF-8) and a byte limit (16 MiB by default). Binary packets have
  their own limits, and DIMACS parsers bound variables, clauses and literals.
- These limits bound input size, not total memory or run time. Run untrusted
  workloads under operating-system memory, time and filesystem limits.
- Output files are created exclusively; existing paths are never overwritten.
- Hash manifests and receipts detect accidental changes; they are not signatures.

## Native code

Native components are compiled from the packaged C++ source with your local
compiler, only when you call a `build_*` function. Each build writes a receipt
binding the source hash, compiler command, Python ABI and binary hash, and
runtimes verify it before loading. This detects accidental or post-build changes;
it does not sandbox the compiler or make the native code safe. Build only with a
trusted toolchain, and treat built libraries as executable code.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository. Do not
include credentials or private data in public issues.
