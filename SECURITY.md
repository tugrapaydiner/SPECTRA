# Security boundaries

SPECTRA is a local research toolkit, not a sandbox or a multi-tenant service.
No comprehensive security certification is claimed for the research archive.

## Public input handling

The base CNF and evidence tools do not import PyTorch or load checkpoints.
Witness/manifest JSON rejects duplicate object keys, nonfinite numbers and
floating-point overflow. Input is limited to 16 MiB by default, with an explicit
positive `--max-json-bytes` override. This limits input bytes, not total decoder
memory or CPU. DIMACS variable/clause/literal limits do not cap arbitrary text
file size or enforce a hard solve deadline.

Run untrusted workloads with operating-system memory/time limits and restricted
filesystem permissions. Do not use special files or attacker-controlled,
concurrently changing paths. Exclusive output creation prevents normal accidental
overwrite; hash/size manifests are not signatures or authenticity guarantees.
Evidence checking rejects symlinked/traversing artifact paths, but it does not
make concurrent filesystem mutation safe.

## Research artifacts and native code

Historical research loaders include trusted-checkpoint deserialization, including
pickle-compatible PyTorch paths. Only load models and archives from a trusted,
verified source in an isolated environment. Do not expose research loaders to
uploaded third-party files. A checksum proves identity only when the expected
checksum itself is trusted. No loader was silently rewritten in this cleanup.

Native execution compiles bundled C++ with the local compiler and uses optional
PyTorch APIs. Historical dependency pins exist for reproducibility, not as a
claim of current vulnerability-free support. Review dependencies separately
before deployment; do not reinterpret an updated dependency's outputs as a
bitwise replay of the old environment.

## Reporting

Do not post credentials, private datasets or an exploit containing sensitive
artifacts in public issues. Use a repository private-vulnerability report when
that GitHub feature is enabled; otherwise contact the maintainer privately
through an established channel. No private reporting address or response-time
commitment is invented here. General non-sensitive bugs can use
[GitHub issues](https://github.com/tugrapaydiner/SPECTRA/issues).
