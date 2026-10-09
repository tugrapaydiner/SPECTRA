# Exact compiler-certificate reproduction — PASS on two environments

This is a post-confirmation semantic reproduction using exposed graphs. It does not create a second holdout or change the one-shot performance result.

## Environments

| Environment | Result | Artifact | ZIP SHA-256 |
|---|---|---:|---|
| Ubuntu 24 / Python 3.13 | **PASS** | `11622578216` | `711485baaeb9160e100ac3e51e20f719c352eeda3590745c9ff65db16daf7b19` |
| Ubuntu 22 / Python 3.11 | **PASS** | `11622553272` | `3296a5645b4faf062b1b700434d25581b0baa1c64a020997cc12d15ab0aedec9` |

Workflow run: `37943976472`.

Both jobs checked out the untouched frozen source commit `0af6fda36c60a110584e7db836ee5724e530fbd4`, verified the freeze SHA-256, ran the actual frozen native/certificate contract file, reacquired and hash-checked all five exposed WAP graph blobs, rebuilt the deterministic cases and native runtime, exported every full integer-only certificate, and ran the independent standard-library relation verifier.

## Exact identity

The following canonical certificate SHA-256 values match the original one-shot artifact byte for byte in **both** environments:

| Case | Canonical certificate SHA-256 |
|---|---|
| WAP02a | `41707089361b681296d5f3ae3b49fd8a1dfa981c0845b387191a3bf0fd010ad1` |
| WAP03a | `ce8084da04d5696ab5ba9fba45b9cc6ee7bc662e039c108238c78247683b9620` |
| WAP04a | `51993b9683802a817ec1294ce63f4b00be32802ab8d78ed21774b681c07b766e` |
| WAP07a | `93f84c3a86fe0f3d12cea7e88c37dbe1cd7ca7dc797b92427b40919bbdf4b0e1` |
| WAP08a | `8b3dffa3cbf0e99e5607184643703b86a661fea28584a908cdd1bbea4eb08244` |

For every case and environment, the independent audit reports:

```text
valid = true
mapping_verified = true
native_code_executed_by_auditor = false
```

The auditor reconstructs the original binary implication graph, computes SCCs independently, checks every original-to-quotient identification and orientation, rewrites every original conflict, and compares the complete palettes, initial domains, atom offsets, arc offsets, and forbidden masks.

## Independent artifact inspection

After workflow completion, both artifacts were downloaded separately. Their ZIP digests matched the GitHub-reported digests. Each archive contained 36 internal manifest entries; all 36 hashes were recomputed with zero mismatches. Each retained five compressed certificate payloads, case and graph receipts, build metadata, environment records, contract logs, and the machine-readable reproduction result.

## Interpretation

The exact compiled relation is deterministic and semantically portable across the tested Linux/Python environments. The Ubuntu 22/Python 3.11 **performance** gate failure reported separately is therefore not caused by a different quotient, certificate, graph, query bank, or relation-verification outcome. It is a timing portability limitation, not a correctness or compilation-identity failure.
