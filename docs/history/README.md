# Historical research

The active tree contains maintained source, regression tests and user-facing
instructions. Reproduce historical experiments from their complete pinned trees:

- [Frozen ITC selector-v10 study](https://github.com/tugrapaydiner/SPECTRA/tree/9e1037f2b46780dc943804f8726cee042108d15a)
  retains the matching native selector, transfer runners, dedicated tests,
  protocols, timing amendments and manual workflow. The current selector is v11;
  old holdout runs must use their original source and declared gates.
- [Retired milestone runners](https://github.com/tugrapaydiner/SPECTRA/tree/21ef4511ed586c7509c55657dfbbccfb396a1bab)
  retains one-shot M02-M15 runners, report generators and superseded protocol
  notes. Numerical replay helpers and current runtime tests stay active.
- [Completed study source and tests](https://github.com/tugrapaydiner/SPECTRA/tree/757babb05bd631c89075f374ba765a197a1fb77b)
  retains the WAP confirmation harness, earlier CNF/SVM studies, their reports,
  and the README chart generator. Active copies of those completed runners
  and their exclusive tests are retired; shipped APIs and their tests remain.
- [Part 3 evidence before cleanup](https://github.com/tugrapaydiner/SPECTRA/tree/38c13990a1ba92229cf5cf2a57b1a600b218889e)
  retains the run artifacts and diagnostic harnesses removed from this checkout.
- [Earlier research archive](https://github.com/tugrapaydiner/SPECTRA/tree/0b68898f3e64063ac3dc8904ae42f0b33590e7af)
  is the immutable archive/research-history-20261016 checkpoint.
- Receipts naming `abd10d3d0ef24300c3e43b8432886c8203fd465b` or
  `45b7fba65d0eb702616b1f617b3e4f0017fb4438` retain those original identities.

Use an isolated historical checkout and verify its recorded input and artifact
hashes. Old manifests describe their original trees, not the current checkout.
All failures, missed gates and exposed-input boundaries remain in those records.
Archiving a result does not promote it into a current claim.

Internal plans and working reviews belong in ignored local storage. Removing an
active copy does not erase earlier public Git history. See
[current status](../STATUS.md) for the maintained project scope.
