# M13 retained measurements

Accepted M13 defensible-measurement evidence, regenerated from Actions run `34155993914` at head `4a700938cece1b041205d8ad16a48d280cfc9a99`.

- Primary workload: actual sequential `CPURecursiveRuntime` complete B=1 solve path.
- Warm timing: 24 distinct held-out inputs × 3 rounds; median `7.585985` ms, p95 `7.819384` ms.
- Cold start: `16417.176` ms, including native extension build/load and first complete solve.
- Physical package energy: unavailable on the hosted runner (`no_package_domain`); joule fields are null, not zero.
- Memory scopes are separated in `memory.json`; RSS peak is a sampled window-local maximum and may miss allocations shorter than the 0.5 ms sample interval.
- The K-input kernel rows are explicitly `precomputed_input_reuse`; they are not evidence of sequential recurrence, cache residency, or a bandwidth bottleneck.
- All throughput and arithmetic-intensity rows use `1 MAC = 2 arithmetic operations`.

The CI artifact ZIP SHA256 is `a64cf2e073eef2c636d68fef9c8ce7c4d517e79714bd27e051e0c4fbfc9c48a1`. Controlled counter fixture logs are synthetic contract evidence, not physical hardware energy measurements.
