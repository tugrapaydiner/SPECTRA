# Exact two-plan migration support (experimental)

This directory contains an **opt-in research implementation** for repeated support
queries between two already valid graph colourings. It is not part of SPECTRA's
confirmed WAP result, and no result in this directory inherits that result's
flagship status.

## Supported contract

Given a conflict graph and two proper colourings `A` and `B`, each original vertex
may use only `A(v)` or `B(v)`. A query further restricts arbitrary original
vertices. The checked service returns either:

- a complete original-vertex assignment containing the unique
  inclusion-minimal migration closure; or
- a fresh original-edge implication path explaining why the request is
  contradictory.

The solver and observer build their own native indexes. A result is returned only
after the observer checks request compliance, the exact two-plan relation,
minimum necessary change, output geometry, and every contradiction path.

## Entry point

```python
from experiments.real_traffic.packet_service import PacketService

service = PacketService.build("native-build")
with service.prepare(((0, 1),), (0, 1), (1, 0)) as prepared:
    result = prepared.solve((((0, 2),), ((0, 2), (1, 2))))
    assert result.audit.sat == 1
    assert result.audit.unsat == 1
    assert result.audit.minimum_checked
```

Native compilation is explicit and currently validated on Linux with a C++17
compiler and matching Python development headers. There is no silent fallback.

## Status and limits

The packet interface, sparse batch engine, and independent observers have strong
exhaustive/randomized tests and sanitizer coverage. Their retained development
measurements come from an already exposed Abilene X01 workload and synthetic
stress cases. The packet path has preparation and tail regressions on some cases,
so it is **not** the project default and is not advertised as a universal speedup.

This contract does not establish:

- safe live network reconfiguration;
- optimal selection of the target plan;
- arbitrary-list or general-CNF acceleration;
- superiority for small request banks or every graph structure;
- fresh confirmation, outside-team replication, or operational adoption.

See `MIGRATION_PROOF.md`, `COMPACT_CORE_PROOF.md`,
`STREAMED_OBSERVER_PROOF.md`, and `PACKET_CORE_PROOF.md` for the exact derivations
and accounting boundaries.
