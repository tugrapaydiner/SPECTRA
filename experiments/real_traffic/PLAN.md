# Real traffic study

The current WAP result is not enough for the broader standard we are aiming for. It uses real public conflict-graph topologies, but its availability lists and repeated restrictions are synthetic, and its confirmation bank contains only satisfiable requests. This study attacks those two weaknesses directly.

## What is real

The first development source is the public Abilene dataset collected by Yin Zhang. It includes fixed routes and one traffic matrix every five minutes. `X01` is used only for development. A later week will remain unopened until the protocol, implementation, baselines, budgets, analysis, and thresholds are frozen.

The case builder:

1. reconstructs every non-loop origin–destination route from the published routing matrix;
2. creates one conflict edge when two directed routes share an internal directed link;
3. builds a topology-only reference colouring and a traffic-weighted target colouring;
4. aligns arbitrary colour names to minimize traffic-weighted migration;
5. protects the busiest routes in the reference plan and migrates the routes with the largest measured change to the target plan; and
6. keeps the resulting request before checking whether it is feasible.

That last rule matters. Infeasible requests are not discarded. A separate standard-library 2-SAT observer labels every fixed request as `SAT` or `UNSAT`, and complete SAT witnesses are hashed. The later runtime must match both outcomes.

## What is still constructed

The Abilene topology, routes, and traffic values are measured public data. The wavelength plans and protection/migration rule are research constructions. They are not claimed to be operator-deployed wavelength assignments, maintenance commands, or service policies.

The conflict model treats the published internal links as directed resources. It does not infer shared fibres, optical impairments, modulation formats, spectrum widths, restoration policies, or physical-layer feasibility that the source data does not contain.

## Development sequence

`X01` is allowed to expose parser errors, weak query diversity, poor SAT/UNSAT balance, or an unhelpful plan-construction rule. Every such failure stays visible. Once the workload is strong enough, the study will freeze:

- the exact source week reserved for confirmation;
- all source hashes and acquisition limits;
- the route-conflict and plan-construction rules;
- query widths and traffic-selection rules;
- the exact SAT and UNSAT certificate contract;
- persistent MiniCard and a compiled decision-diagram baseline;
- complete-cost timing boundaries;
- graph/time clustering and uncertainty calculations; and
- pass/fail gates.

No confirmation week will be opened before that freeze.

## Promotion bar

This branch does not inherit the previous “flagship” label. Promotion requires, at minimum:

- a nontrivial prospectively unopened trace segment;
- both SAT and UNSAT requests selected without solver-outcome filtering;
- independently checkable complete witnesses and contradiction certificates;
- a strong persistent native SAT baseline and a compiled-query baseline;
- complete setup, query, materialization, and checking costs;
- a clear advantage that survives temporal clustering and every retained trace slice; and
- no claim beyond the disclosed trace-driven model.
