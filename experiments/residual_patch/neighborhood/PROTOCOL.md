# Larger residual neighborhoods: development protocol

Base: `6c0d8695a1def36ef112d8864df700a2bc4eb3ae` (PR #23).

The previous 289-parameter repair selector did not beat the strongest simple
control. This additive experiment tests whether a broader repair space supplies
stable action value before another model is trained. Historical source and
results remain unchanged. All existing SAT inputs are development material;
no previous or new confirmation split is opened by this protocol.

## Frozen diagnostic

Use all 192 original training formulas, uniform/community 3-SAT at 64/128
variables. Regenerate the original baseline trajectory at depths 1,024 and
4,096. Retain solved source states as well as every failed state. There is no
solver-based input filtering or favorable-case replacement.

At every failed source state compare: unchanged continuation; a full random
restart; the previous exact greedy one/two-flip patch; and exact minimum-residual
reassignment of 4, 8, or 12 variables. Neighborhoods use either a frontier grown
from a failed clause through clause-variable incidence, or a uniform variable
sample. Each neighborhood size/type has two predetermined proposal seeds. There
are 15 action slots, including controls. Duplicate resulting assignments remain
in the inventory; they do not count as distinct new coverage.

Exact neighborhood enumeration considers every assignment in that restricted
neighborhood, holds other variables fixed, and retains a minimum-residual
assignment. Ties prefer fewer flips, then a deterministic mask order. This is a
classical bounded local optimizer, not a neural generator or new SAT algorithm.
Intermediate assignments are private search states; only the selected complete
assignment is returned. Feature, construction, enumeration, copying, and checking
costs are measured, with enumeration work separate from continuation moves.

For each resulting assignment run the same custom probSAT-style continuation
for 512 moves under 32 independent, predetermined rollout seeds. The first 16
seeds select an action and the second 16 evaluate it; reverse the halves and
average. Also report the biased same-rollout maximum explicitly as such. Do not
call either quantity a true-oracle bound. Evaluate against unchanged continuation,
restart, the old greedy patch, and a global action chosen with the current
formula excluded. Bootstrap whole formulas within the four strata (2,000 draws),
retaining all source states from each selected formula. These are exploratory
conditional intervals, not a correction for all adaptive research decisions.

## Conditional learning and online work

A further compact learned selector is justified only if independent-rollout
selection improves by at least 2 percentage points over the formula-excluded
static action, with a positive lower 95% interval. This is a triage rule, not a
claim that privileged rollout selection is deployable or affordable. Its full
data-collection cost is reported separately. If the gate fails, retain the
negative result and measure simple native online controls; do not train another
model merely to increase the model count.

For full solves, compare native continuation, restart, existing greedy patches,
and neighborhood repair invoked only after 512 moves without a new best residual.
Use the original 32 validation and 96 consumed development formulas and search
seeds 17001/27002, three randomized timing rounds, and complete 2/10ms windows.
Select a single primary new configuration on validation before development.
Model loading/native compilation are separate cold setup; initialization, native
construction, every repair/feature/decision, final original-formula checking and
cleanup belong inside the warm solve window. Late valid answers receive no
within-window credit. Fixed-work replay is separate from latency evidence.

The online improvement target remains +10 points of success at 10ms versus the
validation-selected strongest control, or 2x lower cost at a predeclared matched
quality target. A point-estimate gain alone does not establish this target.
Compare external authors' probSAT when its license/build permits faithful use;
record its exact source and timing boundaries. Existing default CDCL context is
not a current state-of-the-art portfolio. Two distributions are one task.

The broad protocol is published before diagnostic execution. Exact executable
source is frozen locally before each run; later remote executable publication
is not external preregistration. All wrong answers, failed attempts, source
inventories and measured costs must remain available. Correctness success never
changes a failed scientific gate. No L7, hiring, energy or general SAT claim.
