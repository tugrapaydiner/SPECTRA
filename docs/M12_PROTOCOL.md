# Milestone 12 Protocol — Grounded Router/Halter RL Training

**Status: preregistered before any M12 pilot result.**

M12 completes one scope only: a real, bounded reinforcement-learning training path for SPECTRA's M11 adaptive router and halter using strictly loaded, frozen reasoner and grounded-verifier checkpoints.

M12 is not an execution-speed milestone and does not revise M11's accepted sparse-execution claims. It is also not a claim that RL improves reasoning unless the held-out pilot actually shows that.

## Research question

Can the existing adaptive execution policies be optimized by an explicit actor-critic training command with:

- independently verified terminal task correctness;
- a declared compute-cost objective;
- correct per-example terminal/truncation handling;
- no policy-gradient credit for forced actions;
- fixed grounded-verifier potential shaping;
- auditable optimizer/gradient ownership;
- a held-out comparison against simple controls?

## Frozen checkpoint boundary

The RL command accepts two required inputs:

```text
--reasoner-checkpoint <spectra.training v1 checkpoint>
--verifier-checkpoint <grounded_state_verifier auxiliary checkpoint>
```

They are loaded only through:

```text
eval.checkpoint_eval.load_research_trm_checkpoint
eval.grounded_checkpoint.load_grounded_verifier_checkpoint
```

The grounded verifier must be strictly compatible with the exact reasoner checkpoint SHA/task/dimension/vocabulary/sequence metadata.

Before the first episode:

```text
reasoner parameters requires_grad = false
reasoner mode                    = eval
verifier parameters requires_grad = false
verifier mode                    = eval
```

Their tensor-state hashes are recorded before and after RL training. Any mutation fails M12.

## Episode state

The Markov control state at the beginning of supervision transition `k` is:

```text
s_k = (x, y_k, z_k, k, d)
```

where:

- `x` is the fixed puzzle input;
- `y_k, z_k` are the frozen reasoner's explicit M11 recurrent execution state;
- `k` is the supervision-step index;
- `d` is the declared device-state vector.

The reasoner weights are not part of the trainable state.

## Joint action

One transition contains up to two policy decisions.

### Router action

For each token:

```text
a^route_{k,i} in {freeze, compute}
```

The M11 state-transition semantics are preserved. Frozen tokens keep their recurrent state exactly; the execution path decides which operations are physically sparse according to the M11 contract.

The pilot has a **forced all-active router warm-up at `k=0`**. This is an environment action, not a policy decision. Its log-probability and entropy receive zero actor credit.

For `k>0`, the router samples from `RLTokenRouter` during training and uses deterministic argmax during evaluation.

### Halter action

After the reasoner executes transition `k` and independent task correctness is checked, an eligible unsolved non-final example may choose:

```text
a^halt_k in {continue, halt}
```

No halter decision is made when:

- exact terminal correctness has already been reached; or
- the declared step budget has been reached.

The final budget stop is therefore forced and receives **no** policy-gradient credit.

## Independent terminal correctness

For the M12 Sudoku pilot, terminal task success is:

```text
model.verifier.sudoku_correct(puzzle, argmax(logits), box)
```

This symbolic gate is independent of the retained reference solution target. RL inputs intentionally contain puzzle inputs but no solution targets.

A successful puzzle terminates immediately after the transition that produced the valid solution. The halter is not credited for an environment-forced correctness termination.

## Termination versus truncation

Per-example status is explicit for every sampled transition:

```text
valid[k,b]       transition k was actually executed for example b
terminated[k,b]  true environment/policy terminal after transition k
truncated[k,b]   time-limit/model-budget truncation after transition k
```

`terminated` includes:

- independent exact task success;
- a voluntary halter stop, including an unsolved voluntary stop.

`truncated` means the episode did not terminally end but the declared rollout horizon ended. A truncation is allowed to bootstrap the critic from the final recurrent state.

After termination, later batch slots are logically inactive and contribute no reward, value loss, policy loss or entropy term. M12 makes no batched-compaction throughput claim; this is logical per-example episode accounting.

## Declared base objective

M12 optimizes a correctness/compute-proxy objective. It does **not** optimize measured energy.

For each valid transition:

```text
r_base = success_reward * 1[new exact success]
         - halt_failure_penalty * 1[voluntary halt while unsolved]
         - lambda_step * 1[transition executed]
         - lambda_token * active_density
```

Frozen pilot constants:

```text
success_reward       1.00
halt_failure_penalty 0.25
lambda_step          0.01
lambda_token         0.02
gamma                0.99
GAE lambda           0.95
```

`active_density = active_tokens / sequence_length` and the per-step charge are explicitly **logical compute proxies**. They are not joules and are not called energy.

Measured energy may enter a future objective only through SPECTRA's physical measurement protocol with retained measurement provenance. M12 records:

```text
cost_kind              logical_step_token_proxy_v1
measured_energy_used   false
measured_energy_joules null
```

## Grounded potential shaping

The strictly loaded frozen M07-style verifier supplies a state potential:

```text
Phi(s) = grounded_verifier.value_state(x, y, z)
```

M12 uses the policy-invariant potential form with the **same discount as the RL return**:

```text
F(s_k, s_{k+1}) = gamma * Phi(next_effective) - Phi(s_k)
```

Terminal treatment is explicit:

```text
Phi(next_effective) = 0            if terminated[k,b]
Phi(next_effective) = Phi(s_{k+1}) otherwise
```

A time-limit truncation is not treated as an absorbing terminal, so its real next-state potential is retained. This is the same terminal distinction used by the critic bootstrap.

Total reward:

```text
r_k = r_base + F
```

The verifier is a fixed potential only. Its probability is not treated as terminal correctness and receives no gradient.

## Per-example GAE contract

Inputs are `[K,B]` transition tensors plus `[K+1,B]` target-critic values.

TD bootstrap mask:

```text
bootstrap_mask = valid * (1 - terminated)
```

Thus a truncation bootstraps, while a true terminal does not.

The GAE trace-continuation mask is:

```text
trace_mask = valid * (1 - terminated) * (1 - truncated)
```

Thus a truncated rollout uses the final bootstrap in its local TD residual but does not propagate a nonexistent later sampled transition through the trace.

Invalid post-episode slots have exactly zero advantage/return contribution.

## Forced-action credit contract

Actor objectives carry explicit decision masks:

```text
router_decision[k,b] = 1 only if the router actually sampled that decision
halter_decision[k,b] = 1 only if the halter actually sampled that decision
```

Forced warm-up routing, correctness termination and final budget stopping have decision mask zero. Changing a forced action's synthetic/logged log-probability must not change the policy loss; this is a focused test.

## Optimized parameter sets

M12 creates:

```text
actor/router    RLTokenRouter
actor/halter    HaltingPolicy
critic/online   LatentValueHead
critic/target   deepcopy(LatentValueHead), Polyak updated
```

The optimizer owns **exactly**:

```text
router parameters U halter parameters U online critic parameters
```

It must own none of:

```text
reasoner parameters
verifier parameters
target critic parameters
```

The reasoner and verifier execute under no-grad/frozen semantics. Recurrent states are detached before policy/critic consumption. Policy gradient reaches actors only through sampled-action log-probabilities. Value regression reaches the online critic only. The target critic changes only through the explicit Polyak update.

M12 records parameter-name/id ownership and before/after tensor hashes for all five families.

## Loss

The joint actor-critic loss keeps router and halter terms separately observable:

```text
L = L_router_policy
  + L_halter_policy
  + value_coef * L_value
  - entropy_coef * (H_router + H_halter)
```

Frozen coefficients:

```text
value_coef   0.5
entropy_coef 0.01
target tau   0.02
grad clip    1.0
```

Policy losses are normalized only over actual decisions. Value loss is normalized only over valid executed transitions.

## Required hand-verified tests

Before the pilot can count as evidence, focused tests must include:

1. a tiny manually computed terminal rollout where final bootstrap is zero;
2. a tiny manually computed truncated rollout where final bootstrap is retained;
3. per-example mixed termination in the same batch;
4. potential shaping with `gamma*Phi(next)-Phi(current)` and zero terminal potential;
5. truncation retaining the real next potential;
6. forced-action log-probability having zero effect on actor loss;
7. a policy-halt terminal versus a budget truncation being distinguishable;
8. optimizer ownership excluding reasoner/verifier/target critic;
9. target critic having no gradient and changing only by Polyak update;
10. a real training update changing router, halter and online-critic tensors while reasoner/verifier hashes remain unchanged;
11. strict save/reload of the M12 adaptive-policy checkpoint.

## Versioned M12 checkpoint

The training command writes a strict checkpoint with:

```text
format  spectra.adaptive_rl
version 1
kind    router_halter_actor_critic
```

It records:

- router, halter, online critic and target critic states;
- reasoner checkpoint SHA-256;
- grounded verifier checkpoint SHA-256;
- task/architecture compatibility;
- training step/seed;
- all objective constants;
- termination/truncation semantics;
- cost proxy identity and explicit `measured_energy_used=false`;
- optimizer ownership summary;
- policy/reasoner/verifier tensor hashes.

The held-out comparison uses a strict reload of this checkpoint, not the in-memory policy objects.

## Bounded pilot

The CI pilot is intentionally small enough to be reproducible on CPU while exercising the real command.

Pilot fixture preparation:

```text
task                  generated unique 4x4 Sudoku
reasoner               trained TRM checkpoint, strict-reloaded
reasoner dim            32
reasoner layers          1
n / T / N_sup            1 / 1 / 4
reasoner train steps   120
verifier                full-state GroundedStateVerifier
verifier train steps   120
```

The fixture builder produces only training/held-out **puzzle inputs** for the RL command. Reference solutions are not passed into RL reward or evaluation.

RL pilot:

```text
RL updates             120
batch size              16
optimizer              AdamW
learning rate          2e-3
weight decay           0.01
seed                   20260912
step-0 router action   forced all-active, zero PG credit
reactivation policy    allow
```

The pilot may fail to improve. Collapse/no benefit is retained as a result rather than triggering post-hoc reward/threshold changes.

## Held-out comparison

The exact same held-out puzzle inputs, horizon and symbolic success gate are used for:

- `learned_rl`: strict-reloaded M12 router + halter;
- `fixed_depth`: all tokens active, no voluntary halt, full declared horizon;
- `heuristic`: M11 `ConfidenceRouter` with frozen predeclared settings and no learned halter.

Predeclared heuristic:

```text
threshold       0.90
warmup_steps    1
min_active_frac 0.25
```

For every controller M12 records separately:

```text
task success rate
mean structural score
mean realized supervision steps
mean active density over valid transitions
logical step cost
logical token cost
total logical compute proxy
```

No controller is called better solely because it uses less compute; quality and cost are reported separately.

## Training telemetry

Every retained training log records, at minimum:

```text
exact task success
realized steps
active density
router entropy
halter entropy
router policy loss
halter policy loss
value loss
total loss
step proxy cost
token proxy cost
total proxy cost
terminated fraction
truncated fraction
```

Energy is absent unless supplied by a separately retained physical measurement protocol.

## Acceptance gate

M12 passes as a **training-path milestone** only if all of the following hold:

```text
1. a real CLI training command strictly loads reasoner + grounded verifier checkpoints;
2. that command performs optimizer steps and changes router, halter and online critic tensors;
3. reasoner and verifier tensor hashes remain exactly unchanged;
4. target critic is optimizer-excluded/gradient-free and changes only by Polyak update;
5. hand-computed reward/GAE terminal and truncation cases pass;
6. forced actions receive no policy-gradient credit;
7. a strict versioned M12 checkpoint is written and reloaded;
8. held-out learned/fixed/heuristic controller behavior is compared under one protocol;
9. task success, steps, density, actor/value losses and cost are logged separately;
10. the full fast regression suite passes.
```

A held-out RL win is **not required** for the training-path acceptance gate. If learned control collapses, matches, or loses to simple baselines, M12 can still establish a correct training pipeline but the negative policy-quality result must be stated plainly.

## Claims boundary

A PASS establishes a grounded, executable and auditable RL training path for M11's adaptive policies.

It does **not** by itself establish:

- learned adaptive-control superiority;
- exact-solve improvement;
- latency speedup;
- energy reduction;
- GPU/large-batch efficiency;
- batched compaction;
- broad task generalization;
- learned-search benefit.

Those require separate evidence.
