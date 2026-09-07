# Milestone 12 Acceptance Gate — Grounded Router/Halter RL Training

**Decision: PASS as a training-path milestone.**

**Learned-control quality result: NEGATIVE / COLLAPSED.**

M12 establishes that SPECTRA now has a real, checkpoint-grounded, auditable actor-critic training path for the M11 router and halter. The accepted command strictly loads a trained reasoner and compatible grounded verifier, freezes both, executes real adaptive episodes, updates only the declared policies and online critic, maintains a gradient-free target critic by Polyak update, writes and strict-reloads a versioned adaptive-RL checkpoint, and compares the learned controller with fixed-depth and heuristic controls on held-out puzzle inputs.

The bounded pilot did **not** establish useful learned adaptive control. All three held-out controllers solved `0/96`. The learned controller used lower logical compute proxy but had a slightly worse structural score, did not halt early, and converged to an extremely low-activity router regime. That negative result is retained rather than retuning the preregistered objective after seeing the pilot.

Authoritative preregistration: [`M12_PROTOCOL.md`](M12_PROTOCOL.md).

## Accepted execution

```text
branch        research/m12-router-halter-rl
head          c54a719f8c30c958e58b4ecb32abb54a75de3cfd
Actions run   34154069653
job           101842113858
artifact      m12-router-halter-rl-evidence
artifact id   10030392531
ZIP SHA256    8a23f819b00e47e60fd6d7c7f246fb4f362be305253749f2a9c044fdd1f7bdc4
size          346,998 bytes
```

Execution gate:

```text
compile      0
focused      0
fixture      0
training     0
evidence     0
fast         0
focused      22 passed, 1 deselected
full fast    272 passed, 16 deselected, 1 warning
```

The warning is the already-known M10 test-only tensor-to-scalar warning. No M12 regression failed.

Retained evidence hashes:

```text
adaptive_rl.pt             ead326e0ee241008dedeed53beaa881e937fc445dcbb9330e958b438718ccffd
summary.json               6f05923fa400b54582ca73ec2e758f3a3509280454f766a0c0ec58f419c1b597
training_log.jsonl         fd59bba963ac690ea80f769cf923de7a0b921784edcad95b181c3a77575be8b4
heldout_comparison.json    6de45fcc7842466549ef89fcad1fb650d1b42aa1323eb39b1ecef7dc56199dfb
heldout_rows.json          4619c6bdc47d2017ffc7506049d4b7e5d1542ff2d117b25b9273e66a4f093361
ownership.json             3786b76a2d581cc56d64f9e8be91d859f54b7888d0f718b0d3cf433022c6e4cf
checkpoint_reload.json     7da27bce50528e4f296fd2f2c9e0f2863d723978c73d2e068c0f4300526332fd
reasoner.pt                495ad9f7ff1a30cb1cfc8e7cacaa994bb5ca24257287f83757d83b53b9a1c736
grounded_verifier.pt       af65d46981b8194b8d08660364de26fd79003ef552690c4b749f237d968bbf72
rl_inputs.pt               da8285841c7cd16765744d539339664fa472a8504e99fa9f16bfefc1a07aa967
```

---

## 1. This is real RL training, not a helper-only test

The accepted workflow executes the actual CLI:

```bash
python scripts/train_adaptive_rl.py \
  --reasoner-checkpoint .m12/fixture/reasoner.pt \
  --verifier-checkpoint .m12/fixture/grounded_verifier.pt \
  --inputs .m12/fixture/rl_inputs.pt \
  --out .m12/experiment \
  --steps 120 \
  --seed 20260912
```

The command completes 120 optimizer updates, emits 120 training-log rows, writes a strict `spectra.adaptive_rl` v1 checkpoint, strict-reloads that checkpoint, and evaluates the reloaded policies against two held-out control baselines.

The focused unit tests are necessary correctness evidence but are **not** counted as successful RL training by themselves.

## 2. Strict checkpoint grounding

The pilot first prepares and then strictly reloads a genuine training-state reasoner checkpoint:

```text
reasoner checkpoint SHA256   495ad9f7ff1a30cb1cfc8e7cacaa994bb5ca24257287f83757d83b53b9a1c736
dim                          32
layers                        1
heads                         4
n / T / N_sup                 1 / 1 / 4
training steps              120
precision                    FP32 CPU
```

The bounded reasoner is intentionally small. Its final validation result was:

```text
board accuracy      0.0
cell accuracy       0.169921875
```

This is a weak reasoner fixture. M12 therefore cannot be interpreted as evidence of strong Sudoku capability.

The compatible full-state grounded verifier is also strict-loaded:

```text
grounded verifier SHA256  af65d46981b8194b8d08660364de26fd79003ef552690c4b749f237d968bbf72
training steps            120
training state rows       640
positive rows              24
negative rows             616
final validation BCE       0.217111
validation accuracy        0.942708
```

The validation set is strongly class-imbalanced, so the `0.942708` accuracy is **not** treated as evidence of a strong value model. The verifier is accepted here as a strictly compatible frozen potential source, not as a quality claim.

The reasoner tensor-state hash was unchanged across verifier fixture training.

## 3. Target-free RL input artifact

The RL input artifact contains only puzzle inputs:

```text
RL training puzzles      256
RL held-out puzzles       96
reference targets present false
```

The puzzle pool is generated first, exact input duplicates are removed, and the retained rows are partitioned into disjoint reasoner-train, reasoner-validation, RL-train and RL-held-out subsets.

Solution targets are not passed to the RL reward or held-out controller evaluation.

## 4. Episode semantics

At transition `k`, the controller state is:

```text
s_k = (x, y_k, z_k, k, device_state)
```

The router acts before the recurrent transition. Step zero is a preregistered forced all-active warm-up and receives no policy-gradient credit.

After the reasoner executes the transition, the candidate answer is checked by the independent symbolic Sudoku verifier. Exact semantic success terminates the example immediately and bypasses the learned halter.

For an unsolved non-final example, the halter may choose continue or halt. A voluntary unsolved halt is a true terminal event with the declared failure penalty.

An unsolved example reaching the horizon is a **time-limit truncation**, not a terminal decision made by the halter.

## 5. Per-example termination and truncation

M12 carries explicit `[K,B]` masks:

```text
valid
terminated
truncated
```

The GAE TD bootstrap mask is:

```text
valid * (1 - terminated)
```

so a true terminal receives no next-value bootstrap while a time-limit truncation does.

The GAE trace mask is:

```text
valid * (1 - terminated) * (1 - truncated)
```

so truncation uses its real final bootstrap in the local TD residual but does not propagate a nonexistent later sampled transition through the trace.

Post-episode invalid slots contribute zero reward, return, actor loss, value loss and entropy.

## 6. Hand-computed rollout contracts

Focused tests include manually computed cases rather than only implementation-vs-implementation comparisons.

### True terminal

For:

```text
rewards = [1.0, 2.0]
values  = [0.5, 0.4, 9.0]
gamma   = 1
lambda  = 1
terminal at final sampled transition
```

the absurd terminal bootstrap value `9.0` is ignored and the expected targets are:

```text
advantages = [2.5, 1.6]
returns    = [3.0, 2.0]
```

### Time-limit truncation

For:

```text
rewards = [1.0, 2.0]
values  = [0.5, 0.4, 0.3]
gamma   = 1
lambda  = 1
truncation at final sampled transition
```

the final `0.3` bootstrap is retained and the expected targets are:

```text
advantages = [2.8, 1.9]
returns    = [3.3, 2.3]
```

A mixed-batch test additionally proves that an early-terminal example's later invalid slots cannot contaminate its return while another example in the same batch may continue to a truncation.

## 7. Grounded potential shaping

The reward uses the same discount for return and shaping:

```text
F = gamma * Phi(next_effective) - Phi(current)
```

where:

```text
Phi(next_effective) = 0          for a true terminal
Phi(next_effective) = Phi(next)  otherwise, including time-limit truncation
```

Focused hand tests verify both terminal-zeroing and truncation retention.

The grounded verifier receives no gradient and its probability is never treated as independent task correctness.

## 8. Declared compute objective — proxy, not energy

Base reward is:

```text
+ 1.00 * new exact success
- 0.25 * voluntary unsolved halt
- 0.01 * executed transition
- 0.02 * active density
```

The step and token terms are explicitly labeled:

```text
cost_kind               logical_step_token_proxy_v1
measured_energy_used    false
measured_energy_joules  null
```

M12 does not convert active-token or step counts into joules. A measured-energy objective requires the repository's physical energy-measurement protocol and retained measurement provenance.

## 9. Forced-action policy-gradient credit

M12 carries explicit decision masks for router and halter losses.

No actor credit is assigned to:

- forced all-active step-zero routing;
- exact-success environment termination;
- final horizon truncation.

A focused test changes a forced slot's synthetic log-probability from `+100` to `-700` and verifies that the policy loss is identical and the forced slot receives zero gradient.

The historical `halting_episode` helper was also fixed: the final horizon stop no longer makes a fake Bernoulli decision or receives REINFORCE credit.

## 10. Gradient and optimizer ownership

Trainable/optimizer-owned families:

```text
router          1,314 parameters
halter          1,281 parameters
online critic   1,281 parameters
```

Optimizer-excluded families:

```text
target critic   1,281 parameters
reasoner       13,576 parameters
grounded verifier 15,489 parameters
```

The optimizer owned exactly 12 parameter tensors, equal to the union of router + halter + online critic and with zero forbidden overlap.

During every real training update the command checks that reasoner, verifier and target critic have no gradients.

Before/after tensor-state hashes:

```text
router
  before  2f7137f4ab275ad8368cd6ae485175bc9fbe18e307315c38670bffb3711a48b2
  after   e6f13c38206671a11d6cf3ff054d7f4ad411428583fa296448837e921defd669

halter
  before  55232637161d6af0edfe6659c4005697af638f6c6b4c578bc43600850adfd392
  after   859b3cfbfb514f572ac73842f4d83e3b3163757b48c0edc4c1ebf3f02b0cfcb9

online critic
  before  e01d5986a80d13e3fd710df8218676fbb95e0dc82f7c786ee2caa0bac98677a9
  after   aa44a2de49ae66784ffe996cf479d3008045b46c166fe74f7d2f6f88f81c5058

target critic
  before  e01d5986a80d13e3fd710df8218676fbb95e0dc82f7c786ee2caa0bac98677a9
  after   08a1eb97bc6167d5b6c595d93f0c43769683a8a3f5bed10bc4f2bef08c5f22d5

reasoner
  before  8bfcc2ab523175d821a27659fe079163899627e9a2e8acdad0ee79666e048510
  after   8bfcc2ab523175d821a27659fe079163899627e9a2e8acdad0ee79666e048510

verifier
  before  581c262e37b8023e55f77bab38def34c94a6f556b678e29c381e5edba1ee45c5
  after   581c262e37b8023e55f77bab38def34c94a6f556b678e29c381e5edba1ee45c5
```

This establishes real actor/critic updates while the recursive core and grounded verifier remain exactly frozen. The target critic changes only through the explicit Polyak update.

## 11. Strict M12 checkpoint

The trained adaptive controller is stored as:

```text
format   spectra.adaptive_rl
version  1
kind     router_halter_actor_critic
SHA256   ead326e0ee241008dedeed53beaa881e937fc445dcbb9330e958b438718ccffd
```

It binds to the exact reasoner and grounded-verifier checkpoint hashes and records objective, termination/truncation, cost-proxy and ownership metadata.

Strict reload reproduced the exact router, halter, online critic and target critic tensor states before the held-out comparison.

## 12. Training telemetry

The accepted command retained one log row for each of the 120 optimizer updates. Every row records separately:

- exact task success;
- realized steps;
- active density;
- router entropy;
- halter entropy;
- router policy loss;
- halter policy loss;
- value loss;
- total loss;
- step proxy cost;
- token proxy cost;
- total proxy cost;
- terminal fraction;
- truncation fraction;
- gradient norm;
- explicit null measured-energy field.

Mean over the final ten updates:

```text
task success          0.000000
realized steps        4.000000
active density        0.253516
router entropy        0.033420
halter entropy        0.008408
router policy loss   -0.069069
halter policy loss    0.000029
value loss            0.000081
total loss           -0.069418
```

The very low actor entropies and near-minimal effective active density are consistent with policy collapse rather than a rich adaptive strategy.

## 13. Held-out controller comparison

All controllers use the same 96 held-out puzzle inputs, symbolic success check, four-step maximum horizon and cost definition.

| Controller | Success | Structural score | Steps | Active density | Step proxy | Token proxy | Total proxy |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed depth | 0/96 | 0.0378689 | 4.0 | 1.00 | 0.04 | 0.08 | 0.12 |
| heuristic | 0/96 | 0.0378689 | 4.0 | 1.00 | 0.04 | 0.08 | 0.12 |
| learned RL | 0/96 | 0.0348824 | 4.0 | 0.25 | 0.04 | 0.02 | 0.06 |

Deltas for learned RL versus fixed depth:

```text
success rate       +0.000000
structural score   -0.0029865
logical proxy      -0.0600000
```

Every held-out example under every controller ended as `budget_truncated`. The learned halter therefore produced **no held-out early-exit benefit**.

The learned controller's mean active density is exactly `0.25`: with the forced all-active first transition included in the four-step mean, this is consistent with the router collapsing toward freezing almost all optional later work. It cut the declared logical proxy in half but also slightly reduced the already-low structural score.

**No learned-control superiority claim is supported.**

## 14. Failed first end-to-end attempt retained

The first full M12 workflow attempt (`34153655193`) did not count as training success. It reached fixture preparation but the real RL command failed before its first optimizer update because a Python boolean horizon flag was inverted with bitwise `~`, producing integer `-1/-2` and promoting a decision mask to `Long`.

The accepted fix replaced that implicit Python operation with an explicit boolean tensor horizon mask and also casts symbolic success to boolean at the episode boundary. The preregistered reward, hyperparameters, baselines and acceptance criteria were not changed after seeing the failure.

The accepted run is the later clean run `34154069653`.

## 15. What M12 establishes

M12 establishes:

- a real grounded router/halter RL training command;
- strict reasoner and grounded-verifier checkpoint loading;
- independently verified symbolic terminal correctness;
- explicit episode state/action/termination/truncation semantics;
- per-example terminal/truncation-aware GAE;
- discounted potential shaping with correct terminal treatment;
- no invented actor credit for forced actions;
- exact optimizer/gradient ownership boundaries;
- a gradient-free Polyak target critic;
- hand-computed rollout/return tests;
- a strict versioned adaptive-RL checkpoint and strict reload;
- retained policy/value/cost telemetry;
- held-out comparison with fixed and heuristic controls;
- honest negative/collapse evidence from the bounded pilot.

M12 does **not** establish:

- useful learned adaptive control;
- early-halting benefit from the trained halter;
- exact Sudoku solve improvement;
- a strong reasoner or verifier;
- latency speedup;
- measured-energy reduction;
- whole-model sparse scaling;
- batched-compaction efficiency;
- broad task generalization;
- learned-search benefit.

## M12 decision

**PASS — training path only.** The user acceptance gate is satisfied: a real training command updates the intended policies, independently verified rollout cases produce the declared targets, strict policy checkpoints are written/reloaded, and held-out learned behavior is compared to simple controls.

**Policy-quality conclusion: negative.** The bounded pilot collapsed toward low token activity, produced no early halting, solved no held-out examples, and slightly reduced structural score while reducing only the declared logical compute proxy.

**Stop here for M12. Do not begin M13 automatically.**
