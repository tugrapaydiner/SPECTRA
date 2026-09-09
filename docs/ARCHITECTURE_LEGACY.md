> **Note.** Source-code comments reference this document by its legacy name *BLUEPRINT* (e.g. "BLUEPRINT section 18"); the section numbering is unchanged.

# SPECTRA Implementation Blueprint

**Sparse Policy-guided Energy-aware Cache-Ternary Recursive Agent**

Version: 0.6 - Research Rebrand: Edge Test-Time Scaling Laws, Dense RL, and Infinite Synthetic Flywheel
Purpose: A buildable, test-gated blueprint for implementing **SPECTRA**, a self-improving edge reasoning agent that tests whether learned test-time search can substitute for parameter count under strict physical energy budgets.

---

## 0. One-Sentence Project Goal

Build a tiny self-improving reasoning agent that can test the central research hypothesis: **learned test-time compute can substitute for parameter count under a fixed physical joule budget**. The system solves symbolic/grid/local planning tasks using:

- **W1.58A8 ternary computation**: 1.58-bit ternary weights, INT8 activations.
- **INT8 latent-state recursion**: keep `x`, `y`, and `z` cheap inside the loop.
- **RL-driven lazy token routing**: learn which tokens deserve compute instead of using fixed entropy masks.
- **RL-driven hardware-aware halting**: learn when to stop under battery, thermal, latency, RAM, and cgroup limits.
- **Cache-resident Latent MCTS**: run AlphaZero-style test-time search inside INT8 latent state `z` before decoding.
- **Dense verifier-bootstrapped RL**: use the neural energy verifier as a per-step value network, not only a final answer judge.
- **Open-ended System 2 -> System 1 self-play flywheel**: generate new tasks, solve them with latent search, and compile successful trajectories into fast reflexes.
- **Compute-optimal edge scaling laws**: map parameter count vs recursion depth vs active-token count vs physically measured joules.
- **Physical energy profiling**: report accuracy per physically measured joule, not only FLOPs.

---

## 1. Non-Negotiable Build Principles

These rules are mandatory.

1. **MVP first, frontier last.**  
   A working FP16 TRM that overfits one batch is more valuable than a half-built W1.58A8 sparse kernel that does not train.

2. **Every phase has a test gate.**  
   Do not proceed to the next phase unless the current gate passes.

3. **Never optimize a broken model.**  
   No `bitnet.cpp`, sparse SIMD, or hardware deployment until PyTorch correctness is proven.

4. **Separate algorithm validation from hardware validation.**  
   - PyTorch: correctness, learning, ablations.
   - FakeBitLinear: ternary training viability.
   - `bitnet.cpp`: dense W1.58A8 CPU baseline.
   - Custom C++/SIMD: sparse lazy-routing kernel only after active-token density is low enough.

5. **Physical energy beats FLOP estimates.**  
   FLOPs/MACs are diagnostics only. The paper claim must use measured joules.

6. **Do not make ternary Mamba critical.**  
   Initial deployment uses ternary attention / ternary MLP-Mixer / ternary gated MLP blocks. Ternary Mamba/SSM is future work.

---

## 2. Final Project Name and Claim

### Name

**SPECTRA**  
**Sparse Policy-guided Energy-aware Cache-Ternary Recursive Agent**

Implementation codename kept for compatibility: `spectra`.

### Full Title

**SPECTRA: Sparse Policy-guided Energy-aware Cache-Ternary Recursive Agent**

### One-Line Pitch

> To bypass the memory bandwidth bottlenecks of legacy hardware, we introduce **SPECTRA**, a unified reasoning framework that confines AlphaZero-style test-time search entirely within the L3 cache.

### Core Claim

> We propose **SPECTRA** - a **Sparse Policy-guided Energy-aware Cache-Ternary Recursive Agent** - as an autonomous, self-improving edge reasoning system designed to test a frontier hypothesis: **learned, cache-resident test-time search can partially substitute for parameter count under a fixed physical joule budget**. SPECTRA uses W1.58A8 ternary computation, RL-driven token routing and halting, dense verifier-bootstrapped rewards, Cache-Resident Latent MCTS, and an open-ended synthetic self-play flywheel to discover compute-optimal scaling laws on constrained legacy x86 hardware.
>
> The target is not merely battery savings. The target is to map when a tiny model plus learned latent search can outperform much larger zero-shot models under the same measured joule budget.

### Central Hypothesis

```text
For fixed physical energy E, there exists a compute-optimal frontier over:
  parameter count P,
  recursion depth K,
  active-token density |A_k|,
  latent-search rollouts M,
  and verifier strength V.

SPECTRA empirically estimates this frontier on constrained legacy hardware.
```

### What Would Make This Breakthrough-Level?

A strong result is not only "uses less energy." A strong result is:

```text
A 1M-7M parameter W1.58A8 agent with learned latent test-time search
matches or beats much larger zero-shot baselines
on structured reasoning tasks
under the same measured joule budget.
```

This is the scientific story: **parameter scaling is not the only path to intelligence; learned cache-resident test-time compute may be a second axis of scaling.**

---

## 3. Five Pillars

### Pillar 1 — Efficient Architecture

A **W1.58A8 recursive core**:

| Component | Precision |
|---|---|
| Recursive core weights | W1.58 ternary values in `{-1, 0, +1}`, packed into 2-bit storage |
| Recursive activations | INT8 |
| Accumulation | INT16 or INT32 |
| Norm/scales/control heads | FP16/FP32 initially |
| Embeddings | FP16/INT8 depending on stability |
| Verifier head | Higher precision first; quantize later only if safe |

### Pillar 2 — Efficient Computation

- RL-driven lazy active-token routing.
- RL-driven hardware-aware halting.
- Cache-resident Latent MCTS over INT8 latent states.
- Compute-optimal edge scaling law experiments.
- No fixed entropy threshold is considered final; heuristics are prototype baselines only.

### Pillar 3 — Efficient Deployment

Deployment path:

```text
PyTorch FP16 correctness
→ PyTorch FakeBitLinear / STE ternary training
→ dense bitnet.cpp CPU baseline
→ custom fused C++/SIMD sparse ternary kernel
→ optional NPU/GPU backends
```

### Pillar 4 — Stable Training

A **Stability Shield** against ternary-recursive collapse:

- FP16 teacher.
- Fake-ternary student.
- Progressive layerwise ternarization.
- EMA.
- Recursive residual scaling.
- Per-step normalization.
- Gradient clipping.
- Collapse detectors.
- Teacher-trajectory distillation.

### Pillar 5 - Edge-Native Evaluation

Tasks:

1. Sudoku / Maze: symbolic reasoning proof.
2. ARC / ARC-style tasks: abstract grid reasoning.
3. BabyAI-style local planning: grounded sequential reasoning.
4. Smart-home state resolution: realistic local edge decision-making.

Primary execution hardware:

- Legacy x86 laptop / decade-old CPU.
- Linux `cgroups` to simulate catastrophic edge constraints:
  - Catastrophic Edge: 1 CPU core, 512 MB RAM.
  - Standard Edge: 2 CPU cores, 2 GB RAM.
- Intel/AMD RAPL energy profiling to measure actual silicon joules.

This blueprint deliberately avoids claiming access to unavailable external edge devices as required baselines. Those can be future extensions, but the core paper claim must be executable on the available constrained legacy laptop setup.

Metrics:

- Accuracy.
- Latency.
- Peak RAM.
- Physical joules.
- Accuracy per joule.
- Accuracy per MB.
- Accuracy per millisecond.

---

## 4. Architecture Overview

```text
Input symbolic/grid/local-planning problem
        ↓
Tiny spatial/symbolic encoder
        ↓
System 1 self-play-distilled student
        ↓
Cheap symbolic verifier
        ↓
If confident:
    return answer
        ↓
If uncertain:
    W1.58A8 recursive ternary core
        ↓
RL-driven lazy active-token router
        ↓
INT8 latent-state recursion
        ↓
RL-driven hardware-aware halting
        ↓
Cache-resident Latent MCTS + verifier-guided answer selection
        ↓
Output + confidence + measured/estimated cost metadata
```

---

## 5. Core Recursive Reasoning Math

### 5.1 State Variables

Let:

- `x`: encoded input problem.
- `y`: current answer state.
- `z`: latent reasoning state.
- `f_θ`: shared recursive operator.
- `n`: inner latent recursion steps.
- `T`: deep recursion cycles per supervision step.
- `N_sup`: number of deep supervision steps.

Shapes:

```text
x ∈ R[B, L, D]
y ∈ R[B, L, D]
z ∈ R[B, L, D]
```

Where:

- `B` = batch size. Training commonly uses `B=128` or `B=256`; edge deployment is **strictly `B=1`** and therefore becomes GEMV/memory-bandwidth bound rather than GEMM/compute-bound.
- `L` = sequence length.
  - Sudoku: `L = 81`.
  - ARC padded grid: `L = 900` for 30×30.
- `D` = hidden dimension.

### 5.2 Basic TRM Recursion

One recursive cycle:

```text
for i in 1..n:
    z ← f_θ(x, y, z)

y ← f_θ(y, z)
```

A full supervision step:

```text
for t in 1..T:
    y, z ← recursive_cycle(x, y, z)

logits ← decode(y)
halt_logit ← halt_head(y)
```

Detach between deep supervision steps:

```text
y ← stop_gradient(y)
z ← stop_gradient(z)
```

This prevents unbounded backprop through the entire recurrence.

### 5.3 Effective Depth

```text
effective_depth = T × (n + 1) × n_layers
```

Example:

```text
T = 3
n = 6
n_layers = 2

effective_depth = 3 × 7 × 2 = 42
```

### 5.4 Deep Supervision Loss

```math
\mathcal{L}_{deep}
=
\frac{1}{N_{sup}}
\sum_{k=1}^{N_{sup}}
\left[
CE(\mathrm{decode}(y^{(k)}), y^*)
+
\lambda_h \cdot BCE(h^{(k)}, c^{(k)})
\right]
```

Where:

```math
c^{(k)} = \mathbf{1}\left[\arg\max(\mathrm{decode}(y^{(k)})) = y^*\right]
```

### 5.5 Dense RL Reward, Verifier Bootstrapping, and Edge Compute Objective

A sparse final reward is too high-variance for long recursive reasoning. The router, halter, and latent search controller therefore use **dense value bootstrapping** from the neural verifier at every recursion/search step.

Final sparse reward:

```math
R_{final}
=
\mathbf{1}[\text{Correct Answer}]
-
\lambda_E \cdot E_{measured}
```

Per-step dense value from the verifier:

```math
V_k
=
- E_\psi(x, z^{(k)})
```

Dense temporal-difference style reward:

```math
r_k
=
(V_{k+1} - V_k)
-
\lambda_{tokens}|A_k|
-
\lambda_{step}
-
\lambda_E \Delta E_k
```

Return for the routing/halting/search policy:

```math
G_k
=
\sum_{t=k}^{K} \gamma^{t-k} r_t
+
\gamma^{K-k} R_{final}
```

REINFORCE policy gradient objective:

```math
\nabla_\phi J(\phi)
=
\mathbb{E}\left[
\sum_{k=1}^{K}
\nabla_\phi \log \pi_\phi(a^{(k)}|s^{(k)})
\left(G_k - b(s^{(k)})\right)
\right]
```

where:

- `a_i^{(k)} = 1` means token `i` is routed/updated at recursion step `k`.
- `a_i^{(k)} = 0` means token `i` is frozen.
- `A_k = {i : a_i^{(k)} = 1}` is the active-token set.
- `E_measured` is physical energy measured by RAPL during the run.
- `Delta E_k` is incremental energy for step `k` when available, or a calibrated per-step estimate.
- `b(s)` is a learned baseline/value head to reduce variance.
- `V_k = -E_psi(x,z^k)` gives immediate credit assignment before final decoding.

This prevents the router from waiting until the final answer to learn whether a token-freeze decision was good. The model receives a local signal whenever the latent state becomes more or less verifier-compatible.

Compute-optimal edge scaling model:

```math
E_{infer}
\approx
\sum_{k=1}^{K}
\alpha_{device} \cdot |A_k| \cdot P
+
\beta_{device} \cdot M
+
E_{overhead}
```

where:

- `P = N_params` is parameter count.
- `K` is recursive depth.
- `|A_k|` is active-token count at step `k`.
- `M` is Latent MCTS rollout count.
- `alpha_device` is measured joule-per-active-token-per-parameter for B=1 W1.58A8 GEMV.
- `beta_device` captures per-rollout search overhead.
- `E_overhead` captures gather/scatter, RAPL/cgroup overhead, cache misses, and OS noise.

Main scaling-law question:

```text
For a fixed physical joule budget, is it better to use:
  a larger model with fewer search steps,
  or a smaller model with more recursive/latent MCTS test-time compute?
```

Deliverable:

```text
Compute-optimal frontier curves:
  accuracy vs measured joules,
  parameter count vs recursion depth,
  active-token density vs rollout count,
  and model size vs test-time search.
```

### 5.5.1 GAE-λ Dense Value Bootstrapping (Router and Halter Objective)

The terminal-reward REINFORCE estimator of Section 20.2 is retained **only as an
ablation baseline**. The production objective for both the lazy router and the
hardware-aware halter is a verifier-bootstrapped **actor–critic with Generalized
Advantage Estimation (GAE-λ)**, which assigns per-step credit rather than a single
terminal scalar. Fixed entropy/confidence masks remain debugging baselines only.

A learned latent critic `V_φ(z^{(k)}, d)` estimates the value of each latent
reasoning state, conditioned on the device state `d`. The dense per-step reward
uses the (frozen) verifier value `V^ψ_k = -E_ψ(x, z^{(k)})`:

```math
r_k = \left(V^\psi_{k+1} - V^\psi_k\right) - \lambda_{tokens}\frac{|A_k|}{L} - \lambda_{step} - \lambda_E\,\Delta E_k
```

The temporal-difference residual and GAE-λ advantage/return are:

```math
\delta_k = r_k + \gamma\,\bar V_\phi\!\left(z^{(k+1)}\right) - \bar V_\phi\!\left(z^{(k)}\right)
```

```math
A_k = \sum_{l=0}^{K-k} (\gamma\lambda)^l\,\delta_{k+l}, \qquad R_k = A_k + \bar V_\phi\!\left(z^{(k)}\right)
```

Because the critic's input `z` is produced by the **shared** recursive core `f_θ`,
which is itself being updated, the TD targets are violently non-stationary. Two
stop-gradients are therefore **mandatory**:

1. **Target network.** The bootstrap value `\bar V_φ` is a *frozen*, Polyak-averaged
   copy of the critic, updated `\bar V_\phi \leftarrow (1-\tau)\bar V_\phi + \tau\,V_\phi`,
   so the regression targets do not chase the live critic.
2. **Latent stop-gradient.** The critic regresses on `\mathrm{sg}[z]` (`z.detach()`),
   so the value loss never warps the actor's representation; only the policy
   gradient shapes `f_θ`.

Actor–critic loss (advantages standardized across the batch, entropy bonus
`\mathcal{H}`):

```math
\mathcal{L}_{AC} = -\,\mathbb{E}\!\left[\hat A_k\,\log\pi_\phi\!\left(a^{(k)}\mid s^{(k)}\right)\right]
+ c_v\,\mathbb{E}\!\left[\left(V_\phi\!\left(z^{(k)}\right) - \mathrm{sg}[R_k]\right)^2\right]
- c_e\,\mathbb{E}\!\left[\mathcal{H}(\pi_\phi)\right]
```

Reference implementation: `train/rl.py` (`compute_gae`, `dense_step_rewards`,
`dense_actor_critic_loss`, `rollout_router_gae`, `make_target_critic`,
`soft_update`); critic head `model/lazy_router.py::LatentValueHead`.

### 5.6 Recursive Residual Scaling

For stability, do not overwrite states fully early in training.

```math
z^{(k+1)} = RMSNorm(z^{(k)} + \alpha_z \cdot \Delta z^{(k)})
```

```math
y^{(k+1)} = RMSNorm(y^{(k)} + \alpha_y \cdot \Delta y^{(k)})
```

Starting values:

```text
α_z = 0.1 to 0.3
α_y = 0.1 to 0.3
```

Later these can be learned or scheduled upward.

---

## 6. W1.58A8 Ternary Computation

### 6.1 Weight Quantization

Weights are ternary:

```math
W_q \in \{-1, 0, +1\}
```

A common absmean-style quantization:

```math
\gamma = \frac{1}{n}\sum_i |W_i|
```

```math
\tilde{W}_i = \frac{W_i}{\gamma + \epsilon}
```

```math
W_{q,i} = clip(round(\tilde{W}_i), -1, 1)
```

Forward:

```math
Y = X \cdot (\gamma W_q)
```

Training keeps a full-precision master weight:

```text
W_master: FP16/FP32
W_forward: ternary
gradient: STE into W_master
```

### 6.2 Straight-Through Estimator

Because `round()` is not differentiable, use STE:

```math
\frac{\partial W_q}{\partial W} \approx 1
```

In PyTorch:

```python
w_q = ternarize(w)
w_ste = w + (w_q - w).detach()
```

This means:

- Forward uses ternary `w_q`.
- Backward updates full-precision `w`.

### 6.3 Activation Quantization

Do not leave recursive states in FP16/FP32 during deployment.

Deployment target:

```text
W1.58A8
```

Meaning:

```text
weights: ternary packed 2-bit
activations: INT8
accumulation: INT16 or INT32
```

For an activation tensor `x`:

```math
s_x = \frac{\max(|x|)}{127}
```

```math
x_q = clip(round(x / s_x), -128, 127)
```

```math
x \approx s_x \cdot x_q
```

In early training, use fake quantization:

```python
x_q = torch.clamp(torch.round(x / scale), -128, 127)
x_deq = x_q * scale
x_ste = x + (x_deq - x).detach()
```

### 6.4 Why W1.58A8 Matters

Raw ternary weight storage for a 7M parameter model:

```math
7,000,000 \times 1.58 \text{ bits}
=
11,060,000 \text{ bits}
\approx
1.38 \text{ MB}
```

But this is **only raw weight cost**. Real deployment also includes:

- Packing overhead.
- Scale factors.
- Embeddings.
- Norms.
- Output heads.
- Activation buffers.
- Runtime alignment.

Correct claim:

> A 7M-parameter ternary core has roughly 1.4 MB raw weight cost, making cache/SRAM-friendly recursive inference plausible after packing and activation optimization.

---

## 7. Lazy Active-Token Routing

### 7.1 Motivation

Dense recursion updates all tokens every step:

```text
ARC: L = 900 tokens
Sudoku: L = 81 tokens
```

But many tokens become confident early. Updating all tokens wastes compute.

### 7.2 Masked Update Formula

Let `m` be a binary mask:

```math
m_i =
\begin{cases}
1 & \text{if token } i \text{ is uncertain}\\
0 & \text{if token } i \text{ is confident}
\end{cases}
```

Lazy recursion:

```math
z_i^{(k+1)}
=
m_i \cdot f_\theta(x_i, y_i, z_i^{(k)})
+
(1 - m_i) \cdot z_i^{(k)}
```

Vector form:

```math
z^{(k+1)}
=
m \odot f_\theta(x, y, z^{(k)})
+
(1 - m) \odot z^{(k)}
```

### 7.3 RL-Driven Token Routing with Dense Q-Value Bootstrapping

Heuristic entropy masks fail to adapt to dynamic edge degradation. The Lazy Router is therefore framed as an RL policy, not a fixed threshold rule.

**Policy network:**

```math
\pi_\phi(a_i^{(k)} | z_i^{(k)}, d)
```

outputs a binary action per token:

```text
1 = route / compute token
0 = freeze token
```

where `d` is the device-state vector:

```text
cgroup CPU limit
RAM limit
RAPL power state
latency budget
thermal/power mode if available
```

**Dense bootstrapped reward:**

```math
r_k
=
\left[-E_\psi(x,z^{(k+1)}) + E_\psi(x,z^{(k)})\right]
-
\lambda_{tokens}|A_k|
-
\lambda_E \Delta E_k
```

**Final reward:**

```math
R_{final}
=
\mathbf{1}[\text{Correct Answer}]
-
\lambda_E E_{measured}
```

**Policy objective:**

```math
J(\phi)
=
\mathbb{E}\left[
\sum_{k=1}^{K}
\log \pi_\phi(a^{(k)}|s^{(k)})
\left(G_k-b(s^{(k)})\right)
\right]
```

This is the key research upgrade: the router does not wait until the final output to learn. The verifier supplies a dense value estimate at every latent step, allowing the model to learn which token updates immediately improve the reasoning state.

The intended behavior is learned, not hand-coded:

```text
easy/background tokens -> freeze early
ambiguous logical bottlenecks -> keep active
low energy budget -> route fewer tokens
hard task + plugged-in mode -> allow deeper latent search
```

Prototype note: fixed entropy/confidence masks are allowed only as debugging baselines. They are not the final method.

### 7.4 Router Must Actually Save FLOPs

A mask alone does not save compute if the dense model still runs all tokens.

Correct implementation:

```text
1. Compute confidence per token.
2. Select active token indices.
3. Gather active token states.
4. Run recursive update only on active tokens.
5. Scatter updated states back.
6. Freeze confident tokens.
```

### 7.5 PyTorch Prototype Warning

PyTorch gather/scatter timing is not reliable for final claims. Python overhead may erase speedups.

Use PyTorch lazy routing only to measure:

- Accuracy impact.
- Active token percentage.
- Router stability.
- How quickly active tokens shrink.
- Whether sparse execution is worth custom kernels.

### 7.6 Kernel-Go Decision Rule

Only build a custom sparse ternary kernel if:

```text
mean_active_token_density < 30–40%
```

without unacceptable accuracy loss.

Example:

```text
ARC L=900
active tokens drop to 50–200
→ custom kernel justified

active tokens drop only to 700
→ not worth kernel work yet
```

---

## 8. RL-Driven Hardware-Aware Halting

The halter is not a hardcoded threshold in the final system. The final controller is a learned policy that optimizes accuracy under measured or profiled joule budgets.

Policy:

```math
\pi_\eta(h_k \mid y^{(k)}, z^{(k)}, d) \rightarrow \{\text{continue}, \text{halt}\}
```

Reward:

```math
R_{halt} = \mathbf{1}[\text{Correct Answer}] - \lambda_J \cdot \text{MeasuredJoules} - \lambda_T \cdot \text{LatencyMs}
```

Heuristic halt heads remain useful for warm-starting, debugging, and ablation, but the DeepMind-tier target is RL-learned compute rationing.

### 8.1 Device State Input

Pass a device state vector into the model:

```text
d = [
    battery_level,
    thermal_level,
    latency_budget_ms,
    power_mode,
    available_ram_mb,
    device_class_id
]
```

Normalize all values to `[0, 1]`.

Examples:

```text
battery_level: 1.0 = full, 0.0 = empty
thermal_level: 0.0 = cool, 1.0 = severe throttling
latency_budget_ms: normalized against max expected latency
```

### 8.2 Halting Probability

At step `k`:

```math
p_{halt}^{(k)} = \sigma(g_\phi(pool(y^{(k)}), d))
```

### 8.3 Hardware-Bound Compute Penalty

Total loss:

```math
\mathcal{L}
=
\mathcal{L}_{task}
+
\lambda_{halt}\mathcal{L}_{halt}
+
\lambda_{compute}(d)\mathcal{L}_{compute}
```

Where:

```math
\mathcal{L}_{compute}
=
\sum_k
\left[
c_{step}
+
c_{token}\cdot \frac{|A_k|}{L}
+
c_{joule}\cdot \hat{E}_k
\right]
```

`A_k` = active token set at step `k`.

Dynamic compute penalty:

```math
\lambda_{compute}(d)
=
\lambda_0
\cdot
\exp
(
a(1 - battery)
+
b(thermal)
+
c(1 - latency\_budget)
)
```

Interpretation:

- Plugged in / cool / high latency budget → think longer.
- Low battery / thermal throttling / low latency budget → stop earlier.

### 8.4 Halting Collapse Detectors

Log:

```text
average_halt_step
halt_entropy
accuracy_at_halt
p_halt_distribution
```

Bad signs:

```text
always halts at step 1
always halts at max step
high halt confidence while answer is wrong
```

---

## 9. Verifier-Guided Inference

### 9.1 Two-Stage Verifier

Use cheap symbolic verification first.

```text
cheap verifier → neural verifier → candidate selection
```

### 9.2 Sudoku Verifier

Check:

```text
row validity
column validity
3×3 box validity
clue preservation
```

Score:

```math
S_{sudoku}(y) =
w_r V_{row}
+
w_c V_{col}
+
w_b V_{box}
+
w_g V_{given}
```

### 9.3 Maze Verifier

Check:

```text
path starts correctly
path ends at goal
path is continuous
path avoids walls
no illegal jumps
```

### 9.4 ARC-Style Soft Verifier

Check:

```text
valid output size
valid color set
object-count consistency
symmetry consistency
transformation consistency
background preservation
connected-component plausibility
```

### 9.5 Neural Energy Verifier

Energy head:

```math
E_\psi(x, y) \in \mathbb{R}
```

Lower energy = better candidate.

Contrastive loss:

```math
\mathcal{L}_{energy}
=
\max(0, margin + E_\psi(x, y^+) - E_\psi(x, y^-))
```

Use **hard negatives**:

```text
wrong candidates produced by the model itself
near-miss Sudoku boards
invalid maze paths that almost reach the goal
ARC outputs with plausible but wrong transformations
```

Do not rely only on random wrong negatives.

### 9.6 Cache-Resident Latent MCTS

Generating full text candidates on extreme edge devices is memory-prohibitive. Because our W1.58A8 core is only ~1.4 MB, the entire recursive model fits inside the L3 cache. We exploit this to perform **Latent Monte Carlo Tree Search (MCTS)**.

Instead of decoding, we build a search tree entirely in the INT8 latent space ($z$):

1. **Selection:** Traverse latent nodes maximizing PUCT (Predictor + Upper Confidence Tree).
2. **Expansion:** Use the recursive core $f_\theta$ to step to the next latent state $z_{k+1}$.
3. **Evaluation:** Use the Neural Energy Verifier $E_\psi(x, z)$ as the Value Network to score the latent state *without decoding*.
4. **Backpropagation:** Update node values.

This compresses DeepMind's AlphaProof/AlphaGeometry search paradigms into a latency-bound edge framework.

### 9.7 PUCT with Epistemic Lower-Confidence Bound

Naive PUCT treats the value network as a perfect oracle. It is not: the energy
verifier has both **aleatoric** uncertainty (the task may be unsolvable) and
**epistemic** uncertainty (the verifier has not seen this latent). Because the
discrete codebook actions can push `z` out of distribution, an under-trained
verifier can assign falsely low energy (high value) to a degenerate latent, and a
naive search will expand that branch without bound.

The value head is therefore a **deep ensemble** `{E_ψ^{(1)}, …, E_ψ^{(M)}}` whose
disagreement is a calibrated epistemic signal:

```math
\mu(z) = \frac{1}{M}\sum_{m=1}^{M} E_\psi^{(m)}(x,z), \qquad
\sigma(z) = \sqrt{\frac{1}{M}\sum_{m=1}^{M}\big(E_\psi^{(m)}(x,z) - \mu(z)\big)^2}
```

Selection replaces the raw value with a **pessimistic lower-confidence bound**:

```math
V_{\text{LCB}}(z) = -\big(\mu(z) + \beta\,\sigma(z)\big)
```

```math
a^\star = \arg\max_{a}\left[\; Q_{\text{LCB}}(s,a) \;+\; c_{puct}\,P(s,a)\,\frac{\sqrt{N(s)}}{1+N(s,a)} \;\right]
```

In-distribution latents have small `σ` and are unaffected; OOD latents incur a
large `β σ` penalty and are not chased. Implemented in `model/energy.py`
(`EnsembleLatentEnergyVerifier.value_with_uncertainty`) and `eval/latent_mcts.py`
(`LatentNativeMCTS`, `uncertainty_beta`).

**Precision boundary (non-negotiable).** The search statistics `(N, W, Q, P)` are
kept in FP32/FP64; the *environment* latent `z` is stored as literal INT8 codes
plus an FP32 per-token scale (`_LatentNode.z_codes`/`z_scale`). PUCT never
accumulates a Q-value inside the quantized space, and `out_head` is invoked
exactly **once** — to decode the single winning latent at the end of search.

---

## 10. Open-Ended Asynchronous Self-Play Flywheel

Static distillation is not enough. A finite dataset eventually saturates. SPECTRA uses an **open-ended synthetic task generator** so the flywheel keeps producing new training pressure while the device is idle or plugged in.

### 10.1 The Night-Shift Loop

When hardware constraints are lifted, the agent enters **System 2 Exploration Mode**:

1. **Generate:** A task generator proposes new tasks, not just tasks from a static dataset.
2. **Validate:** A symbolic verifier confirms the generated task is well-formed and has at least one valid solution.
3. **Solve:** Latent MCTS and the recursive System 2 core attempt the task under a larger plugged-in joule budget.
4. **Store:** If a verified solution is found, store the full winning trajectory:

```math
\tau = (x, y^*, z^{(1)}, z^{(2)}, ..., z^{(K)}, a^{(1)}, ..., a^{(K)}, E_{measured})
```

5. **Compile:** Train System 1, the router, the halter, and the verifier on successful trajectories.
6. **Promote:** Add tasks the current System 1 fails but System 2 solves into the next curriculum stage.

### 10.2 Infinite Synthetic Task Generator

The flywheel must not depend only on downloaded puzzles. It generates new tasks procedurally.

Task families:

```text
Sudoku: generate 4x4 -> 6x6 -> 9x9 boards with controlled clue difficulty.
Maze: generate solvable mazes with controllable path length and traps.
ARC-style grids: generate object/color/transform rules with symbolic validators.
BabyAI-style plans: generate instruction sequences with increasing subgoal depth.
Smart-home state resolution: generate sensor-policy conflicts with known safe actions.
```

Each task generator must expose:

```text
difficulty parameter
solution verifier
curriculum metadata
failure mode tag
```

### 10.3 Self-Play Data Quality Gate

A generated task is admitted only if:

```text
1. The symbolic verifier says the task is valid.
2. The solution is unique or the accepted answer set is well-defined.
3. System 1 currently fails or has low confidence.
4. System 2 solves it with verifier confirmation.
5. The trajectory improves verifier value monotonically often enough to be useful.
```

### 10.4 Distillation Targets

The successful trajectory trains:

```text
System 1: predict final answer directly.
Router: imitate active-token decisions that led to success.
Halter: imitate the earliest step where verifier confidence became sufficient.
Verifier: contrast successful trajectory states against failed rollouts.
Latent MCTS policy prior: bias future search toward successful latent branches.
```

Loss:

```math
\mathcal{L}_{flywheel}
=
\lambda_y CE(\hat{y}_{S1}, y^*)
+
\lambda_z \sum_k ||\hat{z}^{(k)} - z^{(k)}||_2^2
+
\lambda_a \sum_k CE(\hat{a}^{(k)}, a^{(k)})
+
\lambda_v \sum_k (\hat{V}(z^{(k)}) - V_k)^2
```

The agent "dreams" at night: it invents tasks, solves them slowly with latent search, and compiles the discoveries into fast daytime reflexes.

---

## 11. Stability Shield

### 11.1 Main Risk

SPECTRA combines two unstable regimes:

1. Deep latent recursion with shared weights.
2. Ultra-low-bit quantization-aware training.

Possible failures:

```text
predicts all zeros
predicts one token everywhere
latent state becomes fixed too early
latent state explodes
ternary weights saturate
halting collapses
router freezes important tokens
```

### 11.2 Training Must Be Staged

Correct order:

```text
FP16 recursive teacher
→ fake-ternary recursive student
→ progressive ternarization
→ INT8 activation fake quantization
→ lazy routing
→ hardware-aware halting
→ deployment kernels
```

### 11.3 EMA

EMA is mandatory.

```text
ema_decay = 0.999
```

Use:

```text
raw model for training
EMA model for validation/checkpointing
```

### 11.4 Gradient Clipping

Start with:

```text
clip_grad_norm = 0.5 or 1.0
```

Relax later if stable.

### 11.5 Progressive Ternarization Schedule

Do not ternarize all layers at once.

Suggested order:

```text
Stage 0: FP16 everything.
Stage 1: FFN projections ternary.
Stage 2: output projection ternary.
Stage 3: attention/MLP-mixer projections ternary.
Stage 4: recursive core fully ternary.
Stage 5: activations fake-quantized INT8.
Stage 6: deployment W1.58A8.
```

Keep higher precision:

```text
normalization
scales
halting head initially
small verifier/control heads initially
```

### 11.6 Quantization Warmup

Example schedule:

```text
steps 0–5k: FP16
steps 5k–20k: soft ternary
steps 20k+: hard ternary forward
```

Soft ternary can use a temperature or interpolation:

```math
W_{forward} = (1 - \rho)W + \rho(\gamma W_q)
```

where:

```math
\rho = min(1, step / warmup_steps)
```

### 11.7 Collapse Detectors

#### Output Collapse

Track:

```text
prediction_entropy
most_common_token_ratio
unique_predicted_tokens
```

Bad:

```text
most_common_token_ratio > 0.90
```

#### Fixed-Point Collapse

Track:

```math
\Delta_y^{(k)} = ||y^{(k+1)} - y^{(k)}||_2
```

```math
\Delta_z^{(k)} = ||z^{(k+1)} - z^{(k)}||_2
```

Bad:

```text
Δy and Δz go near zero early while accuracy is bad
```

#### Recursive Explosion

Track:

```text
activation_norm_per_step
gradient_norm_per_block
logit_magnitude_per_step
```

Bad:

```text
norms increase every recursion step
NaN/Inf appears
```

#### Ternary Saturation

Track:

```text
percent_W_negative
percent_W_zero
percent_W_positive
```

Bad:

```text
too many zeros → dead model
too many ±1 → noisy model
distribution stops changing too early
```

#### Router Collapse

Track:

```text
active_token_density
false_freeze_rate
mask_entropy
tokens_reactivated
```

Bad:

```text
router freezes wrong tokens early
active density goes to zero while accuracy is bad
```

### 11.8 Dynamical Isometry and Dimensional-Collapse Penalties

RMSNorm bounds the *magnitude* of the latent but does nothing to stop
**dimensional collapse** (all token vectors converging to one direction) or to
control the spectrum of the per-step map. Detection (Section 11.7) is necessary
but not sufficient; the Stability Shield adds explicit regularizers.

**Orthogonal initialization.** All `2D` projection weights are initialized
(semi-)orthogonally — the dynamical-isometry starting point.

**Dimensional-collapse penalty (VICReg variance + covariance).** Over the flattened
latent `Z ∈ R^{N×D}` (centered), with per-feature std `s_j`:

```math
\mathcal{L}_{var} = \frac{1}{D}\sum_{j=1}^{D}\max\!\big(0,\; \gamma - s_j\big), \qquad
\mathcal{L}_{cov} = \frac{1}{D}\sum_{i\neq j}\big[\mathrm{Cov}(Z)\big]_{ij}^2
```

The variance hinge keeps every feature dimension informative (`s_j \ge \gamma`);
the covariance term decorrelates features, preventing a rank-deficient state.

**Jacobian isometry penalty (Hutchinson estimator).** Let `J = \partial z^{(k+1)}/\partial z^{(k)}`
be the per-step Jacobian. We push its singular values toward **1 (isometry)** —
**not** toward `<1`. A strict contraction (`\lVert J\rVert < 1`) would drive every
trajectory to a single fixed point: that *is* dimensional collapse, the opposite of
the goal. For random probes `u \sim \mathcal{N}(0,I)`:

```math
\mathcal{L}_{iso} = \mathbb{E}_{u}\Big[\big(\lVert J^\top u\rVert_2 - \lVert u\rVert_2\big)^2\Big]
```

which is zero iff `J` is an isometry and is computed with a single vector-Jacobian
product (`create_graph=True`, differentiable). Total stability regularizer:

```math
\mathcal{L}_{stab} = \lambda_{collapse}\,(\mathcal{L}_{var} + \mathcal{L}_{cov}) + \lambda_{iso}\,\mathcal{L}_{iso}
```

A **dimensional-collapse detector** (token-variance ratio `\mathrm{Var}_L(z)/\lVert z\rVert^2`
below `\epsilon`) is added to the collapse dashboard alongside the output /
fixed-point / explosion / ternary-saturation detectors of Section 11.7. Reference:
`model/spectral.py`, `train/collapse_watch.py::dimensional_collapse_ratio`.

---

## 12. Corrected Policy-Improvement Loss

Avoid the broken detached-advantage version.

Bad:

```python
advantage = (target_logp - prev_logp).detach()
losses.append(F.relu(-advantage).mean())
```

The gradient is killed.

Use:

```python
margin = 0.01
losses.append(F.relu(prev_logp.detach() + margin - target_logp).mean())
```

Full version:

```python
def policy_improvement_loss(step_logits, y_target, margin=0.01):
    losses = []
    prev_target_logp = None

    for logits in step_logits:
        logp = torch.nn.functional.log_softmax(logits, dim=-1)
        target_logp = logp.gather(-1, y_target.unsqueeze(-1)).squeeze(-1)
        losses.append(-target_logp.mean())

        if prev_target_logp is not None:
            improve_loss = torch.nn.functional.relu(
                prev_target_logp.detach() + margin - target_logp
            ).mean()
            losses.append(improve_loss)

        prev_target_logp = target_logp

    return sum(losses) / len(losses)
```

---

## 13. Recommended Repository Structure

```text
spectra/
├── README.md
├── BLUEPRINT.md
├── pyproject.toml
├── requirements.txt
├── config/
│   ├── base.yaml
│   ├── sudoku.yaml
│   ├── maze.yaml
│   ├── arc.yaml
│   ├── babyai.yaml
│   ├── smarthome.yaml
│   └── edge_devices.yaml
├── data/
│   ├── datasets.py
│   ├── augment.py
│   ├── sudoku.py
│   ├── maze.py
│   ├── arc.py
│   ├── babyai.py
│   ├── smarthome.py
│   └── generate.py
├── model/
│   ├── trm.py
│   ├── bitlinear.py
│   ├── fake_quant.py
│   ├── operators.py
│   ├── spatial_encoder.py
│   ├── lazy_router.py
│   ├── halting.py
│   ├── verifier.py
│   ├── energy.py
│   ├── system1_student.py
│   └── stability.py
├── train/
│   ├── trainer.py
│   ├── losses.py
│   ├── distill.py
│   ├── ema.py
│   ├── schedules.py
│   └── collapse_watch.py
├── eval/
│   ├── benchmarks.py
│   ├── metrics.py
│   ├── best_of_n.py
│   ├── edge_energy.py
│   ├── latency.py
│   ├── memory.py
│   └── reports.py
├── deploy/
│   ├── export_onnx.py
│   ├── export_bitnet.py
│   ├── pack_ternary.py
│   ├── bitnet_cpp_adapter/
│   ├── cpp_sparse_kernel/
├── tests/
│   ├── test_recursion.py
│   ├── test_overfit.py
│   ├── test_fake_bitlinear.py
│   ├── test_int8_activation.py
│   ├── test_lazy_router.py
│   ├── test_halting.py
│   ├── test_verifier.py
│   ├── test_stability.py
│   ├── test_distillation.py
│   ├── test_export.py
│   └── test_energy_measurement.py
└── scripts/
    ├── train_teacher.py
    ├── train_bit_student.py
    ├── train_system1.py
    ├── eval_edge.py
    ├── profile_power.py
    └── make_report.py
```

---

## 14. Environment

### 14.1 Python Training Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# Windows:
# .venv\Scripts\activate

pip install torch torchvision torchaudio
pip install einops numpy tqdm wandb hydra-core omegaconf pytest
pip install onnx onnxruntime
```

Optional:

```bash
pip install gymnasium babyai
```

### 14.2 Deployment Tooling

Later phases:

```text
bitnet.cpp
CMake
clang/gcc
Linux cgroups tooling
RAPL / perf / psutil profiling helpers
```

---

## 15. First Coding Milestone: FP16 TRM MVP

### 15.1 Minimal Operator

Start with a simple transformer-style block.

```python
class SwapBlock(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.RMSNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, h):
        q = self.norm1(h)
        a, _ = self.attn(q, q, q)
        h = h + a
        h = h + self.ff(self.norm2(h))
        return h
```

### 15.2 TRM Core

```python
class TRM(nn.Module):
    def __init__(
        self,
        dim,
        num_tokens,
        seq_len,
        n_layers=2,
        n=6,
        T=3,
        N_sup=16,
    ):
        super().__init__()
        self.n = n
        self.T = T
        self.N_sup = N_sup

        self.token_embed = nn.Embedding(num_tokens, dim)
        self.row_embed = nn.Embedding(32, dim)
        self.col_embed = nn.Embedding(32, dim)

        self.blocks = nn.ModuleList([SwapBlock(dim) for _ in range(n_layers)])
        self.out_head = nn.Linear(dim, num_tokens)
        self.halt_head = nn.Linear(dim, 1)

        self.norm_y = nn.RMSNorm(dim)
        self.norm_z = nn.RMSNorm(dim)

        self.alpha_y = nn.Parameter(torch.tensor(0.1))
        self.alpha_z = nn.Parameter(torch.tensor(0.1))

    def encode_positions(self, x, height, width):
        B, L = x.shape
        device = x.device
        rows = torch.arange(L, device=device) // width
        cols = torch.arange(L, device=device) % width
        return self.row_embed(rows)[None, :, :] + self.col_embed(cols)[None, :, :]

    def f(self, h):
        for block in self.blocks:
            h = block(h)
        return h

    def recursive_cycle(self, x_emb, y, z):
        for _ in range(self.n):
            update_z = self.f(x_emb + y + z)
            z = self.norm_z(z + self.alpha_z * update_z)

        update_y = self.f(y + z)
        y = self.norm_y(y + self.alpha_y * update_y)
        return y, z

    def forward(self, x, height=9, width=9, y_target=None):
        x_emb = self.token_embed(x) + self.encode_positions(x, height, width)
        y = torch.zeros_like(x_emb)
        z = torch.zeros_like(x_emb)

        step_outputs = []

        for k in range(self.N_sup):
            for _ in range(self.T):
                y, z = self.recursive_cycle(x_emb, y, z)

            logits = self.out_head(y)
            halt_logit = self.halt_head(y.mean(dim=1)).squeeze(-1)

            step_outputs.append({
                "logits": logits,
                "halt_logit": halt_logit,
                "y": y,
                "z": z,
            })

            y = y.detach()
            z = z.detach()

        return step_outputs[-1]["logits"], step_outputs
```

---

## 16. Tests: Mandatory Gates

### Gate 1 — Shape and Backward

```python
def test_shapes_and_backward():
    model = TRM(dim=64, num_tokens=10, seq_len=81, N_sup=4)
    x = torch.randint(0, 10, (2, 81))
    y = torch.randint(0, 10, (2, 81))

    logits, steps = model(x, height=9, width=9, y_target=y)
    assert logits.shape == (2, 81, 10)

    loss = deep_supervision_loss(steps, y)
    loss.backward()

    assert model.blocks[0].ff[0].weight.grad is not None
```

### Gate 2 — Effective Depth

```python
def test_effective_depth():
    m = TRM(dim=64, num_tokens=10, seq_len=81, n=6, T=3, n_layers=2)
    effective_depth = m.T * (m.n + 1) * len(m.blocks)
    assert effective_depth == 42
```

### Gate 3 — Single-Batch Overfit

```python
def test_overfit_single_batch():
    model = TRM(dim=128, num_tokens=10, seq_len=81, N_sup=8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randint(0, 10, (4, 81))
    y = torch.randint(0, 10, (4, 81))

    for step in range(300):
        logits, steps_out = model(x, height=9, width=9, y_target=y)
        loss = deep_supervision_loss(steps_out, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    acc = (logits.argmax(-1) == y).float().mean().item()
    assert acc > 0.95, f"Cannot overfit one batch: acc={acc}"
```

### Gate 4 — Step Accuracy Improves

```python
def test_accuracy_improves_across_steps(trained_model, val_batch):
    x, y = val_batch
    _, steps = trained_model(x, height=9, width=9, y_target=y)

    accs = []
    for s in steps:
        pred = s["logits"].argmax(-1)
        accs.append((pred == y).float().mean().item())

    assert accs[-1] >= accs[0] - 0.02
```

### Gate 5 — No Collapse

```python
def test_no_output_collapse(logits):
    pred = logits.argmax(-1).flatten()
    mode_ratio = torch.bincount(pred).max().float() / pred.numel()
    assert mode_ratio < 0.90
```

---

## 17. Deep Supervision Loss Implementation

```python
def deep_supervision_loss(
    steps,
    y_target,
    lambda_h=0.5,
    lambda_improve=0.1,
    margin=0.01,
):
    total = 0.0
    prev_target_logp = None

    for s in steps:
        logits = s["logits"]
        halt_logit = s["halt_logit"]

        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y_target.reshape(-1),
        )

        correct = (logits.argmax(-1) == y_target).all(dim=1).float()
        halt_bce = F.binary_cross_entropy_with_logits(halt_logit, correct)

        logp = F.log_softmax(logits, dim=-1)
        target_logp = logp.gather(-1, y_target.unsqueeze(-1)).squeeze(-1)

        improve = 0.0
        if prev_target_logp is not None:
            improve = F.relu(prev_target_logp.detach() + margin - target_logp).mean()
        prev_target_logp = target_logp

        total = total + ce + lambda_h * halt_bce + lambda_improve * improve

    return total / len(steps)
```

---

## 18. FakeBitLinear Implementation

```python
class FakeBitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.eps = eps
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def ternarize(self, w):
        # Per-output-channel scale.
        scale = w.abs().mean(dim=1, keepdim=True).clamp_min(self.eps)
        w_scaled = w / scale
        w_q = torch.clamp(torch.round(w_scaled), -1, 1)
        # STE.
        w_ste = w + (w_q * scale - w).detach()
        return w_ste

    def forward(self, x):
        w_q = self.ternarize(self.weight)
        return F.linear(x, w_q, self.bias)
```

### FakeBitLinear Tests

```python
def test_fake_bitlinear_backward():
    layer = FakeBitLinear(32, 64)
    x = torch.randn(4, 10, 32)
    y = layer(x).sum()
    y.backward()
    assert layer.weight.grad is not None

def test_fake_bitlinear_ternary_distribution():
    layer = FakeBitLinear(32, 64)
    with torch.no_grad():
        scale = layer.weight.abs().mean(dim=1, keepdim=True).clamp_min(1e-5)
        w_q = torch.clamp(torch.round(layer.weight / scale), -1, 1)
    assert set(w_q.unique().tolist()).issubset({-1.0, 0.0, 1.0})
```

---

## 19. INT8 Activation Fake Quantization

```python
class FakeActQuant(nn.Module):
    def __init__(self, bits=8, eps=1e-6):
        super().__init__()
        self.bits = bits
        self.qmin = -(2 ** (bits - 1))
        self.qmax = (2 ** (bits - 1)) - 1
        self.eps = eps

    def forward(self, x):
        scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(self.eps) / self.qmax
        x_q = torch.clamp(torch.round(x / scale), self.qmin, self.qmax)
        x_deq = x_q * scale
        return x + (x_deq - x).detach()
```

Use it inside the recursive loop:

```python
z = self.act_quant(z)
y = self.act_quant(y)
```

---

## 20. RL Router Implementation

The router is not implemented as a fixed confidence threshold in the final system. It is a learned policy that outputs a binary compute/freeze decision for each token under the current latent state.

### 20.1 Policy Module

```python
class RLTokenRouter(nn.Module):
    def __init__(self, dim, device_dim=8):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(dim + device_dim, dim),
            nn.GELU(),
            nn.Linear(dim, 2)  # actions: 0=freeze, 1=compute
        )

    def forward(self, z, device_state, sample=True):
        # z: [B, L, D]
        # device_state: [B, device_dim]
        B, L, D = z.shape
        d = device_state[:, None, :].expand(B, L, -1)
        logits = self.policy(torch.cat([z, d], dim=-1))
        dist = torch.distributions.Categorical(logits=logits)
        if sample:
            actions = dist.sample()
        else:
            actions = logits.argmax(dim=-1)
        logprob = dist.log_prob(actions)
        active_mask = actions.bool()
        return active_mask, logprob, logits
```

### 20.2 REINFORCE Update

```python
def router_reinforce_loss(logprobs, reward, baseline):
    # logprobs: list of [B, L] sampled routing log-probs across recursion steps
    # reward: [B]
    advantage = (reward - baseline).detach()
    total_logprob = sum(lp.sum(dim=1) for lp in logprobs)  # [B]
    return -(advantage * total_logprob).mean()
```

The reward is measured after the final answer is verified:

```math
R = \mathbf{1}[	ext{Correct Answer}] - \lambda_{energy}\,J - \lambda_{latency}\,T_{ms}
```

### 20.3 Dense-Masked Prototype Warning

A dense masked version can validate correctness, but it must not be used for final speed claims. Final speed claims require the fused B=1 GEMV sparse ternary kernel in Section 26.


---

## 21.
---

## 21. Hardware Telemetry Input

### Device State Structure

```python
@dataclass
class DeviceState:
    battery_level: float       # 0 to 1
    thermal_level: float       # 0 to 1
    latency_budget_ms: float
    power_mode: float          # encoded 0 to 1
    available_ram_mb: float
    device_class: int
```

Convert to tensor:

```python
def device_state_to_tensor(d: DeviceState, device):
    return torch.tensor([
        d.battery_level,
        d.thermal_level,
        min(d.latency_budget_ms / 1000.0, 1.0),
        d.power_mode,
        min(d.available_ram_mb / 8192.0, 1.0),
        d.device_class / 10.0,
    ], device=device).float()
```

### Linux / Legacy-x86 Telemetry Sources

Linux battery and AC status:

```text
/sys/class/power_supply/
```

Thermal zones:

```text
/sys/class/thermal/
```

CPU and memory constraints:

```text
systemd-run --scope -p CPUQuota=100% -p MemoryMax=512M ...
cgcreate / cgset / cgexec
```

Physical CPU package energy:

```text
/sys/class/powercap/intel-rapl/
perf stat -e power/energy-pkg/
```

RAPL is used as hardware-level silicon telemetry for the legacy x86 execution plan. FLOPs are never used as a substitute for measured joules.

---

## 22. Physical Energy Profiling

### 22.1 Do Not Estimate Joules from FLOPs

Wrong:

```text
ternary op is cheaper → therefore energy is 10× lower
```

Correct:

```text
measure actual power draw during inference loops
```

### 22.2 Extreme Legacy Hardware Simulation

We deliberately reject modern H100 clusters for evaluation. To prove true democratization of reasoning, evaluation is performed on **legacy x86 hardware (decade-old CPU)**.

To simulate varied edge conditions without needing multiple physical devices, we utilize **Linux `cgroups`**:

- **Catastrophic Edge:** Throttled to 1 CPU Core, 512MB RAM limit.
- **Standard Edge:** 2 CPU Cores, 2GB RAM limit.

**Power Measurement:** Physical silicon energy draw is profiled not via FLOP estimation, but directly via hardware-level **Intel/AMD RAPL (Running Average Power Limit)** telemetry, measuring actual joules consumed during the MCTS loops.

### 22.3 Energy Formula

Sample power over time:

```math
E = \int_{t_0}^{t_1} P(t)\,dt
```

Discrete approximation:

```math
E \approx \sum_i P_i \Delta t_i
```

Where:

```math
P_i = V_i I_i
```

Report:

```math
Joules\_per\_problem = \frac{E_{run} - E_{idle}}{N_{problems}}
```

Always subtract idle baseline:

```math
E_{net} = E_{inference} - E_{idle}
```

### 22.4 Profiling Protocol

For every device/model/task:

```text
1. Warm up device for fixed number of runs.
2. Lock power mode where possible.
3. Record idle power for 60 seconds.
4. Run N inference problems.
5. Record voltage/current/power throughout.
6. Repeat at least 5 times.
7. Report mean, std, median, p95.
8. Log ambient/device temperature.
9. Log battery/plugged state.
10. Log OS/process background assumptions.
```

---

## 23. Edge Metrics

### 23.1 Core Metrics

```text
accuracy
pass@1
pass@K
latency mean/p50/p95
joules per problem
peak RAM
model size
active token density
average recursion steps
halt step distribution
```

### 23.2 Efficiency Metrics

```math
AccuracyPerJoule = \frac{Accuracy}{JoulesPerProblem}
```

```math
AccuracyPerMB = \frac{Accuracy}{PeakRAM_{MB}}
```

```math
AccuracyPerMs = \frac{Accuracy}{Latency_{ms}}
```

Optional combined metric:

```math
EdgeReasoningScore =
\frac{Accuracy}
{
(JoulesPerProblem + \epsilon)
\cdot
(LatencyMs + \epsilon)
\cdot
(PeakRAMMB + \epsilon)
}
```

Use combined metric carefully. Always report raw metrics too.

---

## 24. Benchmarks

### 24.1 Sudoku

Purpose:

```text
controlled symbolic reasoning
exact validator
easy to debug
```

Metrics:

```text
cell accuracy
board accuracy
valid board rate
clue preservation
steps to valid solution
```

### 24.2 Maze

Purpose:

```text
path planning
exact path verifier
small-to-large extrapolation
```

Metrics:

```text
path success
path validity
path length ratio to optimal
wall collision rate
```

### 24.3 ARC / ARC-Style

Purpose:

```text
abstract grid reasoning
object/color transformations
generalization pressure
```

Metrics:

```text
exact grid accuracy
pixel accuracy
object consistency
output shape accuracy
pass@K
```

### 24.4 BabyAI-Style Local Planning

Purpose:

```text
grounded local sequential reasoning
edge-relevant robot/planner task
```

Tasks:

```text
Predict: predict next state after action
Plan: produce action sequence
Decompose: split instruction into subgoals
```

Metrics:

```text
plan success
valid action rate
steps to goal
energy per successful plan
```

### 24.5 Smart-Home State Resolution

Purpose:

```text
practical privacy-preserving edge reasoning
battery-aware local decision-making
```

Example input:

```json
{
  "door_sensor": "open",
  "lock": "locked",
  "motion": "none",
  "user_location": "away",
  "thermostat_mode": "home",
  "battery_level": 0.18,
  "time": "23:40"
}
```

Example output:

```json
{
  "action": "send_alert",
  "priority": "medium",
  "reason": "door open while user away but lock reports locked",
  "confidence": 0.74
}
```

Metrics:

```text
action accuracy
safety-critical false negative rate
latency
joules
fallback rate
```

---

## 25. Deployment Roadmap

### Phase D0 — PyTorch Only

Goal:

```text
algorithm works
```

Tests:

```text
overfit
recursion improves
no collapse
fake ternary works
INT8 fake activation works
```

### Phase D1 — Dense bitnet.cpp Baseline

Goal:

```text
real CPU W1.58A8 dense inference baseline
```

Measure:

```text
latency
RAM
joules
accuracy gap vs PyTorch
```

### Phase D2 — Lazy Routing Simulation

Goal:

```text
prove active tokens shrink enough
```

Do not claim real speedup yet.

### Phase D3 — Custom Sparse Ternary Kernel

Goal:

```text
real fused sparse execution
```

Kernel must fuse:

```text
active token gather
packed ternary matmul
scale application
activation quantization
scatter update
```

### Phase D4 — Optional Future Physical Edge Devices

Goal:

```text
real deployment on edge devices
```

---

## 26. Custom Kernel Design Notes

### 26.1 Data Representation

Weights:

```text
2-bit packed ternary code:
00 → 0
01 → +1
10 → -1
11 → reserved
```

Activations:

```text
int8
```

Accumulation:

```text
int16/int32
```

Scale:

```text
per-output-channel or per-block FP16/FP32
```

### 26.2 Fused Sparse Kernel Inputs

```text
X_int8: [B, L, D]
active_idx: dynamic token indices
W_packed: packed ternary weights
W_scale: scales
Y_int8/output buffer
```

Deployment kernel requirement:

```text
The kernel must be aggressively optimized for B=1 inference.
Edge deployments process single instances, so the operation is GEMV,
not training-style GEMM. Memory bandwidth is the primary bottleneck.
```

### 26.3 Fused Operation ($B=1$ Memory Bandwidth Optimization)

Edge inference is strictly Batch Size 1 ($B=1$). Standard PyTorch kernels fail because they optimize for GEMM (matrix-matrix multiplication). At $B=1$, the bottleneck is memory-bound GEMV (matrix-vector streaming), especially on legacy DDR3/DDR4 systems.

The deployment kernel must therefore be aggressively optimized for $B=1$ inference, L1/L2/L3 cache locality, and x86 AVX2 vector lanes. The kernel fuses active-token gather, packed W1.58 ternary GEMV, scaling, requantization, and scatter into one pass.

```cpp
// Strictly optimized for B=1 GEMV to defeat legacy RAM bottlenecks.
// AVX2 path: use __m256i, not ARM NEON vector types.
// This is schematic pseudocode; real implementation packs ternary weights
// into 2-bit lanes and uses custom decode/dot kernels.

for (int t = 0; t < num_active_tokens; ++t) {
    const int token_idx = active_idx_list[t];

    for (int d = 0; d < hidden_dim; d += 32) {
        // 1. Vectorized B=1 gather/load from the active token state.
        __m256i x_vec = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(&X[token_idx * hidden_dim + d])
        );

        // 2. Load packed W1.58 ternary weights for this output block.
        __m256i w_pack = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(&W_packed[d])
        );

        // 3. Ternary GEMV: no floating-point MACs.
        //    Custom routine decodes {-1,0,+1}, applies sign/add-subtract logic,
        //    and accumulates into int16/int32 lanes.
        __m256i acc = ternary_gemv_int8_avx2(x_vec, w_pack);

        // 4. Fused requantize and scatter back to the same active token.
        __m256i y_q = requantize_int32_to_int8_avx2(acc, W_scale, act_scale);
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(&Y[token_idx * hidden_dim + d]),
            y_q
        );
    }
}
```

By fusing gather, GEMV, scale, requantization, and scatter into a single B=1 streaming pass, SPECTRA avoids writing intermediate states to slow main memory and maximizes cache hits on legacy x86 hardware.

### 26.4 Why Fusion Matters

Bad:

```text
PyTorch gather
→ separate matmul
→ separate quantize
→ separate scatter
```

This causes memory bandwidth overhead.

Good:

```text
single kernel keeps active token data hot in cache
```

### 26.5 Roofline Proof: Why a Single B=1 GEMV Is Memory-Bound, and Recursion Fixes It

Be honest about the physics. For an `O×H` weight matrix at batch size 1, each
weight is read once from DRAM and used in one multiply-accumulate. The arithmetic
intensity is therefore fixed by the weight precision alone:

```math
\text{AI} = \frac{\text{ops}}{\text{DRAM bytes}} = \frac{O\,H}{O\,H \cdot (\text{bits}/8)} = \frac{8}{\text{bits}}
```

```text
FP32 weights : AI = 8/32 = 0.25 ops/byte   -> deeply memory-bound
W1.58 (2-bit): AI = 8/2  = 4.0  ops/byte   -> 16x better, but still low
```

**Cache-blocking a single B=1 GEMV cannot help: there is no reuse dimension to
block.** Quantization helps only by cutting the streamed bytes 16×, which is a 16×
speedup *at the memory-bound limit* (`t = \text{bytes}/\text{BW}`), not a move across
the roofline. On a bandwidth-starved legacy core (≈40 GOPS / ≈10 GB/s, ridge
`= 4`), the W1.58 GEMV sits *exactly on* the ridge — balanced, not compute-bound.

The only physically valid way to become compute-bound is to introduce reuse, and
**recursion supplies it.** The same ≈1.4 MB ternary core is re-applied
`K = T\,n\,N_{sup}` times. If the weights stay resident in L2/L3 across the
recursion, DRAM pays for them **once** and the cost amortizes over `K`
applications:

```math
\text{AI}_{recursion} = \frac{K \cdot O\,H}{O\,H \cdot (\text{bits}/8)} = K\cdot\frac{8}{\text{bits}} = 4K
```

So the operating point moves from `AI = 4` (single GEMV, at the ridge) to
`AI = 4K` (recursion-resident), firmly compute-bound for any realistic `K`. This is
realized by the **weight-stationary recursion kernel** `spectra_weight_stationary_gemv`
(decode each ternary row once, dot it against all `K` recursion-step activations
before eviction) and quantified by `eval/roofline.py`. The headline claim is
precise: *the W1.58A8 core is 16× cheaper in DRAM bytes than FP32, and recursion
residency multiplies the effective arithmetic intensity by the recursion depth.*

---

## 27. Baselines and Ablations

### 27.0 Compute-Optimal Frontier Curves

Mandatory deliverable:

```text
plot accuracy vs physically measured joules for:
- parameter count
- recursion depth
- active-token density
- Latent MCTS rollout count
- cgroup memory/CPU limits
```

The main research question:

```text
For a fixed joule budget on legacy x86, what is the optimal mix of model size and test-time thinking?
```

### 27.1 Model Baselines

```text
FP16 TRM
INT8 post-training quantized TRM
FakeBit TRM
Dense W1.58A8 BitTRM
SPECTRA without lazy routing
SPECTRA without hardware halting
SPECTRA without verifier
SPECTRA without System 1
full SPECTRA
```

### 27.2 Operator Ablations

```text
ternary attention block
ternary MLP-Mixer block
ternary gated MLP block
small dense MLP recurrent block
```

Mamba/SSM:

```text
future work only
```

### 27.3 Precision Ablations

```text
FP16 weights / FP16 activations
INT8 weights / INT8 activations
W1.58 / FP16 activations
W1.58 / INT8 activations
W1.58 / INT8 activations + INT32 accumulation
```

Critical comparison:

```text
W1.58A16 vs W1.58A8
```

This proves activation quantization matters.

### 27.4 Routing Ablations

```text
dense recursion
supervised learned router
REINFORCE learned router
PPO-style learned router
oracle compute-policy upper bound
lazy routing with no speed claim
lazy routing with fused kernel
```

### 27.5 Halting Ablations

```text
fixed steps
confidence halting
verifier halting
hardware-aware halting
hardware-aware halting + lazy routing
```

---

## 28. Phase-by-Phase Implementation Plan

### Phase 0 — Repo Setup

Deliverables:

```text
repo structure
pytest setup
config files
logging
seed control
```

Gate:

```text
pytest runs
dummy model trains one step
```

### Phase 1 — FP16 TRM MVP

Deliverables:

```text
spatial encoder
recursive core
deep supervision loss
halting head
overfit test
```

Gate:

```text
single-batch overfit > 95%
```

### Phase 2 — Real Task MVP

Deliverables:

```text
Sudoku dataset
Sudoku augmentation
Sudoku validator
Maze dataset
Maze validator
```

Gate:

```text
augmentation correctness passes
MVP reaches non-trivial validation accuracy
```

### Phase 3 — Recursion Diagnostics

Deliverables:

```text
per-step accuracy
effective depth
grad norms
state delta norms
collapse detectors
```

Gate:

```text
later recursion steps improve or preserve accuracy
no collapse
```

### Phase 4 — FakeBitLinear

Deliverables:

```text
FakeBitLinear
ternary distribution logging
fake ternary TRM
progressive ternarization
```

Gate:

```text
fake ternary model overfits one batch
accuracy gap acceptable
no ternary saturation collapse
```

### Phase 5 — INT8 Activation Fake Quant

Deliverables:

```text
FakeActQuant
W1.58A8 recursive loop
activation scale logging
```

Gate:

```text
W1.58A8 model overfits one batch
activation quantization does not destroy recursion
```

### Phase 6 — Stability Shield

Deliverables:

```text
EMA
gradient clipping
residual scaling
teacher-student state distillation
collapse dashboard
```

Gate:

```text
stable multi-epoch training
no fixed-point collapse
```

### Phase 7 — Lazy Router

Deliverables:

```text
confidence/entropy router
dense-masked prototype
active-token metrics
```

Gate:

```text
active token density falls below 30–40% on useful tasks
accuracy loss acceptable
```

### Phase 8 — Verifier

Deliverables:

```text
Sudoku verifier
Maze verifier
ARC soft verifier
energy head
hard negative mining
```

Gate:

```text
verifier ranks correct candidate above wrong candidate >80%
best-of-N beats single candidate
```

### Phase 9 — Hardware-Aware Halting

Deliverables:

```text
device_state tensor
compute penalty
budget-conditioned halting
telemetry wrapper
```

Gate:

```text
model reduces steps under low-budget states
accuracy/compute tradeoff behaves smoothly
```

### Phase 10 — System 1 Student

Deliverables:

```text
feed-forward student
teacher logits distillation
state distillation
dual-mode inference
```

Gate:

```text
System 1 solves easy examples cheaply
System 2 activates mainly on uncertain examples
```

### Phase 11 — bitnet.cpp Dense Deployment

Deliverables:

```text
weight export
ternary packing
bitnet.cpp adapter
dense CPU benchmark
```

Gate:

```text
outputs match PyTorch within acceptable tolerance
latency/RAM measured
```

### Phase 12 — Fused Sparse Kernel

Deliverables:

```text
C++/SIMD sparse ternary matmul
active-token fused update
benchmark vs dense
```

Gate:

```text
real speed/energy gain on physical device
```

### Phase 13 — Edge-Native Evaluation

Goal:

```text
prove reasoning-per-joule under catastrophic legacy hardware constraints
```

Primary evaluation environment:

```text
legacy x86 laptop / decade-old CPU
Linux cgroups:
- 1 core + 512 MB RAM
- 2 cores + 2 GB RAM
Intel/AMD RAPL energy profiling
B=1 inference only
```

Measure:

```text
accuracy
latency
peak RAM
RAPL joules
active-token count
recursion steps
Latent MCTS rollouts
accuracy per joule
compute-optimal frontier curves
```

Gate:

```text
SPECTRA must show real measured energy/latency savings,
not only FLOP-estimated savings.
```

---

## 29. Logging Checklist

Log these from day one.

### Training Logs

```text
loss_task
loss_halt
loss_distill
loss_compute
accuracy
per_step_accuracy
grad_norm_total
grad_norm_per_block
activation_norm_y
activation_norm_z
state_delta_y
state_delta_z
halt_step
halt_entropy
```

### Quantization Logs

```text
percent_W_negative
percent_W_zero
percent_W_positive
weight_scale_mean
weight_scale_std
activation_scale_mean
activation_clip_rate
```

### Router Logs

```text
active_token_density
active_tokens_per_step
false_freeze_rate if labels available
reactivated_token_count
mask_entropy
```

### Edge Logs

```text
latency_ms_mean
latency_ms_p50
latency_ms_p95
joules_per_problem
idle_power
peak_ram_mb
model_size_mb
temperature
battery_state
power_mode
```

---

## 30. Failure Recovery Guide

### Problem: Cannot overfit one batch

Check:

```text
detach placement
shared weights actually used
gradients reach embed/out_head
learning rate
loss shape
target encoding
```

### Problem: Accuracy flat across recursion steps

Check:

```text
z not being zeroed each step
detach only between supervision steps, not inside inner loop
residual scaling too small
halting head not dominating
```

### Problem: Ternary model predicts all zeros

Check:

```text
ternarization too early
learning rate too low/high
too many weights quantized to zero
activation quant clipping too strong
distillation missing
```

### Problem: Activations explode

Check:

```text
RMSNorm placement
residual alpha too large
gradient clipping
activation quant scale
```

### Problem: Lazy router freezes wrong tokens

Check:

```text
RL router policy freezes too aggressively
router introduced too early
no minimum active token count
verifier not supervising router
```

### Problem: PyTorch lazy routing is slower

Expected.

Do not optimize in Python. Use PyTorch only for algorithm validation.

### Problem: Energy savings do not match FLOP savings

Expected possibility.

Measure physical energy. Check:

```text
memory bandwidth
thermal throttling
OS background tasks
cache misses
power mode
```

---

## 31. Minimal First Commit

Create only these files first:

```text
model/operators.py
model/trm.py
train/losses.py
tests/test_recursion.py
tests/test_overfit.py
config/sudoku.yaml
```

First command:

```bash
pytest tests/test_recursion.py tests/test_overfit.py
```

If green:

```text
you have a working recursive reasoner
```

Then move to real Sudoku/Maze.

---

## 32. What NOT to Build First

Do not start with:

```text
custom C++ sparse kernel
Mamba/SSM
full ARC
hardware-aware halting
System 1 self-play compiler
bitnet.cpp export
```

Start with:

```text
FP16 recursion works
FakeBit recursion works
W1.58A8 recursion works
```

Everything else comes after.

---

## 33. References and Tooling Notes

Use these as starting references while coding/researching:

- TRM-style original blueprint in this repo/conversation: recursive core, deep supervision, overfit gate, augmentation gate, halting, verifier, and evaluation discipline.
- BitNet / BitNet b1.58:
  - BitLinear as drop-in replacement for `nn.Linear`.
  - Ternary weights `{-1, 0, +1}`.
  - W1.58A8: 1.58-bit weights + 8-bit activations.
- `bitnet.cpp`:
  - Use as dense CPU/edge baseline before custom kernels.
- ONNX Runtime quantization:
  - Use for normal INT8 baselines.
- BabyAI:
  - Grounded local planning benchmark.
- Intel/AMD RAPL:
  - Primary legacy-x86 hardware energy telemetry.

---

## 34. Final Research Paper Story

The final paper should tell this story:

1. Recursive reasoning is promising but usually expensive at test time.
2. Edge devices cannot afford massive voting or dense repeated inference.
3. Weight sharing makes recursion storage-efficient, but not automatically energy-efficient.
4. SPECTRA makes recursion edge-native with:
   - W1.58A8 ternary core.
   - INT8 latent recursion.
   - Lazy token routing.
   - RL-driven hardware-aware halting.
   - Verifier-guided inference.
   - System 2-to-System 1 self-play flywheel.
5. The system is tested not only on puzzles but also on edge-native reasoning tasks.
6. Energy is measured physically, not estimated from FLOPs.
7. The main result is reasoning accuracy per real joule.

---

## 35. Final Success Criteria

The project is successful if it shows:

```text
1. FP16 TRM reproduces recursive reasoning behavior.
2. W1.58A8 TRM retains most accuracy.
3. Lazy routing reduces active tokens significantly.
4. RL-driven hardware-aware halting smoothly trades accuracy for compute.
5. System 1 handles easy examples cheaply.
6. Verifier improves reliability.
7. Dense bitnet.cpp beats FP16 CPU baseline in memory/latency.
8. Sparse fused kernel beats dense ternary baseline when active density is low.
9. Physical energy measurements show improved accuracy per joule.
10. Edge-native tasks show practical value beyond toy puzzles.
```

If all 10 pass, SPECTRA is a serious lab-grade project.

---

## 36. One-Line Coding Reminder

**Do not optimize until the overfit test passes. Do not deploy until fake-ternary passes. Do not write sparse kernels until active-token density proves they are worth it.**
