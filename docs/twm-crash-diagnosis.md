# TWM Harness NOOP-Collapse Diagnosis

Date: 2026-10-27

Status: **TWM FIXED (one repair round)**.  Both the 100k-step acceptance run
and the 1M-step confirmation run pass all three criteria.  The repaired TWM
does not, however, show a reliable mechanism advantage over pure PPO; see
Step 3 below.

Evidence labels: **command-verified** (probe JSON, harness JSON, or log
output), **source-inspected** (installed package code), and **inferred**
(consistent explanation not isolated by an ablation).  All measurements are
single-seed, and no third-party source was modified.

## Executive result

The collapse is an **inverted policy-gradient sign in TWM 0.0.59**, not a
reward-range problem, a missing-warmup problem, a dead gradient graph, a
world-model leak, or a NaN/Inf failure.

`Agent.policy_loss` in the installed 0.0.59 package returns:

```text
min(surr1, surr2) - actor_beta_s * entropy
```

`Agent.learn` minimizes that value.  With positive advantages, the gradient of
the minimizer with respect to the selected action logit is **positive**
(+0.02547 in the synthetic sign probe), so the update reduces the probability
of actions that were taken in good-advantage states.  The correct objective is
`-min(surr1, surr2) - actor_beta_s * entropy`.  The source entropy coefficient
is 0.01 and is active, but its gradient is roughly an order of magnitude too
small to preserve exploration against the inverted surrogate.

The harness now overrides only `CompatAgent.policy_loss` with the corrected
sign and adds policy-entropy/NOOP instrumentation to `eval_policy`.  The
installed TWM package remains untouched.

## Step 1: 1a-1f checks

The probes are implemented in `scripts/twm_crash_probe.py` and write JSON to
`~/craftax-workspace/logs/twm_crash_probe_*.json`.  The horizontal checkpoint
probe loads the original 1M diagnostic checkpoint at
`~/craftax-workspace/data/diagnostic_checkpoints/step_01000000.pt`.

### 1a. Dyna warmup

**Source-inspected:** TWM 0.0.59 `Agent.__init__` has no warmup argument, and
`Agent.learn` has no warmup branch.  The harness also has no warmup branch; it
calls `agent.learn` on iteration 1.

**Command-verified:** the probe ran five 640-step policy iterations and logged
the actor parameter delta and pre/post entropy on each iteration.

| Iteration | Actor delta (L1) | Entropy before | Entropy after | Max prob after |
|---:|---:|---:|---:|---:|
| 1 | 397.25 | 2.82394 | 2.81529 | 0.07994 |
| 2 | 484.03 | 2.81529 | 2.76788 | 0.12222 |
| 3 | 566.50 | 2.76788 | 2.30789 | 0.33694 |
| 4 | 686.53 | 2.30789 | 0.28205 | 0.95264 |
| 5 | 641.50 | 0.28205 | 0.00252 | 0.99978 |

**Conclusion:** warmup is not the root cause.  The policy is updated from the
first iteration, and the original loss drives entropy from 2.82 to 0.0025 in
five 640-step updates.  The same probe with the corrected sign keeps entropy at
2.66649 after five iterations.

### 1b. Reward range and clipping

**Command-verified:** 6,400 random-action environment steps produced:

| Statistic | Reward |
|---|---:|
| min | -0.70 |
| max | +1.10 |
| mean | +0.01495 |
| std | 0.13767 |
| 1st percentile | -0.20 |
| 5th-95th percentile | 0.00, 0.00 |
| 99th percentile | +1.00 |
| negative fraction | 1.78% |
| below WM minimum (-1.0) | 0 |
| above WM maximum (+1.0) | 1 |

The world model is configured with `reward_min_value=-1.0`,
`reward_max_value=1.0`, and `reward_num_bins=65`.  `Agent.learn` does not clip
or normalize rewards.  The earlier claim that -0.2 lies outside the WM support
was incorrect for this configuration: -0.2 is inside [-1, 1].  One observed
+1.10 event emits an HL-Gauss out-of-range warning, but the reward head is the
only consumer of that value and it cannot explain a policy that collapses to
NOOP in five updates.  **Not the root cause; one out-of-support event remains
pending for a future WM-only improvement.**

### 1c. Entropy coefficient and policy-gradient sign

**Source-inspected:** `actor_beta_s` is 0.01 by default, and the installed
objective is `min(surr1, surr2) - actor_beta_s * entropy`.

**Command-verified:** on real rollout states with a controlled advantage
spread, the mean policy entropy was 2.82394 and the mean entropy-bonus
magnitude was 0.02824.  The gradient of the source loss with respect to the
selected logit was:

| Loss | Mean selected-logit gradient, positive advantage | Mean selected-logit gradient, negative advantage |
|---|---:|---:|
| Installed `min(surr1, surr2) - beta*entropy` | **+0.025475** | -0.025400 |
| Corrected `-min(surr1, surr2) - beta*entropy` | **-0.025475** | +0.025401 |

The installed sign is therefore opposite to the policy-gradient direction.
The entropy bonus is active but much smaller than the surrogate term.

**Conclusion: root cause.**  The policy is trained to suppress the actions it
takes in positive-advantage states, which produces a fast deterministic
collapse; this matches the measured entropy trajectory in 1a and the final
NOOP policy in 1d.

### 1d. Action sampling and logits

**Command-verified:** 100 sampled actions from the initial policy and 100 from
the original 1M checkpoint:

| Policy | Action counts (index 0 = NOOP) | NOOP fraction | Mean policy entropy | Mean max probability | Finite logits |
|---|---|---:|---:|---:|---|
| Initial | `[5,5,0,5,7,6,5,7,6,11,5,5,9,6,6,6,6]` | 5% | 2.82375 | 0.07262 | true |
| Original 1M checkpoint | `[100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]` | 100% | 3.53e-08 | 1.00000 | true |

The initial policy samples broadly.  The checkpoint policy is saturated but
finite; there is no logit corruption, NaN, or Inf.

### 1e. Policy-gradient flow

**Command-verified:** after one real actor loss backward pass and before the
world-model backward pass:

| Parameter group | Grad tensors | All zero | Finite | Total grad norm |
|---|---:|---|---:|---:|
| Actor | 16 | false | true | 5.11011 |
| Impala | 14 | false | true | 2.90995 |

After clearing all policy gradients and running a world-model loss backward,
`wm_loss_leaked_to_actor` was **false**.  **Conclusion:** the graph is
connected and the WM does not leak into the policy; gradient deadness is not
the root cause.

### 1f. Optimizer and parameter conflicts

**Command-verified:** the agent exposes 14 Impala parameter tensors.  The
actor optimizer contains 30 parameters, the critic optimizer contains 33, and
both contain **all 14** Impala tensors.  The WM optimizer is disjoint from all
policy parameters (`policy_wm_parameter_overlap=0`), all policy parameters are
on CUDA, both policy optimizers are `AdoptAtan2`, and optimizer state is
initially empty.

**Conclusion:** the shared encoder receives two separately normalized
optimizer updates.  This is a real secondary integration risk, but the
sign-corrected run is stable without changing it, so it is not the isolated
cause of the NOOP collapse.

### Root-cause summary

| Check | Result | Role |
|---|---|---|
| 1a warmup | no warmup; policy updates from iteration 1 | ruled out |
| 1b reward | observed [-0.70, +1.10]; support [-1, 1]; no clipping | not causal |
| 1c entropy/sign | installed surrogate gradient has the wrong sign | **root cause** |
| 1d actions | initial policy broad; original checkpoint 100% NOOP | consequence |
| 1e gradients | actor and Impala gradients finite and connected; no WM leak | ruled out |
| 1f optimizers | Impala shared by actor/critic optimizers | secondary risk |

## Step 2: Minimal repair and 100k acceptance run

### Repair

`scripts/twm_baseline_diagnostic.py` now overrides `CompatAgent.policy_loss`
with `-torch.min(surr1, surr2) - self.actor_beta_s * action_entropy`.  It also
adds `policy_entropy_mean`, `noop_fraction`, `mode_counts`, and
`mode_noop_fraction` to the evaluation record.  No TWM source file was edited.

### Micro-check

With the corrected sign, the five-iteration probe ended at policy entropy
2.66649 and max probability 0.16261, versus 0.00252 and 0.99978 before.

### 100k acceptance run

Command:

```bash
python scripts/twm_baseline_diagnostic.py \
  --max-env-steps 100000 --no-save-checkpoints \
  --json-out ~/craftax-workspace/logs/twm_crash_fixed_round1_100k.json
```

| Criterion | Required | Measured at 100k | Result |
|---|---:|---:|---|
| NOOP fraction | < 10% | **2.3%** | pass |
| Policy entropy | > 0.5 | **2.51 nats** | pass |
| Reward/step | > 0 | **+0.0258** | pass |

Full 10k curve:

| Steps | Reward/step | Tier | Ach. | Coverage | NOOP | Policy entropy | Action entropy | WM loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10000 | +0.0260 | 1 | 6 | 0.0030 | 4.5% | 2.53 | 2.57 | 13.88 |
| 20000 | +0.0216 | 1 | 6 | 0.0031 | 1.8% | 2.48 | 2.52 | 11.60 |
| 30000 | +0.0259 | 1 | 6 | 0.0029 | 2.5% | 2.50 | 2.52 | 10.02 |
| 40000 | +0.0320 | 1 | 8 | 0.0022 | 2.1% | 1.99 | 2.00 | 8.78 |
| 50000 | +0.0255 | 1 | 6 | 0.0037 | 2.4% | 2.02 | 2.04 | 8.04 |
| 60000 | +0.0297 | 0 | 6 | 0.0032 | 1.5% | 2.07 | 2.16 | 6.16 |
| 70000 | +0.0291 | 1 | 7 | 0.0042 | 1.4% | 2.36 | 2.42 | 5.12 |
| 80000 | +0.0236 | 0 | 6 | 0.0041 | 3.5% | 2.47 | 2.59 | 4.52 |
| 90000 | +0.0292 | 1 | 6 | 0.0032 | 3.2% | 2.45 | 2.60 | 4.22 |
| 100000 | +0.0258 | 0 | 6 | 0.0033 | 2.3% | 2.51 | 2.56 | 3.87 |

The 100k run reached tier 1 at 8/10 checkpoints, reached achievements=8 at
40k, and killed at least one mob at the 10k checkpoint.  It did not yet reach a
completed tool recipe.  At the final checkpoint the sampled action histogram
was `[46,98,75,140,61,362,248,37,295,48,192,52,41,91,65,52,97]` out of 2,000
actions, with DO (index 5) the most frequent action at 18.1%.

## Step 3: 1M-step confirmation

Command:

```bash
python scripts/twm_baseline_diagnostic.py \
  --max-env-steps 1000000 --no-save-checkpoints \
  --json-out ~/craftax-workspace/logs/twm_crash_fixed_round1_1m.json
```

The run completed 1,000,000 environment steps in **104.33 min**.  The GPU was
shared during this run with an unrelated local process
(`scripts/04_train_jscc.py`, which was already running outside this project),
so wall time and `nvidia-smi` peaks are upper bounds rather than isolated
measurements.  The torch allocator peak for this harness was 10,769.5 MB,
matching the original diagnostic; the 24,191 MB `nvidia-smi` high-water mark
includes the other process.

Acceptance criteria at 1M:

| Criterion | Required | Repaired TWM at 1M | Result |
|---|---:|---:|---|
| NOOP fraction | < 10% | **1.35%** | pass |
| Policy entropy | > 0.5 | **2.010 nats** | pass |
| Reward/step | > 0 | **+0.03015** | pass |

Final evaluation details:

| Metric | Repaired TWM 1M | Pure PPO 1M |
|---|---:|---:|
| Final reward/step | +0.03015 | +0.03095 |
| Last-10 reward/step | +0.02829 | +0.02927 |
| Best reward/step | +0.03610 at 900k | +0.03640 at 860k |
| Final NOOP | 1.35% | 1.20% |
| Last-10 NOOP | 1.44% | 0.90% |
| Final policy entropy | 2.010 | 2.019 |
| Last-10 policy entropy | 2.138 | 1.941 |
| Final action entropy | 2.141 | 2.142 |
| Final coverage | 0.250% | 0.311% |
| Last-10 coverage | 0.344% | 0.348% |
| Best coverage | 0.659% at 750k | 0.464% at 920k |
| Final tier / best tier | 0 / 1 | 0 / 1 |
| Final achievements / best | 6 / 9 at 900k | 6 / 9 at 210k |
| Crafting-labelled eval transitions | 112/1,992 = 5.62% | 127/1,992 = 6.38% |
| Runtime | 104.33 min (shared GPU) | 34.36 min |

The repaired policy is stable for the full 1M steps: NOOP stays below 6.1% at
every checkpoint, policy entropy stays above 1.6, and reward is positive at
every checkpoint.  The previous deterministic-NOOP failure is gone.

### Does the repaired TWM beat pure PPO?

**No reliable advantage is demonstrated.**  Single-seed results are mixed:

- Reward, final NOOP, final coverage, final achievements, and crafting-event
  rate are effectively tied, with small differences in either direction.
- The repaired TWM has a higher last-10 policy entropy (2.138 vs 1.941) and a
  higher best coverage (0.659% vs 0.464%), but a lower final coverage (0.250%
  vs 0.311%) and worse last-10 NOOP (1.44% vs 0.90%).
- Both methods finish at tier 0, never complete a tool recipe, and have nearly
  identical deterministic modes (DO for all 2,000 evaluation actions).
- The TWM harness is about 3x slower wall-clock and 1.7x slower per environment
  step in this configuration, and its 115.53M-parameter WM adds no measured
  mechanism gain.

Because the differences are small, contradictory across metrics, and from one
seed, the correct conclusion is **parity without a demonstrated advantage**,
not "TWM is better" and not "TWM is worse".  The world-model route should not
be used as the headline mechanism on the strength of this result.

## Artifacts and hashes

- Probe script: `scripts/twm_crash_probe.py`
- Harness script after repair: `scripts/twm_baseline_diagnostic.py`
- Probe JSONs: `~/craftax-workspace/logs/twm_crash_probe_*.json`
- 100k JSON: `~/craftax-workspace/logs/twm_crash_fixed_round1_100k.json`
- 100k log: `~/craftax-workspace/logs/twm_crash_fixed_round1_100k.log`
- 1M JSON: `~/craftax-workspace/logs/twm_crash_fixed_round1_1m.json`
- 1M log: `~/craftax-workspace/logs/twm_crash_fixed_round1_1m.log`

SHA-256:

```text
f0dbe5d862d85320afff72111bc8b2ff777f3b3f56f60c508cbce904768af4c2  scripts/twm_baseline_diagnostic.py
b80418ac0a2865cf56c79e9d89528f9322706a6ab43b0f92b349e99d28e57948  scripts/twm_crash_probe.py
af41674552c45bb2b4ece09222faf4fa2a9680c233cbe62879509b458cb9c53f  twm_crash_fixed_round1_100k.json
d85478bd4e15b256633d534b89986ad2d4c2e402d18e41f46bc4089a6adf7bf4  twm_crash_fixed_round1_100k.log
6de08dddbb9e80aa39e8eb4f26c1fdc95f037e501150fad0944b070b4e89a84f  twm_crash_fixed_round1_1m.json
a351834579d41e334cfe2b8599fda9933266ab1e95d5b097b9494af2365a99b4  twm_crash_fixed_round1_1m.log
```
