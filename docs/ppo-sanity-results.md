# Pure PPO Sanity Check (1M Environment Steps)

Date: 2026-10-27

Status: **complete; Step 3 result = Case C (partial)**.  The TWM integration
NOOP collapse it identified has since been diagnosed and repaired; see the
post-repair addendum at the end of this file.

Evidence labels: timings and metrics below are **command-verified** from
`~/craftax-workspace/logs/ppo_baseline_sanity.json`; API/content claims marked
**source-inspected**; causal interpretations are **inferred**, and a one-seed
run cannot establish seed variance.

## Executive result

The pure Craftax_Baselines PPO run does **not** reproduce the TWM harness
collapse.  The final policy samples NOOP for only **1.2%** of evaluation actions,
has **2.019 nats** of policy entropy, and obtains **+0.03095 reward/step**.  It
is therefore not an entropy-collapsed policy.

It is also not a normal exploration baseline: final frontier coverage is only
**0.311%** (best 0.464%), the final crafting tier is **0**, and the deterministic
policy chooses DO for all 2,000 evaluation actions.  The run learned some
low-level behaviour and sparse achievements, but not broad exploration or a
completed tool recipe.  This is **Case C (partial)**, not Case A or Case B.

Most importantly, the earlier "rare-event data starvation" premise does not
hold as a general property of the harness.  Under the pure PPO policy,
inventory-changing (crafting-labelled) evaluation transitions are **127/1,992 =
6.38%**, about **97x** the TWM diagnostic's **0.066%** evaluation rate and
about **70x** its **0.091%** training audit.  The sparsity was produced mainly
by the collapsed TWM policy, not by Craftax's event definition.

## 1. What was run

The script calls the unmodified official `make_train` from Craftax_Baselines
revision `7ce36fa05b84a2c9e758012f1e6da402e1e3a891`.  That function computes a
per-update metric but intentionally discards it and returns only the final
runner state.  To retain 10k-step checkpoints without editing third-party
source, `scripts/ppo_baseline_sanity.py` captures the outer `_update_step`
closure and its initial runner state while `make_train` is constructing the
program, then drives that exact closure with `jax.lax.scan`.

| Item | Value |
|---|---|
| Environment | `Craftax-Classic-Pixels-v1` |
| Seed | 0 |
| Total environment steps | 1,000,000 |
| Parallel envs / rollout | 500 / 20 steps |
| Updates | 100 |
| PPO minibatch | 1,250 x 8 minibatches |
| PPO epochs | 4 |
| Learning rate | 2e-4, linear annealing (official default) |
| Entropy coefficient | 0.01 (official default) |
| Optimistic resets | disabled |
| Checkpoint/evaluation interval | every 10,000 steps |
| Evaluation | 8 envs x 250 sampled actions |
| JAX / Craftax | 0.6.2 / 1.6.1 |
| Runtime | 2,061.84 s = **34.36 min** |
| Throughput | 485.0 env steps/s |
| Peak `nvidia-smi` memory | 6,229 MB |

### Deviations from upstream defaults

Upstream `ppo.py` defaults to 1,024 envs x 64 steps and a 1e9-step budget.  An
exact 10k-step evaluation requires the update size to divide 10,000, so this
run uses 500 x 20.  The TWM diagnostic used 50 x 20; that small PPO batch made
the pure PPO run unnecessarily slow.  The environment, seed, total budget,
20-step rollout horizon, PPO objective, optimizer, and official hyperparameters
are otherwise unchanged.  This is not the upstream 1e9-step benchmark
configuration, and it is a single seed.

## 2. Headline comparison

| Metric | Pure PPO | TWM harness (diagnostic) |
|---|---:|---:|
| Final reward | **+0.03095/step** | -0.00460/step |
| Last-10-checkpoint reward | **+0.02927/step** | -0.00435/step |
| Final crafting tier | **0** (best 1) | 0 |
| Final exploration coverage | **0.311%** | 0.024% |
| Best exploration coverage | **0.464% at 920k** | 0.934% (38.25 tiles) |
| Final NOOP fraction | **1.2%** | 100% |
| NOOP range over checkpoints | **0.20%-5.75%** | 100% at final; final policy NOOP |
| Final policy entropy | **2.019 nats** | final 0; <0.01 at 74/100 |
| Checkpoints with entropy >1.0 | **100/100** | 8/100 |
| Final achievements | **6** | 0 |
| Best achievements | **9 at 210k** | 5 at 520k |
| Kill achievements | **1 final / 1 best** | 0 ever |
| Inventory-changing crafting labels | **6.38% of eval transitions** | 0.066% of eval transitions |
| Deterministic action mode | **DO for all 2,000** | NOOP for all 2,000 |

The TWM row is the pre-existing diagnostic; it is not an official TWM
reproduction.  The comparison is between two local harnesses, not between the
PPO result and the paper's 69.66% anchor.

## 3. Learning dynamics and mechanics

- Reward rises from **+0.01370/step at 10k** to roughly **+0.03/step** by
  200k, then plateaus.  The best checkpoint is **+0.03640 at 860k**.  The
  first-10-checkpoint mean is **+0.02185**, compared with **+0.02927** over the
  last ten checkpoints.  The fitted slope over the whole run is only
  **+8.65e-9 reward/step**, so the trend is a bounded early improvement, not
  continued scaling.
- Policy entropy starts at **2.833**, falls to a minimum of **1.394**, then
  recovers to **2.019** at 1M.  It never collapses: all 100 checkpoints have
  policy entropy above 1.0.
- NOOP is never the dominant sampled action: its maximum over checkpoints is
  **5.75%**.  At 1M, sampled actions are 37.9% DO and 22.85% directional
  movement; all remaining actions are placement/crafting attempts.
- Final evaluation reaches crafting tier **0**; tier 1 appears at 37/100
  checkpoints but is not sustained.  Final resources are wood=3 and sapling=4,
  with no tool in inventory.  The deterministic policy is stuck on DO.
- Achievements are non-monotone: 6 at the final checkpoint, best 9 at 210k.
  One combat achievement is present at the final checkpoint.
- Final coverage is 0.311% (about 12.7 of 4,096 tiles), far below the 5%
  movement criterion.  Best coverage is 0.464% at 920k.  Survival averages
  105.5 steps, with 12 finished episodes in the 2,000-step evaluation.

## 4. The three required questions

### Q1. Did pure PPO learn to move (coverage > 5%)?

**No.**  It selects directional actions and moves locally, but final coverage
is **0.311%** and the best checkpoint is **0.464%**, both below 5%.  The
deterministic policy always selects DO.  This is partial movement, not broad
exploration.

### Q2. Did pure PPO also suffer entropy collapse?

**No.**  Policy entropy is at least **1.394 nats** at every checkpoint and
**2.019** at the end; NOOP never exceeds **5.75%**.  The TWM harness's 100%
NOOP policy is not reproduced by pure PPO under this budget.

### Q3. Does pure PPO reward show an upward trend?

**Yes, but only a small early improvement that plateaus.**  Reward rises from
**+0.01370** at 10k to about **+0.03** by 200k and remains around that level.
The last-10 mean is **+0.02927**, above the first-10 mean of **+0.02185**.
This shows the reward path can produce a positive learning signal; it does not
show a fully working 1M-step Craftax agent.

## 5. Step 3 classification

**Case C - partial.**

- Not Case A: pure PPO is not a normal exploration baseline; coverage is far
  below 5% and recipe progress is zero at the final checkpoint.
- Not Case B: pure PPO does not reproduce the TWM harness's NOOP/entropy
  collapse.  Entropy stays above 1.0, reward is positive, and sampled actions
  are diverse.
- Case C fits the evidence: basic sampled behavior is alive, but broad
  exploration and successful crafting are not learned.  The TWM harness adds
  a separate, much more severe policy collapse that invalidates its earlier
  failure-pattern diagnosis.

The status is `PARTIAL`: neither a clean "TWM problem confirmed" result nor a
pure harness crash.  A one-seed 1M-step result cannot distinguish sample
budget, PPO hyperparameters, sparse reward, and exploration failure.

## 6. What this does to CMS-WM

The CMS-WM design is **not supported in its current form** and must not be
implemented yet.  The specific points needing rework are:

1. **Retract the general starvation premise.**  The 0.091% training audit was
   measured under a collapsed policy.  Pure PPO produces inventory-changing
   events at 6.38% in evaluation.  CMS-WM cannot claim that Craftax's event
   stream is inherently starved.
2. **Separate resource pickup from successful recipe completion.**  Wood and
   sapling changes are common, while no tool and no crafting tier survive.
   The useful hypothesis is about long-horizon recipe credit assignment, not
   generic inventory-event frequency.
3. **Do not attribute the TWM failure to its tokenizer or transformer.**  A
   pure PPO policy also ends at tier 0 without any world model.  The prior
   architectural attribution was unconfirmed and is now further weakened.
4. **Fix and characterize the TWM integration first.**  The TWM harness's 100%
   NOOP policy is a separate integration/policy-training failure.  It must be
   debugged before its world-model metrics can support any method design.
5. **Re-justify CMS-WM against two controls.**  Any future version needs a
   pure PPO control and a repaired TWM control under the same data budget,
   with recipe-level success counts, not only per-bucket reconstruction error.
6. **Keep the first gate offline.**  Before any closed-loop method run, test
   whether a factorized mechanic state improves prediction of successful
   recipe transitions on a fixed, balanced dataset.  If it does not, the
   architecture hypothesis is false.

`docs/method-design.md` carries the corresponding status.  It was updated
from "pending sanity check" to "TWM integration repaired, parity with pure
PPO" once the addendum at the end of this file was measured.

## 7. Artifacts

- Script: `scripts/ppo_baseline_sanity.py`
- JSON: `~/craftax-workspace/logs/ppo_baseline_sanity.json`
- Log: `~/craftax-workspace/logs/ppo_baseline_sanity.log`
- Checkpoints: 100 records, one per 10k environment steps
- Script SHA-256: `f3a47b946413a60641f6a2feade4f3a59ae7c8a70485983e5d19db9cceaaec55`
- JSON SHA-256: `873538b2b898ab2c42b44bb28f3de49de57c9cbcaff77452be51aa732eeaf7ff`
- Log SHA-256: `515b6eec32f0843b9dbf8ad822b4f4c9d884aa3f0fef001ace5e5090c8cd45e8`

## Appendix: full 10k checkpoint curve

All actions are sampled from the policy on 8 evaluation environments for 250
steps.  Coverage is the mean peak fraction of the 64x64 map visited per
environment.  Entropy is the mean categorical policy entropy; action entropy
is the empirical entropy of the sampled action histogram.

| Steps | Reward/step | Tier | Ach. | Coverage | NOOP | Policy entropy | Action entropy | Survival |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10000 | 0.01370 | 0 | 5 | 0.317% | 5.45% | 2.833 | 2.830 | 109.3 |
| 20000 | 0.01770 | 0 | 6 | 0.336% | 5.60% | 2.826 | 2.820 | 143.3 |
| 30000 | 0.01960 | 0 | 6 | 0.345% | 4.85% | 2.808 | 2.803 | 132.1 |
| 40000 | 0.02615 | 0 | 6 | 0.369% | 4.75% | 2.783 | 2.776 | 113.2 |
| 50000 | 0.02005 | 0 | 6 | 0.317% | 5.75% | 2.741 | 2.751 | 128.0 |
| 60000 | 0.02345 | 0 | 5 | 0.302% | 3.85% | 2.651 | 2.649 | 127.2 |
| 70000 | 0.02375 | 0 | 5 | 0.299% | 3.00% | 2.568 | 2.568 | 115.5 |
| 80000 | 0.02540 | 0 | 5 | 0.253% | 3.30% | 2.492 | 2.500 | 132.8 |
| 90000 | 0.02555 | 0 | 6 | 0.241% | 2.25% | 2.363 | 2.331 | 125.0 |
| 100000 | 0.02315 | 0 | 5 | 0.226% | 2.35% | 2.262 | 2.283 | 103.2 |
| 110000 | 0.02845 | 1 | 6 | 0.214% | 1.70% | 2.124 | 2.136 | 84.2 |
| 120000 | 0.02360 | 0 | 6 | 0.220% | 1.70% | 2.049 | 2.055 | 145.4 |
| 130000 | 0.02015 | 0 | 6 | 0.247% | 1.65% | 1.923 | 1.989 | 181.4 |
| 140000 | 0.02225 | 1 | 6 | 0.217% | 0.90% | 1.780 | 1.809 | 122.8 |
| 150000 | 0.02410 | 1 | 6 | 0.183% | 1.15% | 1.755 | 1.803 | 120.2 |
| 160000 | 0.02545 | 1 | 6 | 0.198% | 1.00% | 1.655 | 1.711 | 134.6 |
| 170000 | 0.02395 | 0 | 6 | 0.168% | 1.20% | 1.591 | 1.599 | 109.6 |
| 180000 | 0.02370 | 0 | 4 | 0.134% | 0.65% | 1.486 | 1.593 | 117.2 |
| 190000 | 0.02660 | 0 | 5 | 0.156% | 0.60% | 1.394 | 1.469 | 96.9 |
| 200000 | 0.02750 | 0 | 6 | 0.125% | 0.80% | 1.487 | 1.532 | 115.1 |
| 210000 | 0.02990 | 1 | 9 | 0.146% | 0.80% | 1.596 | 1.642 | 105.4 |
| 220000 | 0.02295 | 1 | 6 | 0.140% | 0.70% | 1.701 | 1.729 | 133.7 |
| 230000 | 0.02680 | 0 | 5 | 0.177% | 0.85% | 1.558 | 1.654 | 104.5 |
| 240000 | 0.02190 | 0 | 5 | 0.140% | 0.75% | 1.727 | 1.764 | 122.3 |
| 250000 | 0.02710 | 1 | 7 | 0.156% | 0.85% | 1.664 | 1.703 | 144.0 |
| 260000 | 0.02555 | 0 | 6 | 0.150% | 0.30% | 1.676 | 1.788 | 101.4 |
| 270000 | 0.02550 | 0 | 6 | 0.143% | 0.85% | 1.685 | 1.740 | 115.2 |
| 280000 | 0.02470 | 0 | 5 | 0.171% | 0.80% | 1.754 | 1.861 | 101.4 |
| 290000 | 0.02375 | 1 | 6 | 0.140% | 1.10% | 1.757 | 1.859 | 126.6 |
| 300000 | 0.03035 | 0 | 6 | 0.156% | 0.70% | 1.637 | 1.657 | 102.4 |
| 310000 | 0.02090 | 1 | 6 | 0.168% | 0.90% | 1.771 | 1.833 | 141.8 |
| 320000 | 0.02250 | 0 | 5 | 0.177% | 0.70% | 1.796 | 1.886 | 152.9 |
| 330000 | 0.02500 | 1 | 6 | 0.143% | 0.60% | 1.762 | 1.816 | 124.7 |
| 340000 | 0.02305 | 0 | 5 | 0.156% | 0.75% | 1.667 | 1.715 | 145.7 |
| 350000 | 0.03120 | 0 | 5 | 0.159% | 0.60% | 1.660 | 1.739 | 99.4 |
| 360000 | 0.03155 | 0 | 6 | 0.208% | 0.75% | 1.762 | 1.854 | 105.7 |
| 370000 | 0.02915 | 0 | 6 | 0.217% | 1.05% | 1.877 | 1.983 | 123.4 |
| 380000 | 0.02825 | 0 | 5 | 0.189% | 0.80% | 1.828 | 1.919 | 120.8 |
| 390000 | 0.02590 | 0 | 6 | 0.256% | 0.80% | 1.803 | 1.898 | 156.9 |
| 400000 | 0.02360 | 1 | 6 | 0.244% | 0.55% | 1.810 | 1.882 | 132.1 |
| 410000 | 0.03000 | 0 | 7 | 0.192% | 0.30% | 1.731 | 1.792 | 113.2 |
| 420000 | 0.03030 | 1 | 7 | 0.174% | 0.50% | 1.799 | 1.895 | 110.4 |
| 430000 | 0.02850 | 0 | 6 | 0.198% | 0.65% | 1.794 | 1.895 | 89.4 |
| 440000 | 0.02625 | 0 | 6 | 0.253% | 0.90% | 1.927 | 2.014 | 140.5 |
| 450000 | 0.02935 | 1 | 7 | 0.241% | 0.80% | 1.899 | 2.017 | 121.0 |
| 460000 | 0.03160 | 0 | 6 | 0.241% | 0.65% | 1.899 | 1.990 | 114.6 |
| 470000 | 0.02540 | 1 | 6 | 0.296% | 1.20% | 1.973 | 2.079 | 150.8 |
| 480000 | 0.02350 | 0 | 5 | 0.211% | 0.55% | 1.789 | 1.874 | 122.0 |
| 490000 | 0.02860 | 1 | 7 | 0.150% | 0.40% | 1.705 | 1.744 | 91.7 |
| 500000 | 0.03155 | 0 | 5 | 0.266% | 0.90% | 1.780 | 1.838 | 104.8 |
| 510000 | 0.02975 | 1 | 7 | 0.235% | 0.30% | 1.738 | 1.797 | 150.3 |
| 520000 | 0.02885 | 1 | 6 | 0.229% | 0.20% | 1.723 | 1.745 | 140.0 |
| 530000 | 0.03175 | 0 | 6 | 0.201% | 0.60% | 1.704 | 1.733 | 97.2 |
| 540000 | 0.02660 | 0 | 6 | 0.235% | 0.55% | 1.881 | 1.936 | 107.7 |
| 550000 | 0.02290 | 0 | 5 | 0.284% | 0.55% | 1.860 | 1.937 | 131.4 |
| 560000 | 0.02705 | 0 | 6 | 0.314% | 0.70% | 1.781 | 1.859 | 133.9 |
| 570000 | 0.02925 | 1 | 7 | 0.241% | 0.65% | 1.758 | 1.824 | 88.4 |
| 580000 | 0.02910 | 1 | 8 | 0.262% | 0.70% | 1.805 | 1.917 | 106.3 |
| 590000 | 0.02860 | 0 | 5 | 0.272% | 0.30% | 1.702 | 1.716 | 100.5 |
| 600000 | 0.02750 | 0 | 6 | 0.229% | 0.90% | 1.848 | 1.877 | 132.1 |
| 610000 | 0.03080 | 0 | 7 | 0.259% | 0.80% | 1.801 | 1.868 | 104.4 |
| 620000 | 0.02635 | 0 | 5 | 0.305% | 0.60% | 1.840 | 1.942 | 147.6 |
| 630000 | 0.03555 | 1 | 6 | 0.229% | 0.70% | 1.822 | 1.881 | 90.2 |
| 640000 | 0.02940 | 0 | 5 | 0.253% | 0.40% | 1.878 | 1.926 | 76.5 |
| 650000 | 0.02900 | 0 | 5 | 0.214% | 1.00% | 1.927 | 1.995 | 111.8 |
| 660000 | 0.02495 | 1 | 7 | 0.311% | 0.75% | 1.931 | 2.018 | 125.1 |
| 670000 | 0.02930 | 0 | 7 | 0.229% | 0.85% | 1.863 | 1.923 | 115.2 |
| 680000 | 0.03180 | 0 | 6 | 0.323% | 0.85% | 1.837 | 1.932 | 102.9 |
| 690000 | 0.02455 | 0 | 5 | 0.284% | 0.85% | 1.814 | 1.919 | 128.3 |
| 700000 | 0.02575 | 0 | 6 | 0.281% | 0.85% | 1.902 | 2.019 | 147.6 |
| 710000 | 0.02800 | 1 | 6 | 0.363% | 0.60% | 1.884 | 1.991 | 116.0 |
| 720000 | 0.03500 | 1 | 6 | 0.299% | 0.60% | 1.826 | 1.905 | 104.3 |
| 730000 | 0.03115 | 1 | 6 | 0.314% | 0.65% | 1.912 | 1.961 | 111.5 |
| 740000 | 0.02930 | 0 | 7 | 0.299% | 0.40% | 1.782 | 1.797 | 109.0 |
| 750000 | 0.03440 | 0 | 6 | 0.296% | 0.40% | 1.836 | 1.908 | 111.1 |
| 760000 | 0.02830 | 1 | 6 | 0.366% | 0.30% | 1.943 | 1.992 | 106.6 |
| 770000 | 0.03465 | 1 | 7 | 0.302% | 0.90% | 1.906 | 2.004 | 104.6 |
| 780000 | 0.02880 | 1 | 6 | 0.327% | 0.55% | 1.910 | 1.974 | 92.6 |
| 790000 | 0.02950 | 0 | 6 | 0.320% | 0.45% | 1.864 | 1.944 | 111.7 |
| 800000 | 0.03220 | 1 | 7 | 0.369% | 0.80% | 1.806 | 1.886 | 129.8 |
| 810000 | 0.02785 | 0 | 6 | 0.278% | 0.60% | 1.858 | 1.921 | 111.7 |
| 820000 | 0.03370 | 0 | 6 | 0.320% | 0.80% | 1.953 | 2.049 | 109.8 |
| 830000 | 0.02695 | 0 | 5 | 0.348% | 1.00% | 2.049 | 2.131 | 146.2 |
| 840000 | 0.02835 | 0 | 7 | 0.375% | 1.15% | 2.024 | 2.129 | 138.2 |
| 850000 | 0.02895 | 0 | 6 | 0.342% | 0.85% | 2.027 | 2.125 | 127.1 |
| 860000 | 0.03640 | 1 | 8 | 0.317% | 0.90% | 1.945 | 2.069 | 103.2 |
| 870000 | 0.03020 | 1 | 7 | 0.366% | 1.15% | 1.963 | 2.080 | 125.7 |
| 880000 | 0.03055 | 0 | 5 | 0.372% | 0.75% | 1.937 | 2.024 | 121.1 |
| 890000 | 0.03275 | 1 | 7 | 0.299% | 0.75% | 1.791 | 1.903 | 111.0 |
| 900000 | 0.02880 | 1 | 6 | 0.296% | 0.50% | 1.915 | 1.971 | 91.3 |
| 910000 | 0.02610 | 1 | 6 | 0.345% | 0.70% | 1.943 | 2.027 | 137.3 |
| 920000 | 0.03040 | 1 | 7 | 0.464% | 1.05% | 1.944 | 2.053 | 116.2 |
| 930000 | 0.03515 | 1 | 8 | 0.333% | 0.65% | 1.766 | 1.865 | 108.0 |
| 940000 | 0.02340 | 1 | 6 | 0.369% | 0.45% | 1.932 | 2.061 | 142.0 |
| 950000 | 0.02510 | 0 | 6 | 0.354% | 1.20% | 2.017 | 2.095 | 175.1 |
| 960000 | 0.02880 | 0 | 7 | 0.311% | 0.90% | 1.990 | 2.097 | 133.3 |
| 970000 | 0.03495 | 1 | 8 | 0.375% | 1.05% | 1.983 | 2.103 | 114.6 |
| 980000 | 0.02890 | 0 | 5 | 0.308% | 0.85% | 1.937 | 2.023 | 106.2 |
| 990000 | 0.02895 | 0 | 6 | 0.305% | 0.95% | 1.880 | 2.003 | 107.8 |
| 1000000 | 0.03095 | 0 | 6 | 0.311% | 1.20% | 2.019 | 2.142 | 105.5 |

## Evidence summary

- All numerical results: **command-verified** from the JSON and log.
- Capture of the official update closure without source edits:
  **source-inspected**.
- Generalization to other seeds or the upstream 1e9-step protocol:
  **unconfirmed**.
- Causal explanation for the remaining tier-0 recipe failure: **inferred**;
  the present data cannot separate sparse reward, exploration, and PPO
  budget/hyperparameters.

## Post-repair addendum: repaired TWM vs pure PPO (1M steps)

Added after the TWM integration NOOP collapse was diagnosed and fixed
(`docs/twm-crash-diagnosis.md`).  Sections 1-7 above are the historical Case C
analysis and are preserved unchanged.  The TWM rows in those tables are the
pre-repair diagnostic and are superseded by the "repaired TWM" column here.

Root cause of the collapse: an inverted PPO surrogate sign in TWM 0.0.59's
`Agent.policy_loss` (it returns `min(surr1, surr2) - beta*entropy` and
`learn` minimizes it, instead of subtracting the surrogate).  The harness now
overrides only that method; no third-party source was changed.  With the
corrected sign the TWM policy no longer collapses to NOOP.

Comparison at 1M environment steps, single seed 0, same harness:

| Metric | Repaired TWM 1M | Pure PPO 1M |
|---|---:|---:|
| Final reward/step | +0.03015 | +0.03095 |
| Last-10 reward/step | +0.02829 | +0.02927 |
| Best reward/step | +0.03610 at 900k | +0.03640 at 860k |
| Final NOOP | 1.35% | 1.20% |
| Last-10 NOOP | 1.44% | 0.90% |
| Final policy entropy | 2.010 | 2.019 |
| Last-10 policy entropy | 2.138 | 1.941 |
| Final coverage | 0.250% | 0.311% |
| Last-10 coverage | 0.344% | 0.348% |
| Best coverage | 0.659% at 750k | 0.464% at 920k |
| Final tier / best tier | 0 / 1 | 0 / 1 |
| Final achievements / best | 6 / 9 at 900k | 6 / 9 at 210k |
| Crafting-labelled eval transitions | 112/1,992 = 5.62% | 127/1,992 = 6.38% |
| Deterministic action mode | DO for all 2,000 | DO for all 2,000 |
| Runtime | 104.33 min (shared GPU) | 34.36 min |

Interpretation:

- The repaired TWM is at **parity with pure PPO**, not better.  The differences
  are small, contradictory across metrics, and from one seed.
- TWM is slightly higher on last-10 policy entropy (2.138 vs 1.941) and best
  coverage (0.659% vs 0.464%); slightly lower on final coverage and worse on
  last-10 NOOP.
- Neither method completes a tool recipe or reaches final tier > 0.
- **No world-model mechanism advantage is demonstrated**, so the world-model
  route must not be the headline mechanism for CMS-WM on this evidence.

This resolves the "fix and characterize the TWM integration first" item in
Section 6; the TWM harness is no longer in a collapsed state.  The remaining
CMS-WM re-argument requirements in Section 6 are unchanged.
