# TWM Baseline Diagnostic (1M Environment Steps)

Date: 2026-10-27

Host: WSL2 Ubuntu-22.04, RTX 3090 (24,576 MiB)

Project: `/mnt/e/Project/NyxFelineLabs/craftax-world-model`

Script: `scripts/twm_baseline_diagnostic.py` (sha256 `db613f1bb129...`)

Log: `~/craftax-workspace/logs/twm_baseline_diagnostic.log`

JSON: `~/craftax-workspace/logs/twm_baseline_diagnostic.json` (sha256 `ef1f38305992...`)

Checkpoints: `~/craftax-workspace/data/diagnostic_checkpoints/step_XXXXXXXX.pt` (100 files, 146 GB total)

Scope: one seed, one diagnostic run. The purpose is failure-mode discovery, not reproduction of the paper's 69.66% anchor.

Evidence labels used below: **command-verified** (script output/JSON), **source-inspected** (installed package code), **inferred** (consistent explanation not isolated by an experiment), and **unconfirmed** (no measurement).

## Executive result

The run completed all 1,000,000 environment steps in **77.87 min** with a torch-allocator peak of **10,769.5 MB** (`nvidia-smi` peak: 12,558 MB). The world model fit improved substantially, but closed-loop behavior did not:

- WM training loss: **13.3413 -> 2.0474**.
- Mean held-out frame-state cross-entropy after the last 10 checkpoints: **0.3577 normal**, **0.3891 exploration**, **0.5463 combat**, **0.5468 crafting**.
- Crafting is the persistent worst bucket: final error **0.5291 vs 0.4040 normal (1.31x)**; averaged over the last 10 checkpoints the ratio is **1.54x**.
- The policy produced **0 crafting tier**, **0 kills**, and **0 achievements** at the final checkpoint. It selected **NOOP for all 2,000 evaluation actions**.
- **74/100** checkpoints were effectively deterministic (action entropy below 0.01), while the policy repeatedly alternated between deterministic modes and short exploratory bursts. Mean last-10 action entropy was **0.0045**.
- A 100,000-transition training audit found **91 crafting events (0.091%)**, **3,925 exploration events (3.93%)**, and **18,307 combat-labelled events (18.31%)**.

The measured failure chain is therefore: **unstable policy/data coverage -> almost no crafting or tool-use experience -> the crafting/combat prediction buckets remain out of distribution -> no closed-loop mechanic progress**. The predictive gap is real, but this run does not isolate whether the gap is caused by the tokenizer, the transformer, or data starvation; the data-starvation explanation is already sufficient to explain it.

## 1. Agent interface and diagnostic configuration

### Agent interface (source-inspected)

`Agent.learn` accepts either one `MemoriesWithNextState` or a list of them. Each memory set contains:

- `state`: `(C, H, W)` float tensor, here `(3, 63, 63)`.
- `action`: scalar integer tensor; `action_log_prob`, `reward`, `value`, and `done`: scalar tensors.
- `world_embed`: empty here, because the diagnostic does not condition the policy on a world-model embedding.
- `next_state`: `(C, 1, H, W)`.
- `from_real_env`: boolean.
- `has_world_model_embed`: boolean; all sets in one call must agree.

`Agent.learn` concatenates the memory sets, computes GAE independently for each set, then updates the shared Impala encoder, actor, and critic. **Agent therefore includes policy training**, so the diagnostic uses Agent and does not fall back to Craftax_Baselines PPO. Default values are `batch_size=16`, `epochs=2`; this run explicitly used `batch_size=128`, `epochs=1`. `Agent.interact_with_env` is a batch-1 Python loop. For throughput, this diagnostic batched real-env interaction with `jax.jit(jax.vmap(...))` but kept the same policy call sequence: `impala(frame) -> actor(features, sample_action=True)`.

### Run configuration

| Item | Value |
|---|---|
| Environment | `Craftax-Classic-Pixels-v1` |
| Observation / actions | `(63, 63, 3)` float32 / 17 |
| Seed | 0 |
| Environment steps | 1,000,000 |
| Parallel envs / rollout | 50 / 20 steps = 1,000 steps per iteration |
| Iterations | 1,000 |
| WM | `dim=512`, `depth=4`, `block_size=81`, patch `7x7`, 115.53 M parameters |
| WM training | batch 8, 8-frame windows, 1 AdamW step per 1,000 env steps, lr `1e-4` |
| Policy training | `Agent.learn`, batch 128, 1 epoch per 1,000 env steps |
| Dyna ratio | one imagined trajectory (<=24 frames) per 1,000 real steps (our choice; paper ratio not exposed) |
| Checkpoints | every 10,000 env steps; WM, policy, critics, and optimizers |
| Evaluation | fixed 8 envs x 250 steps per checkpoint |
| Holdout | 1,000 fixed two-frame transition windows, 250 per mechanic, collected once with a random policy |

Known deviations and limitations:

- This is a simplified diagnostic harness, not the official TWM training recipe. The paper anchor (`69.66%` at 1M env steps) is **not directly comparable** to the reward values here.
- Held-out units are **two-frame transition windows**, not 1,000 full trajectories. This keeps per-checkpoint evaluation tractable and is stated rather than hidden.
- Mechanic labels come directly from `EnvState` (inventory, achievements, player/mob positions), while training uses pixels. This is privileged diagnostic labelling, not privileged model input.
- The held-out windows were collected with a random behavior policy. They therefore test generalization outside the learned policy's narrow distribution.
- Single seed: no confidence intervals or seed-variance claims are possible.
- Exact cumulative kill count is `UNEXTRACTABLE` in this harness. The reported `kills` is the count of the two Craftax combat achievement flags, not a kill counter. It stayed zero.

## 2. Learning curve and mechanic progress

The full 10k-step table is in [Appendix A](#appendix-a-full-10k-curves). Key values:

| Steps | WM loss | Train reward/step | Eval reward/step | Tier | Ach. | Kills* | Coverage (tiles) | Action entropy | Codes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10k | 13.3413 | -0.00510 | -0.00460 | 0 | 0 | 0 | 1.00 | 0.000 | 138 |
| 100k | 3.7627 | -0.00340 | +0.00990 | 0 | 4 | 0 | 18.62 | 2.783 | 843 |
| 200k | 2.5480 | -0.00590 | -0.00480 | 0 | 0 | 0 | 15.75 | 0.000 | 1,072 |
| 300k | 2.4050 | -0.00510 | -0.00460 | 0 | 0 | 0 | 1.00 | 0.000 | 1,275 |
| 400k | 2.2487 | +0.00950 | -0.00365 | 0 | 1 | 0 | 15.62 | 0.368 | 1,414 |
| 500k | 2.2361 | -0.00660 | -0.00460 | 0 | 0 | 0 | 1.00 | 0.000 | 1,557 |
| 600k | 2.0748 | -0.00770 | -0.00480 | 0 | 0 | 0 | 15.75 | 0.000 | 1,667 |
| 700k | 1.9785 | -0.00600 | -0.00460 | 0 | 0 | 0 | 1.00 | 0.000 | 1,766 |
| 800k | 2.1271 | -0.00370 | -0.00480 | 0 | 0 | 0 | 15.75 | 0.000 | 1,859 |
| 900k | 1.9313 | -0.00540 | -0.00460 | 0 | 0 | 0 | 1.00 | 0.000 | 1,951 |
| 1M | 2.0474 | -0.00620 | -0.00460 | 0 | 0 | 0 | 1.00 | 0.000 | 2,061 |

`*` Kill achievements, not a cumulative kill counter.

Mechanic summary:

| Mechanic | Final | Best observed | Interpretation |
|---|---:|---:|---|
| Crafting tier | 0 | 0 | Never advanced beyond tier 0 at any checkpoint |
| Achievements | 0 | 5 at 520k | Non-monotone; final checkpoint had zero |
| Kill achievements | 0 | 0 | No combat milestone reached |
| Exploration coverage | 1 tile (0.024%) | 38.25 tiles (0.934%) | Short bursts, no sustained frontier expansion |
| Resources | all 0 | wood 2, sapling 3 | Never obtained a tool or crafted an item |
| Survival | 145.2 steps mean | 211 steps max | 10 episodes over 2,000 final evaluation steps |
| Reward | -0.00460/step | +0.01590/step at one checkpoint | No sustained return; last-10 checkpoint mean -0.00435/step |

The final policy chose action index 0 (**NOOP**) for all 2,000 evaluation steps. Across the 100 checkpoints, the deterministic action mode changed repeatedly (12 distinct modes), and action entropy was below 0.01 at 74 checkpoints while exceeding 1.0 at only 8. This is **policy instability**, not a stable exploratory policy that merely needs more training.

## 3. Prediction-error curves

The full 10k-step holdout table is in [Appendix B](#appendix-b-full-10k-holdout-curves). Bucket means:

| Checkpoint set | Normal | Exploration | Combat | Crafting |
|---|---:|---:|---:|---:|
| First 10 checkpoints (10k-100k) | 1.9777 | 2.0276 | 2.4976 | 2.7007 |
| Last 10 checkpoints (910k-1M) | 0.3577 | 0.3891 | 0.5463 | 0.5468 |
| Final checkpoint | 0.4040 | 0.3953 | 0.5035 | 0.5291 |
| Best (minimum) | 0.3112 | 0.3594 | 0.5017 | 0.5111 |

All buckets fell by at least **91.9%** from the start of training. The model is learning; the remaining problem is not a failure to optimize the training objective. The ordering is stable:

- Crafting is the highest final bucket and **1.31x normal** at the final checkpoint.
- Combat is second at **1.25x normal**.
- Exploration is essentially tied with normal at the final checkpoint (`0.3953 vs 0.4040`) even though coverage collapsed.
- Averaged over the last 10 checkpoints, craft/normal and combat/normal are both **1.54x**; exploration/normal is **1.09x**.

The held-out windows come from a random behavior policy, so these are distribution-shift errors. They are not evidence that the WM fails on the training distribution, where loss reached 2.05.

## 4. Failure Mode Analysis

### Q1. Which game mechanic fails most severely?

**Crafting and combat, with crafting worse at the final checkpoint.** Crafting has the highest held-out prediction error (`0.5291`, 1.31x normal) and zero closed-loop progress (`craft_tier_max=0`, no tools, all final inventory entries zero). Combat is next (`0.5035`, 1.25x normal) and the agent never earned either combat achievement. Exploration is not the highest prediction-error bucket, but the policy explores only one tile at the final checkpoint, so this is also a behavioral failure.

### Q2. What is the failure mode: prediction error, policy stagnation, memory loss, or exploration failure?

The dominant measured failure is **unstable policy plus severe data starvation**, not insufficient WM optimization:

1. **Policy stagnation/instability.** Final weights produce a deterministic NOOP policy. Action entropy is `<0.01` at 74/100 checkpoints; the deterministic mode changes across checkpoints and only eight checkpoints have entropy above 1.0. Reward neither improves nor deteriorates materially: first-10 mean `-0.00325/step`, last-10 mean `-0.00435/step`.
2. **Rare-event starvation.** A stratified 100 x 1,000-step training audit (100,000 transitions sampled across training) observed only **91 crafting events (0.091%)**, versus **3,925 exploration (3.93%)** and **18,307 combat-labelled (18.31%)** events. 65/100 audited windows saw zero crafting events. A model cannot learn recipe-conditioned transitions from this stream at a useful rate.
3. **Evaluation confirms the same coverage gap.** Across 200,000 evaluation transitions the labels were: normal 158,269; exploration 8,276; combat 32,524; crafting **131 (0.066%)**. The final checkpoint saw no crafting event at all.
4. **Prediction error has a persistent mechanic gradient.** Despite a 92% fall in normal-frame error, crafting remains at least 1.31x normal at the end and 1.54x on average over the final 10 checkpoints. Combat shows the same pattern. This is consistent with held-out events being outside the narrow behavior distribution.

The data cannot distinguish "the model cannot represent inventory/achievement transitions" from "the model never saw enough such transitions". The latter is sufficient and is directly measured.

### Q3. Is the failure attributable to a TWM architectural defect?

**Not established; the likely architecture connection is inferred, not verified.** Source inspection shows that the 115.53 M model is a pixel-patch world model:

- 7x7x3 patches are mapped to nearest-neighbor discrete codes (2,061 codes by the final checkpoint; configured cap 100,000).
- A block-causal transformer predicts the next patch tokens.
- Reward and terminal predictions are scalar heads.
- There is no explicit inventory slot, achievement state, recipe graph, persistent map, or entity table.

That design is a plausible reason why sparse HUD changes (inventory/achievement events) and spatially extended exploration are harder to predict than ordinary frames. It is also consistent with crafting/combat being the persistent error buckets. However, this run has no ablation that freezes data coverage, swaps the tokenizer, or supplies the same data to a structured-state model. Therefore the module-level causal claim is **inferred/unconfirmed**, not proven.

The Dyna schedule is also not a sufficient explanation by itself: one imagined trajectory per 1,000 real steps is deliberately conservative and was chosen here because the package does not expose the paper's exact ratio. It can amplify policy issues, but it cannot create crafting data that the behavior policy never collects.

## 5. Consequences for the method design

The diagnostic supports three design targets, in order:

1. Make discrete mechanic transitions representable and learnable (inventory/crafting/achievements), with recipe/achievement priors.
2. Make data collection event-aware so rare crafting and achievement transitions are not left to an unstable PPO policy.
3. Add persistent spatial/entity memory if exploration and combat remain under-sampled after the first two are fixed.

The design itself is in `docs/method-design.md`. It is deliberately not described as another incremental TWM module.

## Appendix A: full 10k curves

| Steps | WM loss | Train reward | Eval reward | Tier | Ach | Kills* | Coverage (tiles) | Action entropy | Mode | Codes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10000 | 13.3413 | -0.00510 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 9 | 138 |
| 20000 | 11.4184 | -0.00190 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 16 | 376 |
| 30000 | 9.8094 | -0.00510 | -0.00495 | 0 | 0 | 0 | 2.8750 | 0.255 | 16 | 477 |
| 40000 | 8.8543 | -0.00800 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 12 | 535 |
| 50000 | 8.0515 | -0.00440 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 12 | 598 |
| 60000 | 6.1763 | -0.00540 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.009 | 12 | 650 |
| 70000 | 5.1577 | -0.00300 | -0.00480 | 0 | 0 | 0 | 15.7500 | 0.000 | 1 | 697 |
| 80000 | 4.6147 | -0.00430 | -0.00480 | 0 | 0 | 0 | 15.7500 | 0.000 | 1 | 774 |
| 90000 | 4.0359 | -0.00430 | -0.00480 | 0 | 0 | 0 | 15.7500 | 0.000 | 1 | 789 |
| 100000 | 3.7627 | -0.00340 | +0.00990 | 0 | 4 | 0 | 18.6250 | 2.783 | 4 | 843 |
| 110000 | 3.5811 | -0.00220 | -0.00460 | 0 | 0 | 0 | 8.1250 | 0.000 | 2 | 853 |
| 120000 | 3.3413 | -0.00630 | -0.00460 | 0 | 0 | 0 | 8.1250 | 0.000 | 2 | 887 |
| 130000 | 3.1385 | -0.00420 | -0.00460 | 0 | 0 | 0 | 8.1250 | 0.009 | 2 | 916 |
| 140000 | 3.0974 | -0.00700 | -0.00475 | 0 | 0 | 0 | 9.2500 | 0.000 | 4 | 934 |
| 150000 | 2.9207 | -0.00710 | -0.00475 | 0 | 0 | 0 | 9.2500 | 0.000 | 4 | 958 |
| 160000 | 2.8169 | -0.00360 | -0.00405 | 0 | 0 | 0 | 9.7500 | 0.028 | 4 | 985 |
| 170000 | 2.7203 | +0.00210 | -0.00070 | 0 | 2 | 0 | 17.7500 | 2.620 | 12 | 1001 |
| 180000 | 2.6409 | -0.00690 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 9 | 1030 |
| 190000 | 2.4896 | -0.00710 | -0.00495 | 0 | 0 | 0 | 1.1250 | 0.123 | 9 | 1048 |
| 200000 | 2.5480 | -0.00590 | -0.00480 | 0 | 0 | 0 | 15.7500 | 0.000 | 1 | 1072 |
| 210000 | 2.5121 | -0.00500 | -0.00480 | 0 | 0 | 0 | 15.7500 | 0.000 | 1 | 1098 |
| 220000 | 2.4691 | -0.00480 | -0.00480 | 0 | 0 | 0 | 15.7500 | 0.074 | 1 | 1124 |
| 230000 | 2.5054 | -0.00620 | -0.00460 | 0 | 0 | 0 | 8.1250 | 0.000 | 2 | 1147 |
| 240000 | 2.5518 | -0.00480 | -0.00460 | 0 | 0 | 0 | 8.1250 | 0.000 | 2 | 1167 |
| 250000 | 2.3960 | -0.00430 | -0.00460 | 0 | 0 | 0 | 8.1250 | 0.000 | 2 | 1172 |
| 260000 | 2.4185 | -0.00280 | -0.00480 | 0 | 0 | 0 | 15.7500 | 0.000 | 1 | 1189 |
| 270000 | 2.4073 | -0.00430 | -0.00480 | 0 | 0 | 0 | 15.7500 | 0.000 | 1 | 1222 |
| 280000 | 2.3475 | -0.00530 | -0.00540 | 0 | 0 | 0 | 19.1250 | 0.030 | 1 | 1235 |
| 290000 | 2.3247 | -0.00440 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 8 | 1247 |
| 300000 | 2.4050 | -0.00510 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 8 | 1275 |
| 310000 | 2.3558 | +0.01180 | -0.00110 | 0 | 2 | 0 | 21.7500 | 0.720 | 4 | 1281 |
| 320000 | 2.3218 | -0.00690 | -0.00475 | 0 | 0 | 0 | 9.2500 | 0.000 | 4 | 1307 |
| 330000 | 2.2764 | -0.00320 | +0.01300 | 0 | 4 | 0 | 15.0000 | 2.734 | 4 | 1316 |
| 340000 | 2.2963 | -0.00570 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 9 | 1330 |
| 350000 | 2.2144 | -0.00610 | +0.00645 | 0 | 3 | 0 | 38.2500 | 2.472 | 4 | 1341 |
| 360000 | 2.2262 | -0.00640 | -0.00475 | 0 | 0 | 0 | 9.2500 | 0.000 | 4 | 1349 |
| 370000 | 2.2377 | -0.00680 | -0.00475 | 0 | 0 | 0 | 9.2500 | 0.004 | 4 | 1379 |
| 380000 | 2.2738 | -0.00650 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 14 | 1394 |
| 390000 | 2.2339 | -0.00350 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 14 | 1401 |
| 400000 | 2.2487 | +0.00950 | -0.00365 | 0 | 1 | 0 | 15.6250 | 0.368 | 1 | 1414 |
| 410000 | 2.2675 | -0.00270 | -0.00480 | 0 | 0 | 0 | 15.7500 | 0.000 | 1 | 1432 |
| 420000 | 2.2412 | -0.00300 | -0.00480 | 0 | 0 | 0 | 15.7500 | 0.000 | 1 | 1449 |
| 430000 | 2.2220 | -0.00590 | -0.00480 | 0 | 0 | 0 | 15.7500 | 0.000 | 1 | 1460 |
| 440000 | 2.2411 | +0.00190 | -0.00535 | 0 | 1 | 0 | 2.8750 | 0.433 | 14 | 1472 |
| 450000 | 2.2701 | -0.00500 | -0.00480 | 0 | 0 | 0 | 15.7500 | 0.000 | 1 | 1494 |
| 460000 | 2.2176 | -0.00650 | -0.00480 | 0 | 0 | 0 | 15.7500 | 0.000 | 1 | 1502 |
| 470000 | 2.2520 | -0.00420 | -0.00480 | 0 | 0 | 0 | 15.7500 | 0.000 | 1 | 1518 |
| 480000 | 2.2787 | -0.00730 | -0.00480 | 0 | 0 | 0 | 15.7500 | 0.000 | 1 | 1534 |
| 490000 | 2.2919 | +0.01570 | -0.00355 | 0 | 1 | 0 | 9.0000 | 1.839 | 14 | 1550 |
| 500000 | 2.2361 | -0.00660 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 14 | 1557 |
| 510000 | 2.2355 | -0.00780 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 14 | 1567 |
| 520000 | 2.1541 | +0.00800 | +0.01590 | 0 | 5 | 0 | 19.8750 | 2.590 | 12 | 1572 |
| 530000 | 2.1585 | -0.00410 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 12 | 1594 |
| 540000 | 2.0697 | -0.00480 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 12 | 1598 |
| 550000 | 2.0643 | +0.00060 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.241 | 9 | 1609 |
| 560000 | 2.0743 | -0.00580 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 9 | 1620 |
| 570000 | 2.0669 | -0.00500 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 9 | 1625 |
| 580000 | 1.9692 | -0.00530 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 9 | 1640 |
| 590000 | 2.0534 | -0.00490 | -0.00480 | 0 | 0 | 0 | 15.7500 | 0.000 | 1 | 1663 |
| 600000 | 2.0748 | -0.00770 | -0.00480 | 0 | 0 | 0 | 15.7500 | 0.000 | 1 | 1667 |
| 610000 | 2.0711 | -0.00630 | -0.00480 | 0 | 0 | 0 | 15.7500 | 0.000 | 1 | 1679 |
| 620000 | 2.1028 | +0.00350 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 7 | 1691 |
| 630000 | 2.1828 | -0.00370 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 7 | 1699 |
| 640000 | 2.0993 | -0.00480 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 7 | 1704 |
| 650000 | 2.0913 | -0.00510 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 15 | 1719 |
| 660000 | 2.0545 | -0.00300 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 15 | 1735 |
| 670000 | 2.0337 | -0.00290 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 15 | 1744 |
| 680000 | 1.9620 | -0.00200 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 14 | 1744 |
| 690000 | 2.0072 | -0.00540 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 14 | 1760 |
| 700000 | 1.9785 | -0.00600 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 14 | 1766 |
| 710000 | 1.9667 | +0.01740 | -0.00170 | 0 | 1 | 0 | 17.2500 | 0.473 | 1 | 1770 |
| 720000 | 1.9794 | -0.00580 | -0.00480 | 0 | 0 | 0 | 15.7500 | 0.000 | 1 | 1781 |
| 730000 | 1.9834 | -0.00170 | -0.00480 | 0 | 0 | 0 | 15.7500 | 0.000 | 1 | 1782 |
| 740000 | 1.9771 | -0.00370 | +0.00025 | 0 | 3 | 0 | 17.1250 | 0.610 | 1 | 1798 |
| 750000 | 2.0005 | +0.01740 | +0.00795 | 0 | 3 | 0 | 20.6250 | 2.553 | 2 | 1804 |
| 760000 | 2.0367 | -0.00550 | -0.00460 | 0 | 0 | 0 | 8.1250 | 0.000 | 2 | 1812 |
| 770000 | 2.0536 | -0.00360 | -0.00460 | 0 | 0 | 0 | 8.1250 | 0.000 | 2 | 1822 |
| 780000 | 2.1269 | +0.00360 | -0.00480 | 0 | 0 | 0 | 15.7500 | 0.000 | 1 | 1842 |
| 790000 | 2.1103 | -0.00440 | -0.00480 | 0 | 0 | 0 | 15.7500 | 0.000 | 1 | 1844 |
| 800000 | 2.1271 | -0.00370 | -0.00480 | 0 | 0 | 0 | 15.7500 | 0.000 | 1 | 1859 |
| 810000 | 2.1565 | -0.00300 | -0.00460 | 0 | 0 | 0 | 8.1250 | 0.000 | 2 | 1873 |
| 820000 | 2.1236 | -0.00330 | -0.00460 | 0 | 0 | 0 | 8.1250 | 0.000 | 2 | 1879 |
| 830000 | 2.0772 | -0.00220 | +0.00540 | 0 | 5 | 0 | 4.5000 | 1.661 | 0 | 1892 |
| 840000 | 2.0740 | -0.00490 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 16 | 1897 |
| 850000 | 2.0665 | -0.00030 | -0.00385 | 0 | 1 | 0 | 1.8750 | 0.250 | 0 | 1907 |
| 860000 | 2.0299 | -0.00580 | -0.00360 | 0 | 1 | 0 | 1.6250 | 0.566 | 0 | 1913 |
| 870000 | 2.0029 | -0.00620 | -0.00410 | 0 | 1 | 0 | 1.7500 | 0.634 | 0 | 1919 |
| 880000 | 1.9870 | -0.00590 | -0.00385 | 0 | 0 | 0 | 2.0000 | 0.729 | 0 | 1935 |
| 890000 | 1.9494 | -0.00540 | -0.00365 | 0 | 1 | 0 | 1.1250 | 0.013 | 0 | 1943 |
| 900000 | 1.9313 | -0.00540 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 16 | 1951 |
| 910000 | 1.9051 | -0.00440 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 16 | 1960 |
| 920000 | 1.9230 | +0.00070 | -0.00460 | 0 | 0 | 0 | 8.1250 | 0.008 | 2 | 1964 |
| 930000 | 1.9369 | -0.00830 | -0.00460 | 0 | 0 | 0 | 8.1250 | 0.000 | 2 | 1982 |
| 940000 | 1.9769 | -0.00550 | -0.00460 | 0 | 0 | 0 | 8.1250 | 0.000 | 2 | 1987 |
| 950000 | 2.0283 | +0.00920 | -0.00375 | 0 | 0 | 0 | 10.6250 | 0.011 | 3 | 2004 |
| 960000 | 2.0874 | -0.00660 | -0.00385 | 0 | 0 | 0 | 10.6250 | 0.000 | 3 | 2015 |
| 970000 | 2.0799 | -0.00370 | -0.00385 | 0 | 0 | 0 | 10.6250 | 0.000 | 3 | 2023 |
| 980000 | 2.0661 | -0.00540 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 0 | 2032 |
| 990000 | 2.1064 | -0.00510 | -0.00440 | 0 | 0 | 0 | 1.1250 | 0.026 | 0 | 2048 |
| 1000000 | 2.0474 | -0.00620 | -0.00460 | 0 | 0 | 0 | 1.0000 | 0.000 | 0 | 2061 |

`Mode` is the most frequent action index in the 2,000-action evaluation. Index 0 is NOOP; index 1 is LEFT. The mode is diagnostic only; it is not a claim that all actions in an evaluation were identical, except where action entropy is reported as zero.

## Appendix B: full 10k holdout curves

Each entry is the mean frame-state cross-entropy over 250 fixed two-frame windows in that bucket. Lower is better. These windows were collected with a random behavior policy and are not on-policy.

| Steps | Normal | Exploration | Combat | Crafting |
|---:|---:|---:|---:|---:|
| 10000 | 5.5648 | 5.6369 | 6.1976 | 6.5811 |
| 20000 | 3.2005 | 3.2795 | 3.9482 | 4.3353 |
| 30000 | 2.4072 | 2.4750 | 3.1420 | 3.4266 |
| 40000 | 1.7459 | 1.8048 | 2.3344 | 2.6488 |
| 50000 | 1.4677 | 1.5194 | 1.9971 | 2.1600 |
| 60000 | 1.3275 | 1.3670 | 1.8336 | 1.8940 |
| 70000 | 1.1894 | 1.2192 | 1.5695 | 1.7051 |
| 80000 | 1.0416 | 1.0733 | 1.3947 | 1.5383 |
| 90000 | 0.9548 | 0.9887 | 1.3189 | 1.4111 |
| 100000 | 0.8773 | 0.9124 | 1.2395 | 1.3068 |
| 110000 | 0.7891 | 0.8440 | 1.1425 | 1.2099 |
| 120000 | 0.7343 | 0.7972 | 1.0977 | 1.1410 |
| 130000 | 0.6963 | 0.7580 | 1.0512 | 1.0749 |
| 140000 | 0.6189 | 0.6734 | 0.9500 | 0.9706 |
| 150000 | 0.5760 | 0.6240 | 0.8916 | 0.9059 |
| 160000 | 0.5455 | 0.5889 | 0.8495 | 0.8587 |
| 170000 | 0.5070 | 0.5540 | 0.7862 | 0.8153 |
| 180000 | 0.5121 | 0.5581 | 0.7847 | 0.8180 |
| 190000 | 0.5275 | 0.5602 | 0.7724 | 0.7992 |
| 200000 | 0.4675 | 0.5046 | 0.7108 | 0.7491 |
| 210000 | 0.4472 | 0.4875 | 0.6864 | 0.7238 |
| 220000 | 0.4434 | 0.4819 | 0.6794 | 0.7097 |
| 230000 | 0.4400 | 0.4874 | 0.6590 | 0.7124 |
| 240000 | 0.4329 | 0.4809 | 0.6474 | 0.6986 |
| 250000 | 0.4344 | 0.4793 | 0.6447 | 0.6892 |
| 260000 | 0.4185 | 0.4653 | 0.6240 | 0.6757 |
| 270000 | 0.3941 | 0.4368 | 0.5878 | 0.6433 |
| 280000 | 0.4081 | 0.4425 | 0.5937 | 0.6359 |
| 290000 | 0.4031 | 0.4530 | 0.5887 | 0.6544 |
| 300000 | 0.4279 | 0.4728 | 0.6169 | 0.6672 |
| 310000 | 0.4403 | 0.4781 | 0.6170 | 0.6624 |
| 320000 | 0.4153 | 0.4515 | 0.5854 | 0.6319 |
| 330000 | 0.4140 | 0.4456 | 0.5764 | 0.6173 |
| 340000 | 0.4423 | 0.4764 | 0.6109 | 0.6452 |
| 350000 | 0.4523 | 0.4829 | 0.6150 | 0.6441 |
| 360000 | 0.4137 | 0.4477 | 0.5725 | 0.6078 |
| 370000 | 0.3975 | 0.4298 | 0.5587 | 0.5879 |
| 380000 | 0.3984 | 0.4539 | 0.5754 | 0.6228 |
| 390000 | 0.4357 | 0.4868 | 0.6189 | 0.6548 |
| 400000 | 0.4374 | 0.4833 | 0.6094 | 0.6447 |
| 410000 | 0.3989 | 0.4440 | 0.5595 | 0.6006 |
| 420000 | 0.3956 | 0.4347 | 0.5517 | 0.5864 |
| 430000 | 0.3959 | 0.4353 | 0.5532 | 0.5847 |
| 440000 | 0.3869 | 0.4278 | 0.5484 | 0.5772 |
| 450000 | 0.3744 | 0.4192 | 0.5294 | 0.5696 |
| 460000 | 0.3882 | 0.4349 | 0.5367 | 0.5786 |
| 470000 | 0.4041 | 0.4359 | 0.5452 | 0.5734 |
| 480000 | 0.4334 | 0.4649 | 0.5729 | 0.5957 |
| 490000 | 0.4095 | 0.4443 | 0.5530 | 0.5838 |
| 500000 | 0.4050 | 0.4395 | 0.5637 | 0.5858 |
| 510000 | 0.4403 | 0.4715 | 0.5924 | 0.6140 |
| 520000 | 0.4407 | 0.4727 | 0.5926 | 0.6139 |
| 530000 | 0.4047 | 0.4511 | 0.5762 | 0.5983 |
| 540000 | 0.4350 | 0.4719 | 0.6124 | 0.6178 |
| 550000 | 0.4496 | 0.4907 | 0.6220 | 0.6335 |
| 560000 | 0.4473 | 0.4943 | 0.6029 | 0.6347 |
| 570000 | 0.4642 | 0.4952 | 0.6328 | 0.6434 |
| 580000 | 0.4573 | 0.5097 | 0.6373 | 0.6521 |
| 590000 | 0.4398 | 0.4717 | 0.6042 | 0.6224 |
| 600000 | 0.4454 | 0.4725 | 0.6276 | 0.6294 |
| 610000 | 0.4275 | 0.4545 | 0.6114 | 0.6180 |
| 620000 | 0.3559 | 0.4010 | 0.5905 | 0.5790 |
| 630000 | 0.3915 | 0.4330 | 0.6090 | 0.5937 |
| 640000 | 0.4260 | 0.4491 | 0.6158 | 0.6014 |
| 650000 | 0.3872 | 0.4271 | 0.5996 | 0.5831 |
| 660000 | 0.4055 | 0.4403 | 0.5965 | 0.5986 |
| 670000 | 0.4381 | 0.4673 | 0.6160 | 0.6236 |
| 680000 | 0.3786 | 0.4153 | 0.5616 | 0.5828 |
| 690000 | 0.3920 | 0.4181 | 0.5662 | 0.5853 |
| 700000 | 0.4028 | 0.4262 | 0.5872 | 0.5898 |
| 710000 | 0.3940 | 0.4260 | 0.5729 | 0.5804 |
| 720000 | 0.3829 | 0.4246 | 0.5720 | 0.5848 |
| 730000 | 0.3681 | 0.4162 | 0.5921 | 0.5902 |
| 740000 | 0.3872 | 0.4321 | 0.6265 | 0.6111 |
| 750000 | 0.3730 | 0.4136 | 0.5880 | 0.5941 |
| 760000 | 0.3199 | 0.3656 | 0.5017 | 0.5183 |
| 770000 | 0.3595 | 0.3952 | 0.5485 | 0.5403 |
| 780000 | 0.3527 | 0.3768 | 0.5314 | 0.5225 |
| 790000 | 0.3287 | 0.3594 | 0.5169 | 0.5111 |
| 800000 | 0.3549 | 0.3833 | 0.5420 | 0.5301 |
| 810000 | 0.3322 | 0.3801 | 0.5232 | 0.5422 |
| 820000 | 0.3380 | 0.4180 | 0.5967 | 0.5918 |
| 830000 | 0.3354 | 0.4231 | 0.6050 | 0.5888 |
| 840000 | 0.4000 | 0.4472 | 0.6276 | 0.5942 |
| 850000 | 0.4355 | 0.4553 | 0.6329 | 0.5975 |
| 860000 | 0.4160 | 0.4424 | 0.6084 | 0.5864 |
| 870000 | 0.4078 | 0.4292 | 0.5917 | 0.5803 |
| 880000 | 0.4053 | 0.4161 | 0.5890 | 0.5684 |
| 890000 | 0.4367 | 0.4517 | 0.6177 | 0.5957 |
| 900000 | 0.3965 | 0.4130 | 0.5993 | 0.5827 |
| 910000 | 0.4023 | 0.4294 | 0.6173 | 0.5975 |
| 920000 | 0.3670 | 0.4001 | 0.5647 | 0.5628 |
| 930000 | 0.3436 | 0.3838 | 0.5524 | 0.5471 |
| 940000 | 0.3391 | 0.3825 | 0.5392 | 0.5402 |
| 950000 | 0.3277 | 0.3868 | 0.5646 | 0.5553 |
| 960000 | 0.3112 | 0.3635 | 0.5301 | 0.5247 |
| 970000 | 0.3544 | 0.4002 | 0.5711 | 0.5639 |
| 980000 | 0.3461 | 0.3624 | 0.5032 | 0.5152 |
| 990000 | 0.3820 | 0.3869 | 0.5169 | 0.5317 |
| 1000000 | 0.4040 | 0.3953 | 0.5035 | 0.5291 |

## Appendix C: key log excerpts and artifact checks

Startup and first iteration (the first rollout includes XLA compilation):

```text
env=Craftax-Classic-Pixels-v1 seed=0 max_env_steps=1000000
envs=50 rollout_steps=20 (1000 env steps/iteration)
torch=2.4.1+cu121 jax=0.6.2 devices=[CudaDevice(id=0)]
env obs=(63,63,3) actions=17
holdout: loaded cache /root/craftax-workspace/data/diagnostic_holdout.npz (1000 windows)
WM params=115.53M
phases rollout=172.35s metrics=0.01s values=0.10s wm_train=1.49s memories=0.09s imagine=0.51s learn=1.01s
phases rollout=0.46s metrics=0.01s values=0.02s wm_train=0.50s memories=0.25s imagine=3.48s learn=0.71s
```

Representative steady-state iteration:

```text
phases rollout=0.50s metrics=0.02s values=0.02s wm_train=0.49s memories=0.10s imagine=0.82s learn=0.68s
```

Final checkpoint and completion:

```text
iter  1000/1000 | steps 1000000 | wm_loss 2.0474 | reward -0.0062 | it 2.54s | elapsed 77.59 min | peak 10769.5 MB | codes 2061 | tier 0 | ach 0
--- checkpoint @ 1000000 env steps ---
eval: reward -0.0046 | tier 0 | ach 0 | kills 0 | coverage 0.000 | survival 145.2
holdout state loss: normal 0.404 (n=250) | exploration 0.395 (n=250) | combat 0.503 (n=250) | crafting 0.529 (n=250)
ckpt wall 16.6s
DONE 1000000 env steps in 77.87 min (1.30 h), peak torch 10769.5 MB
```

Only one non-fatal warning appeared in the full log:

```text
WARNING | hl_gauss_pytorch.hl_gauss: value of 1.100000023841858 is out of the support [-1.0, 1.0] ...
```

The 1.1 reward slightly exceeds the configured `[-1, 1]` support. It was not treated as blocking, but the next run should widen reward support or clip only after logging how often it occurs.

Artifact checks:

- JSON contains exactly 100 checkpoints, from 10,000 to 1,000,000 env steps.
- The final checkpoint loads and contains `wm`, `wm_opt`, `impala`, `actor`, `critic`, both optimizer states, `env_steps`, and metadata.
- Checkpoint series: 100 files, 146 GB; recurring checkpoint/evaluation wall time was ~16-20 s after first-use compilation.

## Evidence summary

- Run duration, losses, rewards, VRAM, event counts, action entropy, checkpoint count, and holdout errors: **command-verified** from the JSON/log/checkpoint files.
- Agent interface, tokenizer architecture, block-causal transformer, reward support, and absence of explicit mechanic/map state: **source-inspected** in installed TWM 0.0.59.
- Causal attribution of the crafting/combat error to the pixel tokenizer or block-causal design: **not established**. The data supports data starvation and policy instability as sufficient explanations.
- 69.66% anchor comparability: **unconfirmed**. This diagnostic is not the official training recipe and must not be presented as a reproduction.
