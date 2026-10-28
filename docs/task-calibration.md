# Craftax Task Difficulty Calibration

Date: 2026-10-28

Status: **candidate found — Case A provisionally, reward-shaped Pixels**.

The completed comparison runs used seed 0 and 200,000 environment steps, with
50 parallel environments, 20 rollout steps, and evaluation every 10,000 steps
(20 checkpoints). PPO and TWM used the same rollout shape. PPO used the
unmodified Craftax_Baselines update closure with the project evaluation
instrumentation; TWM used the repaired project harness. No third-party source
was modified.

## Results

`tier>0` counts checkpoints where the maximum tier in the evaluation batch was
greater than zero. Reward and coverage are the final-10 checkpoint means;
coverage is percent of the 64x64 map. NOOP and entropy are final-checkpoint
values. Rewards include shaping where applicable.

| Variant | Agent | Tier>0 checkpoints | Reward, last 10 | Final reward | Coverage, last 10 | Final coverage | Final NOOP | Final entropy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Pixels baseline | TWM | 9/20 | +0.02844 | +0.03110 | 0.400% | 0.391% | 1.80% | 2.319 |
| Pixels baseline | PPO | 4/20 | +0.02854 | +0.02905 | 0.286% | 0.204% | 0.75% | 1.780 |
| Pixels + first-visit shaping, aligned eval key schedule | TWM | 10/20 | **+0.05419** | **+0.06185** | **1.126%** | 1.099% | 0.30% | 1.974 |
| Pixels + first-visit shaping | PPO | 10/20 | +0.04249 | +0.04250 | 0.644% | 0.610% | 0.15% | 1.892 |
| Pixels, 500-step episodes | TWM | 10/20 | +0.02833 | +0.03010 | 0.377% | 0.342% | 4.00% | 2.390 |
| Pixels, 500-step episodes | PPO | 4/20 | +0.02854 | +0.02905 | 0.286% | 0.204% | 0.75% | 1.780 |
| Classic Symbolic | PPO | 19/20 | +0.03316 | +0.03210 | 0.587% | 0.607% | 0.30% | 1.330 |
| Classic Symbolic | TWM | UNSUPPORTED | N/A | N/A | N/A | N/A | N/A | N/A |
| Pixels curriculum | TWM | 0/20 | +0.07593 | +0.07723 | 1.273% | 1.266% | 0.90% | 1.681 |
| Pixels curriculum | PPO | 0/20 | +0.07006 | +0.07379 | 1.346% | 1.273% | 0.15% | 1.398 |

Unaligned-evaluation TWM screening row for reward shaping, retained only for
audit: 11/20 tier-positive, last-10 reward +0.05322, last-10 coverage 1.177%.
The aligned evaluation run above is the result used for the Case A decision.

### Interpretation

- **Reward shaping is the only completed paired variant that meets the strong
  separation screen.** After aligning the checkpoint evaluation key schedule,
  both agents reach tier > 0 at 10/20 checkpoints. TWM's final-10 reward mean
  is +0.01170 higher (about 27.5% relative) and its final-10 coverage is about
  1.75x PPO's. The reward and coverage signal, rather than tier count,
  distinguishes this variant.
- On the unshaped Pixels control, the final-10 rewards are effectively equal.
  Short episodes also produce nearly equal final-10 rewards. The larger tier
  count for TWM in those rows is not accompanied by a clear reward advantage.
- Craftax 1.6.1 supports `Craftax-Classic-Symbolic-v1`; PPO completed the
  200k run. The current TWM visual encoder and world model require 63x63x3
  images, while Symbolic observations are flat vectors (1,345 values), so the
  TWM run exits as unsupported without changing TWM's architecture.
- Craftax exposes `EnvParams.max_timesteps`; the short-episode wrapper set it
  to 500. The PPO evaluation horizon is 250 steps, so that evaluation itself
  cannot observe a full 500-step episode. This task variant did not create a
  reward advantage.
- The curriculum wrapper passed its JIT smoke and both agents completed 200k.
  Neither reached tier > 0. The TWM evaluation run was repeated after an
  initial process exit at 30k; holdout prediction evaluation was disabled on
  the completed rerun because it is not part of this behavioral screen.

## Protocol caveats

- Training seed is 0 for every completed run. This is a single-seed screen,
  not evidence of seed-robust separation.
- The completed reward-shaped TWM decision row uses the same evaluation seed
  (4242) and checkpoint fold-in schedule as PPO. The harnesses still derive
  action and per-step environment keys differently, so these are not
  trajectory-paired evaluations. Baseline and short-episode TWM screening
  runs used the earlier evaluation key scheme and are less directly
  comparable. No causal estimate or seed-robust result is established.
- TWM and PPO retain their different policy/world-model implementations and
  compute costs. Training environment count, rollout horizon, total steps, and
  checkpoint spacing were matched; the runs were not an official TWM paper
  reproduction.
- TWM baseline completed in 30.78 min; PPO baseline in 10.58 min. TWM
  reward-shaped completed in 18.98 min for the aligned evaluation run; PPO
  reward-shaped in 10.65 min. TWM
  short-episode completed in 17.63 min; PPO short-episode in 12.60 min. The
  GPU was shared with unrelated work during parts of the session, so these
  are observed wall times, not isolated throughput benchmarks.
- Some TWM attempts were partial and then rerun: the reward-shaping wrapper
  integration attempt stopped at 10k, and the first curriculum attempt
  stopped after 30k. The complete result rows each use exactly 200k steps,
  but the cumulative TWM environment steps exceeded the requested per-variant
  limit: reward shaping used 10k failed + 200k initial + 200k aligned-eval
  steps (410k total), and curriculum used 30k partial + 200k completed steps
  (230k total). This is a protocol deviation and is disclosed rather than
  counted as additional completed comparisons.
- The PPO Symbolic JSON's top-level `env` field says Pixels because of an
  earlier reporting bug. Its recorded `config.ENV_NAME` is
  `Craftax-Classic-Symbolic-v1`, and the run used the symbolic PPO network.

## Decision

**Case A, provisional:** reward-shaped Pixels is a candidate where TWM and PPO
separate under the measured screen. This does not reverse the original 1M
unshaped Pixels parity result or establish general TWM sample-efficiency.

Next, run the full 1M diagnosis on the exact reward-shaped task with the
aligned evaluation key schedule, reporting raw Craftax reward separately from
the +0.1 first-visit shaping bonus. The current result is single-seed and
does not justify a general TWM advantage claim; replicate before making that
claim or implementing CMS-WM.

## Artifacts

Logs and JSON are under `~/craftax-workspace/logs/calibration/`:

- `twm_baseline.{log,json}`, `ppo_baseline.{log,json}`
- `twm_reward_shaping.{log,json}`, `ppo_reward_shaping.{log,json}`
- `twm_short_episode.{log,json}`, `ppo_short_episode.{log,json}`
- `ppo_symbolic.{log,json}`; TWM unsupported message is in
  `twm_symbolic_unsupported.log`
- `twm_curriculum.log`, partial `twm_curriculum.json`, and completed
  `twm_curriculum_rerun.{log,json}`
- `twm_reward_shaping_eval_aligned.{log,json}`
- `calibration_env_smoke.log`

The project-local wrappers and harness options are in `scripts/calibration_env.py`,
`scripts/ppo_baseline_sanity.py`, and `scripts/twm_baseline_diagnostic.py`.
