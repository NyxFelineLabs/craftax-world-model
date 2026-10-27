# Integration Smoke Results (1000 steps)

Date: 2026-10-27
Host: WSL2 Ubuntu-22.04, RTX 3090 (24576 MiB)
Project: `/mnt/e/Project/NyxFelineLabs/craftax-world-model`
Script: `scripts/integration_smoke.py` (sha256 `e2364824e80a...`, 458 lines)
Log: `~/craftax-workspace/logs/integration_smoke.log`
JSON: `~/craftax-workspace/logs/integration_smoke.json`

Scope: mechanism + measured cost only. No convergence was expected or evaluated.

## Tested configuration

| Item | Value |
|---|---|
| Environment | `Craftax-Classic-Pixels-v1`, obs `(63,63,3)`, 17 actions |
| Env batch | 8 |
| Env frames per iteration | 32 (256 environment transitions) |
| World-model sequence length | 8 frames |
| TWM | `dim=512, depth=4, block_size=81`, tokenizer `dim=147, distance_threshold=0.5` |
| TWM params | 115.53 M |
| Imagined rollout | K = 8 frames |
| PPO fallback | 8 envs x 32 steps = 256 transitions, 4 epochs, 8 minibatches, 1 update |
| Warmup (excluded from timing) | 1 full iteration, 248.04 s |
| Measured iterations | 1000 |

The warmup covers JAX/XLA compilation, cuDNN autotune, and the first tokenizer
bootstrap. It is a real startup cost but is excluded from the per-step table.

## 3a. Measured numbers

| Phase | Total (s) | Per-step (ms) | % of total |
|---|---:|---:|---:|
| env sample | 252.866 | 252.866 | 13.04% |
| jax->torch convert | 35.695 | 35.695 | 1.84% |
| TWM train step | 661.433 | 661.433 | 34.12% |
| TWM imagine | 407.379 | 407.379 | 21.01% |
| torch->jax convert | 1.496 | 1.496 | 0.08% |
| PPO update | 569.764 | 569.764 | 29.39% |
| **TOTAL** | **1938.774** | **1938.774** | **100.00%** |

Note: with `--steps 1000`, per-step milliseconds are numerically equal to total
seconds (`1000 * total_s / steps = total_s`). This is arithmetic, not a copied
column.

- **1000-step wall time: 1938.77 s = 32.31 min** (measured)
- **Average per-step: 1938.77 ms** (measured)
- **Last 100 steps (steady state): 114.07 s = 1140.7 ms/step** (measured;
  `elapsed_1000 - elapsed_900`)
- **Peak VRAM (torch allocator): 10532.9 MB** (measured)
- **Device memory in use at the end (nvidia-smi): 12404.0 MB** (measured)
- **Tokenizer codebook: 89 codes at the end** (38 after warmup), i.e. it
  saturates early and does not grow without bound (measured)
- **TWM loss: 14.006 (first) -> 0.922 (last), no non-finite values** (measured)
- **JAX<->torch bridge: 1.84% + 0.08% = 1.92% of total** (measured)

Per-100-step progress (measured, verbatim):

```text
step   100/1000 | elapsed   256.61s | per-step 2566.06 ms | peak 10532.9 MB | codes 79
step   200/1000 | elapsed   519.59s | per-step 2597.96 ms | peak 10532.9 MB | codes 83
step   300/1000 | elapsed   745.87s | per-step 2486.22 ms | peak 10532.9 MB | codes 85
step   400/1000 | elapsed   970.09s | per-step 2425.24 ms | peak 10532.9 MB | codes 85
step   500/1000 | elapsed  1188.03s | per-step 2376.07 ms | peak 10532.9 MB | codes 86
step   600/1000 | elapsed  1406.74s | per-step 2344.56 ms | peak 10532.9 MB | codes 87
step   700/1000 | elapsed  1568.04s | per-step 2240.05 ms | peak 10532.9 MB | codes 87
step   800/1000 | elapsed  1683.76s | per-step 2104.70 ms | peak 10532.9 MB | codes 88
step   900/1000 | elapsed  1824.70s | per-step 2027.44 ms | peak 10532.9 MB | codes 89
step  1000/1000 | elapsed  1938.77s | per-step 1938.77 ms | peak 10532.9 MB | codes 89
```

The first ~300 iterations are slower because the nearest-neighbour tokenizer
adds new codes in a Python loop (`add_codes_`). Once the codebook stabilizes the
per-step cost drops from ~2.6 s to ~1.14 s. This is measured, not inferred.

### Environment throughput

256 real environment transitions per iteration:

- 1938.77 ms / 256 = **7.57 ms per environment step** (full-run average)
- 1140.7 ms / 256 = **4.46 ms per environment step** (steady state)
- ~132 env steps/s (full average), ~224 env steps/s (steady state)

## 3b. Projected cost

### Unit ambiguity (important)

"100k steps" is ambiguous. Measured extrapolations for three interpretations:

| Budget | Loop iterations | Full-run average (1.939 s/iter) | Steady state (1.141 s/iter) |
|---|---:|---:|---:|
| 100k loop iterations | 100,000 | **53.9 h** | **31.7 h** |
| 100k environment steps | 391 | **12.6 min** | **7.4 min** |
| 1M environment steps (paper anchor) | 3,906 | **2.10 h** | **1.24 h** |

The paper's headline anchor (Craftax-classic, 69.66%) is at **1M environment
steps**, not 100k loop iterations. At the measured throughput, a paper-comparable
run costs ~1.2-2.1 h per seed.

### Memory

- torch allocator peak: 10.53 GB. Device used at end: 12.40 GB.
- **Risk:** a short 5-step debug run of the same script reported
  `nvidia-smi` used = 23.81 GB at teardown, i.e. transient allocations can
  approach the 24.58 GB device limit. The 1000-step run did not OOM, but there
  is little headroom for a larger batch or longer sequences. (measured, single
  observation)
- The 115.5 M-parameter world model uses `num_residual_streams=4` (library
  default), which multiplies activation memory. This is the first knob to turn.

### Checkpointing

At 1.2-2.1 h per 1M-env-step run, checkpointing is optional but still advisable:
save every ~10-15 min (roughly every 500-800 iterations). A single 100k-loop
iteration run (31.7-53.9 h) would absolutely require checkpoint/resume.

## 3c. Feasibility verdict

**Case C - not feasible as designed**, by the stated thresholds: 1000 steps took
32.31 min, above the 20-minute cutoff.

Nuance: this is a *mild* Case C. No single phase exceeds 50%, and the cross-
framework bridge is only 1.92% of the total. The time is spread across real
compute:

| Root cause | Share | Evidence |
|---|---:|---|
| TWM train step (115.5 M params, seq len 8, 4 residual streams) | 34.1% | measured |
| JAX PPO update (re-rollout + 4 epochs x 8 minibatches) | 29.4% | measured |
| TWM imagined rollout (8 autoregressive steps) | 21.0% | measured |
| Craftax env sampling (256 transitions) | 13.0% | measured |
| JAX<->torch bridge | 1.9% | measured |

Two structural findings matter more than the raw seconds:

1. **The bridge is not the problem.** Rewriting TWM in JAX is not justified by
   conversion overhead (1.9%). The cost is model compute.
2. **The PPO fallback is architecturally wrong for Dyna.** `ppo.make_train`
   rolls out its own environment and cannot consume imagined transitions
   (source-inspected; no injection API). The 29.4% PPO phase therefore does not
   advance the Dyna loop at all. The TWM package already ships the correct
   learner: `Agent.learn(memories)` accepts a list, so real and imagined
   memories can be trained together in PyTorch.

### Optimization path (inferred, not measured)

| Change | Expected effect | Confidence |
|---|---|---|
| Replace JAX PPO with TWM `Agent.interact_with_env` + `Agent.forward` + `Agent.learn([real, imagined])` | removes ~29% and makes imagined data useful | high |
| Train the world model every 2nd iteration | removes ~17% | high |
| `num_residual_streams=1`, `dim=256`, `wm_time=4` | roughly halves WM train + imagine | medium |
| Reduce imagined rollout K=8 -> 4 | ~10% | high |
| Larger env batch (32-64) | improves env throughput and PPO sample use | medium, memory-limited |
| bf16/autocast for TWM | additional speed and memory | medium, needs numerical check |

If all of the above land, a per-iteration cost around 0.5-0.7 s is plausible
(inferred), which would make even 100k loop iterations a ~14-19 h job. This was
not measured and must be re-measured before committing to it.

## 3d. Impact on Direction 1

- **100k loop iterations in 24 h?** No. Measured 53.9 h (full average) or 31.7 h
  (steady state) per seed.
- **3-seed x multi-config matrix?** No. 3 seeds x 1 config = 161.6 h (6.7 days)
  at the full-run average; a 3x3 matrix would be ~20 days.
- **What is realistic?** Count environment steps, not loop iterations. A
  paper-comparable 1M-env-step run is 1.2-2.1 h per seed:
  - 1 seed x 3 configs: ~3.7-6.3 h
  - 3 seeds x 2 configs: ~7.4-12.6 h
  - 3 seeds x 3 configs: ~11.2-18.9 h
  All fit inside 24 h at the measured rate, before any optimization.
- **Minimal publishable experiment:** 3 seeds x 2 configs (baseline vs Dyna)
  at 1M environment steps each, ~7-13 h total, plus evaluation. If time is
  tighter, 1 seed x 3 configs at 1M steps (~4-6 h) is the smallest defensible
  pilot.

## Required pilot components discovered by this smoke

1. **Upstream tokenizer bug (component defect).** `improving-transformers-world-model`
   0.0.59 crashes with:

   ```text
   RuntimeError: masked_fill_ only supports a 0-dimensional value tensor,
   but got tensor with 1 dimension(s).
   ```

   at `world_model.py:280` in `NearestNeighborTokenizer.forward`, whenever two
   or more new patches are discovered in one batch. The smoke uses a local
   subclass `StableNearestNeighborTokenizer` that reproduces the upstream logic
   and replaces only that assignment. Third-party files are untouched. The pilot
   needs the same workaround (or a pinned upstream fix).
2. **Reward support is too narrow.** Craftax emits rewards below zero
   (observed `-0.2`), outside the README's `reward_min_value=0.0,
   reward_max_value=1.0`. One HL-Gauss saturation warning was emitted. The pilot
   should widen the support (e.g. `reward_min_value=-1.0`) or fit it to the data.
3. **Craftax -> torch adapter** (`reset()` / `forward(actions)`) returning torch
   tensors, HWC -> CHW.
4. **World-model optimizer/training step** (the package has none).
5. **Dyna-with-warmup schedule** and real+imagined memory plumbing.
6. **Evaluation harness** (episode return over 1M env steps).

## Appendix: key log excerpts

Warmup and setup:

```text
env obs=(63, 63, 3) num_actions=17
WorldModel params=115.53M
PPO external injection: UNSUPPORTED (make_train has no external-transition API); using real-env update
warmup wall time: 248.04s (tokenizer codes=38)
```

Final table and totals:

```text
Phase                  |  Total (s) | Per-step (ms) | % of total
----------------------------------------------------------------
env sample             |    252.866 |       252.866 |     13.04%
jax->torch convert     |     35.695 |        35.695 |      1.84%
TWM train step         |    661.433 |       661.433 |     34.12%
TWM imagine            |    407.379 |       407.379 |     21.01%
torch->jax convert     |      1.496 |         1.496 |      0.08%
PPO update             |    569.764 |       569.764 |     29.39%
----------------------------------------------------------------
TOTAL                  |   1938.774 |      1938.774 |    100.00%

Peak VRAM (torch allocator): 10532.9 MB
Peak VRAM (nvidia-smi used): 12404.0 MB
Final TWM loss: 0.9218 (first 14.0059, non-finite 0)
Tokenizer codes at end: 89
100k extrapolation: 53.85 h
```

Reward-range warning (one occurrence):

```text
WARNING | hl_gauss_pytorch.hl_gauss - value of -0.20000000298023224 is out of
the support [0.0, 1.0] ... consider widening the support
```

## Evidence labels

- Phase timings, VRAM, loss, tokenizer counts, throughput: **verified by command**
  (script output + JSON).
- PPO injection impossibility: **source-inspected** in Phase A + re-confirmed
  here (`make_train` has no external-transition parameter).
- Optimization gains, per-transition projections, unit reinterpretation:
  **inferred** from measured numbers, not measured.
- Official-paper code comparison: still **unconfirmed** (Phase A).