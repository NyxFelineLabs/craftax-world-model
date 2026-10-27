# Data-Driven World-Model Design for Craftax

Date: 2026-10-27

## Status: pending sanity check

**Updated after the pure PPO control run in `docs/ppo-sanity-results.md`.**
The material below is preserved as the pre-sanity design draft, not as an
implementation-ready method.  Its central empirical premise did not survive
the control experiment.

- Pure PPO did **not** reproduce the TWM harness's NOOP/entropy collapse:
  final NOOP was 1.2%, policy entropy 2.019 nats, and reward +0.03095/step.
- Pure PPO still failed broad exploration and recipe progression: coverage
  0.311%, final tier 0, and no tool in inventory.
- The Step 3 result is therefore **Case C (partial)**.
- The earlier 0.091% crafting-event rate was measured under a collapsed TWM
  policy.  Pure PPO produced inventory-changing transitions at 6.38% in its
  final evaluation, so "rare-event data starvation" is not a general property
  of Craftax or the harness.

What must be re-argued before CMS-WM can be implemented:

1. Separate resource pickup from successful recipe completion.  Wood/sapling
   changes were common; a completed tool recipe remained absent.
2. Justify a world-model intervention against both a pure PPO control and a
   repaired TWM control, not against the collapsed diagnostic policy.
3. Treat the tokenizer/transformer attribution as unconfirmed.  A pure PPO
   policy reached tier 0 without any world model.
4. Define a recipe-level success metric and an offline prediction gate before
   proposing closed-loop benefit.
5. Demonstrate that the remaining failure is a WM representation or planning
   problem rather than sparse reward, credit assignment, or a 1M-step budget.

Do not implement CMS-WM on the basis of the earlier diagnostic alone.

Input evidence: `docs/diagnostic-results.md`, `docs/ppo-sanity-results.md`, and
the two 1M-step JSON artifacts in `~/craftax-workspace/logs/`.

This document proposes a **new domain-specific world model**, not an incremental loss or tokenizer tweak on top of the ICML TWM. The diagnosis first identifies which failures are real, then each proposed component is tied to one of those measurements.

## 0. What the TWM diagnostic observed (superseded as a method premise)

### Established for that run

1. **The WM optimizes its training loss.** Training loss fell from 13.3413 to 2.0474, and held-out normal-frame error fell from 5.5648 to 0.4040 (-92.7%).
2. **A stable mechanic gradient remains.** At the final checkpoint, crafting error was 0.5291 versus 0.4040 normal (1.31x); combat was 0.5035 (1.25x). Over the last ten checkpoints, craft/normal and combat/normal were both 1.54x.
3. **Closed-loop behavior is unstable and usually degenerate.** Final evaluation used NOOP for all 2,000 actions. Action entropy was below 0.01 at 74/100 checkpoints and above 1.0 at only 8/100. Deterministic modes changed repeatedly, and coverage peaked at 38.25/4,096 tiles (0.93%) before falling back to one tile at the end.
4. **Crafting experience was almost absent under the collapsed TWM policy.** A 100,000-transition training audit contained 91 crafting events (0.091%), versus 3,925 exploration (3.93%) and 18,307 combat-labelled (18.31%) transitions. Final inventory was all zero; the maximum inventory ever observed across checkpoint evaluations was wood=2 and sapling=3, with no tool of any kind. This rate did not replicate under pure PPO.
5. **Mechanic progress never starts.** Crafting tier was 0 at every checkpoint. Achievements reached 5 at one checkpoint but ended at 0. No kill achievement was ever obtained.

### Not established

- The diagnostic does **not** show that the pixel tokenizer or block-causal transformer causes the crafting/combat gap. The same gap is fully consistent with the measured absence of those transitions.
- It does **not** establish that the policy collapse is intrinsic to TWM's architecture. It may be a property of this simplified schedule, the default entropy coefficient, or the short 1,000-update policy budget.
- It does **not** reproduce the paper's 69.66% score. The harness uses a different policy loop, WM update rate, Dyna ratio, and evaluation. The paper score is an external anchor, not a baseline measured here.

The design below therefore has two jobs: (a) make sparse mechanic transitions representable, and (b) make them observable often enough to learn. A model-only change without the second job would be another untested incremental idea.

## 1. Design principles

1. **Factor the latent by game semantics, not just by image patches.** Inventory counts, tool tier, achievements, vitals, entities, and map occupancy have different transition rules and should not be forced through one undifferentiated patch-code stream.
2. **Use the simulator's mechanics as structure, not as test-time input.** Craftax recipes form a directed acyclic prerequisite graph; achievements are monotone; inventory counts are non-negative integers; the map has 4-neighbour topology. These priors can constrain predictions while the deployed model still receives only pixels and actions.
3. **Treat rare events as first-class training targets.** In the collapsed TWM run, crafting was 0.091% of the training stream; the pure PPO control produced 6.38% inventory-changing evaluation transitions, so this is not a general Craftax data rate. The remaining rare event is successful recipe completion, which still needs event-balanced measurement and replay.
4. **Separate model evaluation from policy quality.** The next experiment must first demonstrate that mechanic transitions can be predicted on a fixed, mechanic-balanced dataset. Only then should closed-loop reward be used to judge the model.
5. **Keep a baseline-compatible pixel path.** The visual encoder can reuse the TWM patch tokenizer or a learned discrete encoder. This preserves a controlled comparison and makes each new component ablatable.

## 2. Candidate A (pre-sanity draft, on hold): Craftax Mechanic-State World Model (CMS-WM)

### Failure mode addressed

Crafting had the highest final held-out error (0.5291, 1.31x normal; 1.54x over the last ten checkpoints), yet the behavior policy produced only 91 crafting events in a 100,000-transition audit and never left tier 0. That co-occurrence was real in the TWM run but did not generalize: pure PPO produced many more inventory-changing events yet still finished at tier 0. The remaining representation hypothesis is therefore unproven.

### Game prior

- Recipes are a known directed acyclic graph over inventory and tools.
- Inventory quantities are small non-negative integers.
- Achievements and tech unlocks are monotone: once true, they do not become false.
- Most actions leave most state unchanged; the useful target is a **sparse delta**, not a full re-render of the HUD.

### Architecture

```text
 pixels o_t (63x63x3)
        |
        +--> visual patch encoder --> V_t (spatial visual tokens)
        |
        +--> mechanic posterior q(m_t | o_t)  [pixels only at inference]
                 |-- inventory counts (12 slots, integer)
                 |-- tool/tech tier and recipe frontier
                 |-- achievement bits (22, monotone)
                 |-- vitals (health/food/drink/energy)
                 |-- local entity slots (relative position, type, health/mask)
                 `-- local map patch (occupancy/visited/frontier)
                              |
 action a_t ------------------+--> sparse event head
                                      |
                                      v
                           delta-m_t = (delta_inv, unlock bits,
                                       damage/terminal events)
                                      |
                           recipe-DAG / monotone mask
                                      |
                                      v
                           predicted m_{t+1}
                                      |
              V_t, m_t, a_t --> mechanic-conditioned transition transformer
                                      |
                     +----------------+----------------+
                     v                                 v
              visual next-frame head             structured next-state heads
```

The visual patch path is intentionally retained. The change is that the latent state is **factorized into visual and mechanic tokens**. A cross-attention block lets visual evidence update mechanic slots and lets mechanic predictions condition future visual prediction.

### Training signal and privileged labels

During training only, `EnvState` supplies labels for the mechanic posterior and the delta head. At inference, those labels are unavailable; the model must infer the slots from pixels. This is a standard privileged-training / pixel-only-deployment setup. To check leakage, an oracle variant may feed true `EnvState` for an upper-bound ablation, but that variant must never be the headline result.

### Loss

```text
L = L_visual_next_frame
  + lambda_1 * L_inventory_count
  + lambda_2 * L_achievement_bits
  + lambda_3 * L_sparse_event_delta
  + lambda_4 * L_recipe_DAG_violation
  + lambda_5 * L_monotone_violation
  + lambda_6 * L_entity_damage_or_terminal
```

The recipe-DAG mask is applied to the predicted delta before it is fed back into the next state. Invalid transitions are masked, not merely penalized: this is the main reason the design is domain-specific rather than a generic auxiliary loss.

### Event-balanced acquisition

An auxiliary **recipe-frontier sampler** proposes goals near the agent's current unlocked recipe frontier. Rollouts that change an inventory slot, unlock an achievement, or cross a recipe prerequisite are stored in a stratified replay buffer. Pixel-batch sampling is uniform only within a bucket; event buckets are up-weighted so crafting is not 0.09% of every optimization batch.

This is part of the WM training system, not a claim that the WM alone can fix a collapsed policy. The sampler may use scripted exploration during data collection; evaluation remains pixel-only and action-only.

### Relation to TWM

This is **not** a TWM head added to the existing loss. It replaces the single patch-code latent with a factorized visual/mechanic state, adds a recipe-constrained transition rule, and changes the training distribution. The TWM transformer and tokenizer can be reused as components and as the control baseline, but the system boundary and state representation are different.

### Expected effect and evaluation

- Held-out crafting state error: target below 1.2x normal (current 1.54x over the last ten checkpoints).
- Exact inventory-delta accuracy and event precision/recall on a fixed mechanic-balanced holdout.
- Recipe-violation rate on imagined rollouts.
- Closed-loop: first tool by 100k steps, tier >= 1 by 200k, and non-zero achievement progress sustained across checkpoints.
- Ablations: remove recipe mask, remove monotone mask, remove event replay, and replace mechanic slots with a pure visual token baseline.

### Implementation estimate

Engineering estimate, **not measured**: 600-1,000 lines, +10-20 M parameters, +1-2 GB VRAM, and roughly 2-3 hours for a 1M-step run at the current 12.5 GB / 77.9 min baseline scale. The first build should use a fixed offline dataset before any closed-loop run.

## 3. Candidate B: Topographic Frontier Memory (TFM)

### Failure mode addressed

Exploration coverage peaked at only 38.25/4,096 tiles (0.93%) and was one tile at the final checkpoint. The final policy was NOOP. The WM's exploration prediction error was not the worst bucket (0.3953 vs 0.4040 normal), so this is primarily a **behavior and persistent-state** failure, not evidence that local frame prediction is broken.

### Game prior

Craftax's world is a 64x64 grid with local movement, walls, and a visited/frontier structure. A persistent allocentric map is a much better prior for exploration than asking a finite block of patch tokens to remember every visited tile.

### Architecture

```text
 o_t --> local visual tokens --> local map writer --> M_t (64x64 occupancy/visited/frontier)
                                      ^                     |
                                      |                     v
                                  action a_t          frontier reader
                                                             |
                                                   goal/conditioning vector
                                                             |
                                        transition transformer + policy conditioning
```

`M_t` is written only in a local neighbourhood around the player. The map head predicts occupancy and frontier changes; it is not a rendered image. A frontier-conditioned goal vector can be consumed by the policy without exposing privileged state at evaluation.

### Relation to TWM

TFM adds a persistent spatial memory outside the patch-token sequence. It is compatible with CMS-WM but is a separate module and a separate ablation. It does not attempt to replace the visual dynamics model.

### Expected effect and evaluation

- New-tile rate per 1,000 steps and frontier efficiency (new tiles per movement action).
- Held-out map transition accuracy and long-horizon coverage.
- Closed-loop coverage target: at least 2x the baseline's best checkpoint coverage under a matched budget.
- Ablation: persistent map versus an equal-parameter recurrent token memory.

### Implementation estimate

Engineering estimate, **not measured**: 200-350 lines, +3-6 M parameters, +0.3-0.8 GB VRAM, and roughly 10-15% additional training time.

## 4. Candidate C: Entity-Relational Combat Model (ERCM)

### Failure mode addressed

Combat was the second-highest final error bucket (0.5035, 1.25x normal) and the agent earned zero combat achievements. The audit saw 18.31% combat-labelled transitions, but these are mostly proximity labels; there were no kill events. The model has many frames with nearby mobs but no successful combat outcome to learn from.

### Game prior

Combat is local and entity-centric: player and mobs have positions, health, masks, and damage events. A relational slot model is a better prior than treating each 7x7 patch independently.

### Architecture

```text
 o_t --> entity slot extractor (player, zombies, skeletons, cows)
                 |
                 v
       relational attention over (relative position, health, mask)
                 |
                 +--> damage / death / kill event heads
                 +--> next-entity-state and terminal prediction
```

Entity slots are supervised during training from `EnvState`, inferred from pixels at deployment. Kill events are trained with class-balanced replay. The head is evaluated on event precision/recall, not only pixel reconstruction.

### Relation to TWM

ERCM replaces generic patch tokens for the combat-relevant part of the scene with entity slots and relational attention. It is a separate candidate and a later ablation; it should not be merged into CMS-WM before CMS-WM has an offline result.

### Expected effect and evaluation

- Combat-bucket state error below 1.2x normal.
- Damage/death/kill event F1 and calibration.
- First kill achievement by 500k steps; sustained combat progress after that.
- Ablation: entity slots versus equal-parameter patch tokens.

### Implementation estimate

Engineering estimate, **not measured**: 300-500 lines, +5-8 M parameters, +0.5-1 GB VRAM, and roughly 15% additional training time.

## 5. Priority and recommendation (provisional)

| Rank | Candidate | Why this rank | Main risk |
|---:|---|---|---|
| 1 | CMS-WM | Directly targets the highest and most persistent error bucket; recipe DAG gives a strong domain prior | Its starvation premise was weakened by pure PPO; needs a new recipe-level justification |
| 2 | TFM | Final coverage was one tile; persistent topology is a clean, separable fix | Map memory cannot by itself stop a NOOP policy |
| 3 | ERCM | Combat error is second and no kill outcome exists | Combat has even less outcome data; could be premature |

**Recommendation on hold: do not build CMS-WM yet.** The pure PPO sanity
control invalidated the earlier claim that the failure was primarily a
TWM-specific rare-event problem.  First debug the TWM integration's policy
collapse, quantify successful recipe transitions rather than raw inventory
changes, and establish a pure PPO baseline that either explores adequately or
shows a reproducible sample-budget limit.  Only after those controls may
CMS-WM be re-argued as an offline prediction experiment.

The earlier first gate (at least 1% labelled crafting events) was satisfied by
pure PPO at 6.38%, yet recipe progression still failed.  That gate is therefore
insufficient and must be replaced by a cookbook-recipe success gate.

## 6. Why this fits IEEE Transactions on Games

The contribution is not "a slightly better generic world model". It is a game-mechanic prior turned into an architectural constraint: recipe DAGs, monotone achievements, integer inventory, persistent map topology, and entity relations. That is the kind of domain insight ToG values, and it yields metrics that game designers and RL researchers can both read: tier progress, achievement unlocks, frontier coverage, event accuracy, and return.

The evaluation should still report the paper's 1M-step, Craftax-Classic reward anchor, but only after the official TWM protocol is reproduced or the comparison is explicitly labelled as a different harness. The diagnostic baseline here is not 69.66%, so the paper must not claim a direct win over that number. A defensible protocol is: same observation/action interface, same 1M-step budget, at least 3 seeds for the final comparison, per-bucket prediction curves, mechanic progress curves, and an official-protocol baseline alongside the diagnostic baseline.

The pre-sanity scientific story was that a pixel world model could fit its
loss while the agent failed to generate the rare events that matter.  The pure
PPO control weakens that story: a policy without any world model generated far
more inventory-changing events and still failed recipe progression.  CMS-WM,
TFM, and ERCM remain possible game-specific hypotheses, but none is currently
supported as a method contribution without the new controls described above.

## 7. Evidence and scope labels

- Data quoted in Sections 0-1: **command-verified** from the diagnostic JSON.
- Architecture descriptions and module boundaries: **design proposal**, not implemented or measured.
- Parameter/VRAM/time costs: **engineering estimates**, explicitly not measured.
- Causal attribution to TWM's tokenizer or transformer: **not claimed**; the diagnostic supports only the observed association and the measured data-starvation mechanism.
- Paper-anchor comparability: **unconfirmed**; the current harness differs from the official recipe.
