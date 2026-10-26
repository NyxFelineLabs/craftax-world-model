# Pilot Readiness Check (Phase A)

Date: 2026-10-27
Host: WSL2 Ubuntu-22.04 on Windows, RTX 3090 (24576 MiB)
Project: `/mnt/e/Project/NyxFelineLabs/craftax-world-model`

Scope of this document: read-only inspection plus minimal non-training verification, run before the 100k-step pilot. **No training was executed** (no optimizer step, no micro-pilot).

Evidence labels used below:

- **verified by command** - a command produced the stated output (exit code recorded).
- **source-inspected** - read from the installed third-party source, without executing it.
- **inferred** - a conclusion drawn from verified facts, not directly executed.
- **unconfirmed** - could not be established from commands or files in this environment.

## A0. Environment confirmation

| Check | Command | Exit | Result |
|---|---|---:|---|
| Interpreter | `which python; python --version` | 0 | `/root/miniconda3/envs/craftax-wm/bin/python`, Python 3.10.22 |
| torch / JAX | `python -c "import torch, jax; ..."` | 0 | `torch 2.4.1+cu121 cuda True`; `jax 0.6.2 [CudaDevice(id=0)]` |
| Package versions | `pip show improving-transformers-world-model craftax` | 0 | TWM `0.0.59`, Craftax `1.6.1` |

All five values match the previous stage report (torch 2.4.1+cu121, JAX 0.6.2, Craftax 1.6.1, TWM 0.0.59), so the checks continued. **verified by command**

## A1. Craftax observation space

### A1.1 Observation shapes

The four environment names requested in the check, plus one extra name that was measured to resolve the 63x63 question:

| Environment | Exit | obs shape | obs dtype | num actions |
|---|---:|---|---|---:|
| `Craftax-Classic-v1` | 1 | - | - | `ValueError: Unknown craftax environment: Craftax-Classic-v1` |
| `Craftax-Symbolic-v1` | 0 | `(8268,)` | float32 | 43 |
| `Craftax-Pixels-v1` | 0 | `(130, 110, 3)` | float32 | 43 |
| `Craftax-Classic-Symbolic-v1` | 0 | `(1345,)` | float32 | 17 |
| `Craftax-Classic-Pixels-v1` (added) | 0 | `(63, 63, 3)` | float32 | 17 |

**verified by command.** `Craftax-Classic-v1` is not a valid name in Craftax 1.6.1; the supported names (registry scan of the installed package source) are:

```text
Craftax-Classic-Pixels-v1          Craftax-Classic-Symbolic-v1
Craftax-Classic-Pixels-AutoReset-v1   Craftax-Classic-Symbolic-AutoReset-v1
Craftax-Classic-Pixels-NoAutoReset-v1 Craftax-Classic-Symbolic-NoAutoReset-v1
Craftax-Pixels-v1                  Craftax-Symbolic-v1
Craftax-Pixels-AutoReset-v1        Craftax-Symbolic-AutoReset-v1
Craftax-Pixels-NoAutoReset-v1      Craftax-Symbolic-NoAutoReset-v1
```

### A1.2 Why the pixel sizes differ

**source-inspected** (`craftax/craftax/constants.py`, `craftax/craftax_classic/constants.py`, observation-space definitions, `renderer.py`):

| Constant | Craftax (full) | Craftax-classic |
|---|---:|---:|
| `BLOCK_PIXEL_SIZE_AGENT` | 10 | 7 |
| `OBS_DIM` (tiles) | (9, 11) | (7, 9) |
| default `map_size` | (48, 48) | (64, 64) |
| observation-space formula | `(OBS_DIM[0] + INVENTORY_OBS_HEIGHT) * block, OBS_DIM[1] * block, 3` | same formula |
| resulting obs | `(130, 110, 3)` | `(63, 63, 3)` |

The classic pixel renderer uses 7x7-pixel tiles over a 9x9 visible window (`(7 + 2) * 7 = 63`), which is exactly the 63x63 image the TWM README describes.

### A1.3 A5.1 conclusions (observation space)

- `Craftax-Classic` pixel observation: `(63, 63, 3)` float32 in `[0, 1]` (`get_obs` divides the uint8 render by 255). **verified by command**
- `Craftax-Symbolic-v1`: `(8268,)` float32 vector (191 map features + inventory), 43 actions. **verified by command**
- `Craftax-Classic-Symbolic-v1`: `(1345,)` float32 vector, 17 actions. **verified by command**
- Which environment does `image_size=63, channels=3` match? **`Craftax-Classic-Pixels-v1`**, exactly and only. It does not match `Craftax-Pixels-v1` (`(130, 110, 3)`). **verified by command**
- Layout/format mismatch: Craftax returns `(H, W, C)` images; TWM's `forward` declares `Float['b c t h w']`. An adapter must transpose HWC -> CHW and add batch/time axes. **verified by signature + measured shape**
- Which environment does the ICML 2025 69.66% anchor belong to? The arXiv abstract states: "We mostly focus on the challenging Craftax-classic benchmark, where our method achieves a reward of 69.66% after only 1M environment steps, significantly outperforming DreamerV3, which achieves 53.2%, and exceeding human performance of 65.0%". So the anchor is **Craftax-classic**. **verified by command** (arXiv abs page, `https://arxiv.org/abs/2502.01591v3`)
- Rendering used for that anchor: **pixels (63x63x3)**, because the TWM consumes image patches and `image_size=63, channels=3` matches classic-pixels exactly, and the TWM README comment says "craftax is 3 channels 63x63". **inferred** (the abstract does not state the observation type).
- Pilot environment recommendation: **`Craftax-Classic-Pixels-v1`** (17 actions), i.e. the same environment family and observation space as the anchor. If a symbolic pilot were run instead, TWM's image tokenizer/`image_size` would have to be replaced by a vector encoder, which the package does not provide. **inferred**

## A2. TWM interface

### A2.1 Signatures

**verified by command** (`inspect.signature` on the installed 0.0.59):

```text
WorldModel.__init__(self, image_size, patch_size, channels, num_actions=0,
  reward_min_value=0.0, reward_max_value=1.0, reward_num_bins=0.0,
  transformer_use_token_embed=True,
  tokenizer: 'NearestNeighborTokenizer | Module | dict' = {},
  transformer: 'BlockCausalTransformer | dict' = {},
  hl_gauss_loss_kwargs: 'dict' = {'sigma_to_bin_ratio': 1.0},
  is_terminal_loss_weight=1.0, reward_loss_weight=1.0, dropout_embed=0.1)

WorldModel.forward(self, state_or_token_ids: "Float['b c t h w'] | Int['b t h w']",
  rewards: "Float['b t'] | None" = None, actions: "Int['b t a'] | None" = None,
  is_terminal: "Bool['b t'] | None" = None, cache=None, remove_cache_len_from_time=True,
  return_cache=False, detach_cache=False, return_loss=True, return_loss_breakdown=False,
  return_embed=False, freeze_tokenizer=True)

WorldModel.sample(self, prompt: "Float['b c t h w']", time_steps, cache=None,
  rewards=None, actions=None, filter_fn=<min_p_filter>, filter_kwargs={}, temperature=1.5,
  return_token_ids=False, return_cache=False, return_rewards_and_done=False)
```

Input tensor: `(batch, channels, time, height, width)` of float frames, or pre-tokenized integer token ids `(batch, time, height, width)`. **verified by command**

### A2.2 Loss

`forward(..., return_loss=True)` (the default) computes the loss internally and returns a scalar: state-token cross entropy + `is_terminal` BCE (weight `is_terminal_loss_weight`) + reward HL-Gauss loss (weight `reward_loss_weight`, enabled only when `reward_num_bins > 0`). Setting `return_loss_breakdown=True` also returns the three components. The caller does not construct targets. **source-inspected** (world_model.py lines 780-916) and **verified by command** (a real forward/backward returned `loss 11.68` in the previous stage).

### A2.3 What the wheel ships

The distribution contains only six modules - `world_model.py`, `agent.py`, `distributed.py`, `mock_env.py`, `tensor_typing.py`, `__init__.py`. There is no `README.md` and no `examples/` directory inside the installed package (the README is only in the PyPI `METADATA`). **verified by command**

`agent.py` provides a complete model-based RL agent:

| Class | Role |
|---|---|
| `Impala` | convolutional encoder with GRU; defaults `image_size=63`, `channels=3`, `init_conv_kernel=7` |
| `Actor` | policy head; `num_actions` is a required keyword |
| `Critic` | value head; HL-Gauss loss with `min_value=0, max_value=5, num_bins=32` by default |
| `Agent` | holds impala/actor/critic; `policy_loss`, `critic_loss`, `learn`, `interact_with_env`, `forward` |

- `Agent.interact_with_env(env, memories=None, world_model=None, max_steps=inf)` rolls out the **real** environment. **source-inspected**
- `Agent.forward(world_model, init_state, memories=None, max_steps=inf, use_world_model_embed=False)` rolls out **imagined** trajectories inside the world model (decorated with `torch.no_grad()`). **source-inspected**
- `Agent.learn(memories, lam=0.95, gamma=0.99, batch_size=16, epochs=2)` performs GAE and updates **only the actor and critic (plus the shared Impala encoder)** - `actor_optim` / `critic_optim`. There is **no world-model optimizer and no world-model training step** anywhere in the package. **source-inspected**

### A2.4 Environment protocol the agent expects

From the shipped `mock_env.py` **source-inspected**:

```python
reset() -> Float['c h w']                                    # one frame
forward(actions: Int['a']) -> (Float['c h w'], Scalar, Bool[''])  # next frame, reward, done
```

`Agent.interact_with_env` wraps `env.reset` and `env(...)` with `outputs_to_device(device)`, and `to_device` maps only torch tensors (`t.to(device) if is_tensor(t) else t`). A Craftax adapter therefore has to return **torch tensors** itself; returning numpy/JAX arrays would not be moved to the GPU. **source-inspected**

### A2.5 Minimal verification (no training)

Run with the shipped mock environment to prove the stack executes on this torch/cuDNN build. **verified by command**, exit 0:

```text
torch 2.4.1+cu121 cuda True
WM ok: num_actions 17 can_pred_reward True                 # WorldModel(..., num_actions=17, reward_num_bins=65)
Agent ok: actor.num_actions 17
interact_with_env OK: memories 2 next_state (3, 1, 63, 63) from_real_env True
imagined rollout OK: memories 1 from_real_env False        # Agent.forward(world_model, init_state)
peak MB 739.79904
```

No optimizer step was taken: `interact_with_env` and `forward` are rollouts only.

### A2.6 A5.2 conclusions (TWM interface)

- Input tensor: `(b, c, t, h, w)` float frames - for Craftax classic-pixels `(b, 3, t, 63, 63)`. Integer token ids `(b, t, h, w)` are also accepted. **verified by command**
- Loss: computed internally by `forward(return_loss=True)`; no external target needed. **verified by command**
- Built-in training loop: the policy/critic loop exists (`Agent.learn`), but **the world-model training loop does not**. The pilot must supply it. **source-inspected**
- Missing pieces the pilot must add: (1) the world-model optimizer/training step, (2) a Craftax -> torch environment adapter, (3) the "Dyna with warmup" schedule from the paper (train the world model on real data first, then start mixing imagined data), (4) evaluation/metrics plumbing.

## A3. Craftax_Baselines PPO

### A3.1 Layout

**verified by command** (`~/craftax-external/Craftax_Baselines`):

```text
ppo.py         ppo_rnd.py      ppo_rnn.py     wrappers.py
models/{actor_critic.py, icm.py, rnd.py, __init__.py}
logz/{batch_logging.py, __init__.py}
analysis/{view_ppo_agent.py, __init__.py}
requirements.txt  Dockerfile  README.md  build.sh  run_docker.sh
```

`requirements.txt` (verbatim) requests `jax[cuda12_pip]` - note this extra name does not exist in the installed JAX 0.6.2, which is the benign warning recorded in `docs/setup-log.md` section 10.1.

### A3.2 Entry point signature

**source-inspected** (`ppo.py`):

- `make_train(config) -> train(rng)` - the env is created inside `make_train` from `config["ENV_NAME"]` via `make_craftax_env_from_name`, then wrapped with `LogWrapper` plus either `OptimisticResetVecEnvWrapper` or `AutoResetEnvWrapper` + `BatchEnvWrapper`.
- `train(rng)` returns `{"runner_state": runner_state}` (line 616) and is otherwise a self-contained `jax.lax.scan` rollout + PPO update.
- `run_ppo(config)` upper-cases the argparse namespace, calls `wandb.init(...)` when `config["USE_WANDB"]`, then runs `jax.vmap(jax.jit(make_train(config)))` and prints `SPS`.
- CLI defaults (from `__main__`): `--env_name Craftax-Symbolic-v1`, `--num_envs 1024`, `--total_timesteps 1e9`, `--num_steps 64`, `--update_epochs 4`, `--num_minibatches 8`, `--lr 2e-4`, `--layer_size 512`, `--use_wandb True`, `--use_optimistic_resets True`.

### A3.3 External callability

**verified by command** with `sys.path.insert(0, "~/craftax-external/Craftax_Baselines")` and no source edits:

```text
import ppo OK
make_train: (config)
make_train(...) returned: function signature: (rng)   # accepts a config dict, returns train(rng)
wandb installed / flax installed / optax installed / distrax installed / pygame installed / gymnax installed
```

Import is possible from another process because the module only defines functions and classes at import time; `wandb.init` is reached only inside `run_ppo` (and only when `USE_WANDB` is set), and argument parsing is guarded by `if __name__ == "__main__"`.

### A3.4 Can Dyna data be injected without editing the source?

**No, not into `make_train` as-is.** **source-inspected**: the rollout is generated inside the scan over the internally created Craftax environment (`env.reset` / `env.step` at lines 203 and 226); there is no parameter for externally supplied transitions, no replay buffer, and no world-model hook. Injection would require either

1. writing our own loop that imports the reusable pieces (`models/actor_critic.py`, `wrappers.py`) - no source modification; or
2. runtime monkey-patching the wrapper chain from our own script - also no file edits, but more fragile.

### A3.5 A5.3 conclusions (PPO interface)

- Entry point: `make_train(config) -> train(rng)`; driver wrapper `run_ppo(config)`; PPO hyper-parameters are a flat config dict (argparse defaults listed above). **verified by command**
- External call: yes, importable and callable from another process without touching the repository. **verified by command**
- Dyna injection without source edits: **not through `make_train`**; the loop must be re-authored by us from the importable components, or patched at runtime. **source-inspected**
- Stack: **JAX + Flax + Optax** (`flax.training.train_state.TrainState`, `optax.chain`, `jax.vmap`/`jax.lax.scan`), not PyTorch. **verified by command + source-inspected**

## A4. Official ICML 2025 implementation

**verified by command** (arXiv abstract page `2502.01591v3`, and HTTP status probes; nothing was cloned):

| Probe | Result |
|---|---|
| `curl https://arxiv.org/abs/2502.01591` | exit 0, 43849 bytes |
| github links on the abs page | none found (grep for `github.com` returned no repository link) |
| arXiv "code" link | none found on the abs page |
| `https://github.com/MichaelTMatthews/twm` | HTTP 404 |
| `https://github.com/MichaelTMatthews/Craftax_Baselines` | HTTP 200 |
| `https://github.com/lucidrains/improving-transformers-world-model` | HTTP 404 |
| `https://github.com/Danijar/dreamerv3` | HTTP 200 |

Abstract facts **verified by command**: the paper's three contributions are "Dyna with warmup", the "nearest neighbor tokenizer" for image patches, and "block teacher forcing"; the headline anchor is Craftax-classic 69.66% at 1M steps; additional preliminary results are on Craftax-full, MinAtar and two-player games.

### A5.4 conclusions (official implementation)

- Official code URL: **unconfirmed.** No code repository is linked from the arXiv abstract page. The paper's own code (a JAX/Flax system, judging by the language of the abstract and the Craftax ecosystem) was not located.
- `improving-transformers-world-model` on PyPI is published by `lucidrains` and is a **third-party PyTorch reimplementation** of the paper's method; its own repository URL is 404, so the wheel is the only artifact we have. **verified by command**
- `Craftax_Baselines` (HTTP 200) is a JAX PPO baseline repository for Craftax (used for baseline comparisons such as PPO), not the paper's TWM implementation. **inferred**
- Differences relevant to the pilot: the PyTorch package supplies a world model + PPO agent but no world-model training loop and no environment adapter; the official system (unavailable) would have contained the harness. **inferred**

## A5.5 Executability

**Can a minimal TWM + PPO loop be assembled without modifying any third-party source? Yes** - with the caveat that a small amount of glue code must be written in this repository (writing our own scripts is not third-party source modification).

Everything needed is importable and was verified to run:

| Piece | Status |
|---|---|
| Craftax classic-pixels environment (`(63,63,3)`, 17 actions) | verified by command |
| TWM world model + internal loss | verified by command |
| TWM imagined rollout (`Agent.forward`) and real-environment rollout (`Agent.interact_with_env`) | verified by command (mock env) |
| Policy/critic learning (`Agent.learn`) | source-inspected (not executed; it performs optimizer steps, which this phase forbids) |

Components that do **not** exist anywhere in the installed packages and must be written by us:

1. **Craftax -> torch adapter** implementing the `reset()` / `forward(actions)` protocol: run the JAX env, transpose HWC -> CHW, keep the `[0, 1]` float scaling, convert reward/done to torch tensors on the GPU.
2. **World-model training step**: `loss = world_model(frames, actions=..., rewards=..., is_terminal=...)` plus AdamW - the package has no world-model optimizer.
3. **Dyna-with-warmup schedule**: hold back imagined data until the world model has trained for a warmup period (paper contribution (a)).
4. **Imagined-rollout -> learner plumbing**: feed `Agent.forward(...)` memories into `Agent.learn(...)` together with real memories.
5. **Evaluation/metrics harness**: reward aggregation over the pilot horizon to compare against the 69.66% anchor.

Design sketch (pseudocode only, not implemented):

```text
# 1. environment (JAX, existing package)
env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=True)

# 2. adapter (our code): torch Module with reset()/forward(actions) -> torch tensors
adapter = CraftaxClassicPixelsAdapter(env, device="cuda")   # HWC->CHW, /255 kept, reward/done -> tensors

# 3. world model + its optimizer (our code, package supplies the module)
wm = WorldModel(image_size=63, patch_size=7, channels=3, num_actions=17, reward_num_bins=65,
                transformer=dict(dim=512, depth=4, block_size=81),
                tokenizer=dict(dim=7*7*3, distance_threshold=0.5)).to("cuda")
wm_opt = torch.optim.AdamW(wm.parameters(), lr=...)

# 4. agent (package supplies the module; learn() updates actor/critic only)
agent = Agent(impala=dict(image_size=63, channels=3), actor=dict(dim=..., num_actions=17), critic=dict(dim=...)).cuda()

for step in range(pilot_steps):
    real = agent.interact_with_env(adapter, max_steps=...)        # real rollout
    frames = real.next_state                                     # (c,1,h,w) -> to (b,c,t,h,w) buffer
    loss = wm(frames, actions=..., rewards=..., is_terminal=...)  # world-model loss
    wm_opt.zero_grad(); loss.backward(); wm_opt.step()            # (2) world-model training

    if step >= warmup_steps:                                     # (3) Dyna with warmup
        imagined = agent.forward(wm, init_state, max_steps=...)
        agent.learn([real, imagined])                            # (4) shared policy update
    else:
        agent.learn([real])

    if step % eval_every == 0:                                   # (5) evaluation
        log(mean_episode_reward)
```

## Unconfirmed items

- Official ICML 2025 code repository: not located (no link on the arXiv abstract page). **unconfirmed**
- How many actions per frame the paper feeds through `Int['b t a']`, i.e. whether Craftax's repeated-action semantics are used. **unconfirmed**
- The paper's exact world-model hyper-parameters (optimizer, learning rate, warmup length, `reward_num_bins`, patch/dim/depth values beyond the README example). Only the README example values are known: `image_size=63, patch_size=7, channels=3, dim=512, depth=4, block_size=81`, `tokenizer dim=147, distance_threshold=0.5`. The `reward_num_bins=65` used in the smoke test above was our choice. **unconfirmed**
- Whether the 69.66% anchor used pixel or symbolic observations: pixel is inferred from the 63x63x3 TWM configuration. **inferred, not confirmed**

## Operational note

Querying four environments inside a single JAX process exhausted GPU memory (23.9 GB of 24.6 GB in use) and stalled in cuDNN autotuning for over 30 minutes without producing output. Each environment query was therefore re-run in its own process with `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_FLAGS=--xla_gpu_autotune_level=0`, after which every query completed in seconds to minutes. The pilot harness should set a memory fraction or these XLA flags rather than relying on default preallocation. **verified by command**
