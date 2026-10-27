#!/usr/bin/env python3
"""Pure Craftax_Baselines PPO sanity check on Craftax-Classic-Pixels-v1.

Purpose: separate failures in the local TWM diagnostic harness from failures in
the official Craftax_Baselines PPO implementation.

This script calls the unmodified `make_train` from Craftax_Baselines.  The
official `make_train` computes its per-update metric but intentionally discards
it and only exposes the final runner state.  To retain 10k-step checkpoints
without editing third-party source, this script captures the outer
`_update_step` and its initial runner state while `make_train` is constructing
the training program, then drives that exact closure with `jax.lax.scan`.
No Craftax, Craftax_Baselines, or TWM source file is modified.

Diagnostic alignment:
  - same environment: Craftax-Classic-Pixels-v1
  - same seed: 0
  - same total environment-step budget: 1,000,000
  - same 20-step rollout horizon as the TWM diagnostic
  - one update per 10k-step checkpoint: 500 envs x 20 steps
  - same evaluation: 8 envs x 250 steps
  - same mechanic extraction: inventory, achievements, player position

Deviations from the upstream CLI defaults:
  - upstream defaults are 1,024 envs x 64 steps and 1e9 total steps;
    to permit exact 10k-step evaluation, this run uses 500 x 20.  The TWM
    diagnostic used 50 x 20 and 1,000 updates; its 1k-transition PPO batches
    made the pure PPO run unnecessarily slow, so the env count is increased
    while the rollout horizon stays fixed.
  - optimistic resets are disabled so the PPO and TWM paths use the same
    AutoReset + BatchEnvWrapper path.

Usage:
  python scripts/ppo_baseline_sanity.py
  python scripts/ppo_baseline_sanity.py --total-timesteps 20000
"""

import os

# Must be set before importing jax.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import importlib.metadata
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

import jax
import jax.numpy as jnp

from craftax.craftax_env import make_craftax_env_from_name


BASELINES_DIR = pathlib.Path(
    os.environ.get(
        "CRAFTAX_BASELINES_DIR",
        str(pathlib.Path.home() / "craftax-external" / "Craftax_Baselines"),
    )
)
if not (BASELINES_DIR / "ppo.py").exists():
    raise SystemExit(f"Craftax_Baselines not found at {BASELINES_DIR}")
sys.path.insert(0, str(BASELINES_DIR))

from ppo import make_train  # noqa: E402


LOG_DIR = pathlib.Path.home() / "craftax-workspace" / "logs"
BUCKETS = ["normal", "exploration", "combat", "crafting"]
INV_NAMES = [
    "wood",
    "stone",
    "coal",
    "iron",
    "diamond",
    "sapling",
    "wood_pickaxe",
    "stone_pickaxe",
    "iron_pickaxe",
    "wood_sword",
    "stone_sword",
    "iron_sword",
]

# Same indices as scripts/twm_baseline_diagnostic.py.
ACH_CRAFT = [0, 1, 3, 5, 6, 7, 9, 10, 13, 14, 16, 17, 18, 19, 20, 21]
ACH_COMBAT = [8, 12]


class _CapturedScan(Exception):
    pass


def parse_args():
    p = argparse.ArgumentParser(description="Pure PPO Craftax sanity baseline")
    p.add_argument("--env", default="Craftax-Classic-Pixels-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--num-envs", type=int, default=500)
    p.add_argument("--rollout-steps", type=int, default=20)
    p.add_argument("--checkpoint-every", type=int, default=10_000)
    p.add_argument("--eval-envs", type=int, default=8)
    p.add_argument("--eval-horizon", type=int, default=250)
    p.add_argument("--eval-seed", type=int, default=4242)
    p.add_argument("--json-out", default=str(LOG_DIR / "ppo_baseline_sanity.json"))
    return p.parse_args()


def build_config(args):
    # Official CLI defaults, with only the rollout shape and run budget changed.
    return {
        "ENV_NAME": args.env,
        "NUM_ENVS": args.num_envs,
        "TOTAL_TIMESTEPS": args.total_timesteps,
        "LR": 2e-4,
        "NUM_STEPS": args.rollout_steps,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 8,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.8,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 1.0,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
        "DEBUG": False,
        "USE_WANDB": False,
        "SAVE_POLICY": False,
        "NUM_REPEATS": 1,
        "LAYER_SIZE": 512,
        "USE_OPTIMISTIC_RESETS": False,
        "OPTIMISTIC_RESET_RATIO": 16,
        "EXPLORATION_UPDATE_EPOCHS": 4,
        "TRAIN_ICM": False,
        "ICM_REWARD_COEFF": 1.0,
        "ICM_LR": 3e-4,
        "ICM_FORWARD_LOSS_COEF": 1.0,
        "ICM_INVERSE_LOSS_COEF": 1.0,
        "ICM_LAYER_SIZE": 256,
        "ICM_LATENT_SIZE": 32,
        "E3B_REWARD_COEFF": 1.0,
        "USE_E3B": False,
        "E3B_LAMBDA": 0.1,
        "SEED": args.seed,
    }


def capture_official_update(config, seed):
    """Capture make_train's own initial state and outer update closure.

    This invokes the unmodified `make_train`.  The monkeypatch only intercepts
    the outer `jax.lax.scan` call and aborts before any training update runs.
    """
    captured = {}
    original_scan = jax.lax.scan

    def capture_scan(fn, init, xs=None, length=None, **kwargs):
        if (
            getattr(fn, "__name__", "") == "_update_step"
            and xs is None
            and length == config["NUM_UPDATES"]
        ):
            captured["step"] = fn
            captured["init"] = init
            raise _CapturedScan
        return original_scan(fn, init, xs, length=length, **kwargs)

    jax.lax.scan = capture_scan
    try:
        try:
            make_train(config)(jax.random.PRNGKey(seed))
        except _CapturedScan:
            pass
    finally:
        jax.lax.scan = original_scan

    if "step" not in captured or "init" not in captured:
        raise RuntimeError("failed to capture official PPO update closure")
    return captured["step"], captured["init"]


def build_eval_env(env_name):
    env = make_craftax_env_from_name(env_name, auto_reset=True)
    params = env.default_params
    num_actions = int(env.action_space(params).n)
    obs_shape = tuple(env.observation_space(params).shape)
    if obs_shape != (63, 63, 3):
        raise SystemExit(f"expected (63,63,3) obs, got {obs_shape}")
    if num_actions != 17:
        raise SystemExit(f"expected 17 actions, got {num_actions}")

    def compact_metrics(state, done):
        inv = state.inventory
        inv_vec = jnp.stack(
            [
                inv.wood,
                inv.stone,
                inv.coal,
                inv.iron,
                inv.diamond,
                inv.sapling,
                inv.wood_pickaxe,
                inv.stone_pickaxe,
                inv.iron_pickaxe,
                inv.wood_sword,
                inv.stone_sword,
                inv.iron_sword,
            ]
        ).astype(jnp.int32)

        tier = jnp.where(
            inv.diamond > 0,
            4,
            jnp.where(
                (inv.iron_pickaxe > 0) | (inv.iron_sword > 0),
                3,
                jnp.where(
                    (inv.stone_pickaxe > 0) | (inv.stone_sword > 0),
                    2,
                    jnp.where(
                        (inv.wood_pickaxe > 0) | (inv.wood_sword > 0), 1, 0
                    ),
                ),
            ),
        ).astype(jnp.int32)

        def min_hostile_dist(mobs):
            d = jnp.max(
                jnp.abs(mobs.position - state.player_position), axis=-1
            ).astype(jnp.float32)
            alive = mobs.mask & (mobs.health > 0)
            return jnp.min(jnp.where(alive, d, 999.0))

        hostile_dist = jnp.minimum(
            min_hostile_dist(state.zombies), min_hostile_dist(state.skeletons)
        )
        mobs_alive = (
            state.zombies.mask.sum()
            + state.skeletons.mask.sum()
            + state.cows.mask.sum()
        ).astype(jnp.int32)

        return {
            "pos": state.player_position.astype(jnp.int32),
            "health": state.player_health.astype(jnp.float32),
            "food": state.player_food.astype(jnp.float32),
            "drink": state.player_drink.astype(jnp.float32),
            "energy": state.player_energy.astype(jnp.float32),
            "timestep": state.timestep.astype(jnp.int32),
            "inv": inv_vec,
            "tier": tier,
            "ach": state.achievements,
            "hostile_dist": hostile_dist,
            "hostile_near": hostile_dist <= 2.0,
            "mobs_alive": mobs_alive,
            "done": done,
        }

    def raw_step(key, state, action):
        obs, state, reward, done, _info = env.step(key, state, action, params)
        return obs, state, reward, done, compact_metrics(state, done)

    def reset_one(key):
        return env.reset(key, params)

    step_batched = jax.jit(jax.vmap(raw_step, in_axes=(0, 0, 0)))

    @jax.jit
    def reset_batched(keys):
        return jax.vmap(reset_one)(keys)

    return step_batched, reset_batched, num_actions


def tree_numpy(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), tree)


def update_visited(visited, positions):
    b = positions.shape[0]
    xs = np.clip(positions[:, 0], 0, 63)
    ys = np.clip(positions[:, 1], 0, 63)
    new = ~visited[np.arange(b), ys, xs]
    visited[np.arange(b), ys, xs] = True
    return new


def mark_initial(visited, positions):
    b = positions.shape[0]
    xs = np.clip(positions[:, 0], 0, 63)
    ys = np.clip(positions[:, 1], 0, 63)
    visited[np.arange(b), ys, xs] = True


def classify_step(prev, cur, new_tile):
    inv_delta = np.abs(cur["inv"] - prev["inv"]).sum(axis=-1)
    ach_new = cur["ach"] & ~prev["ach"]
    craft = (inv_delta > 0) | ach_new[:, ACH_CRAFT].any(axis=-1)
    combat = cur["hostile_near"] | ach_new[:, ACH_COMBAT].any(axis=-1)
    labels = np.zeros(inv_delta.shape[0], dtype=np.int8)
    labels[new_tile] = 1
    labels[combat] = 2
    labels[craft] = 3
    return labels


def make_policy_act(apply_fn):
    @jax.jit
    def act(params, obs, key):
        pi, value = apply_fn(params, obs)
        action = pi.sample(seed=key)
        mode = jnp.argmax(pi.probs, axis=-1)
        return action, mode, pi.entropy(), value

    return act


def eval_policy(params, apply_fn, step_batched, reset_batched, rng, num_envs, horizon):
    act = make_policy_act(apply_fn)
    rng, reset_key = jax.random.split(rng)
    obs, state = reset_batched(jax.random.split(reset_key, num_envs))

    visited = np.zeros((num_envs, 64, 64), dtype=bool)
    mark_initial(visited, np.asarray(state.player_position))
    coverage_peak = np.zeros(num_envs)

    prev = None
    episode_lengths = []
    current_age = np.zeros(num_envs, dtype=np.int64)
    rewards = []
    metrics_list = []
    action_counts = np.zeros(17, dtype=np.int64)
    mode_counts = np.zeros(17, dtype=np.int64)
    policy_entropies = []
    counts = {b: 0 for b in BUCKETS}

    for _ in range(horizon):
        rng, act_key, step_key = jax.random.split(rng, 3)
        actions, modes, entropies, _ = act(params, obs, act_key)
        act_np = np.asarray(jax.device_get(actions), dtype=np.int32)
        mode_np = np.asarray(jax.device_get(modes), dtype=np.int32)
        action_counts += np.bincount(act_np, minlength=17)
        mode_counts += np.bincount(mode_np, minlength=17)
        policy_entropies.append(np.asarray(jax.device_get(entropies), dtype=np.float32))

        next_obs, state, reward, done, metrics = step_batched(
            jax.random.split(step_key, num_envs), state, jnp.asarray(act_np)
        )
        cur = tree_numpy(metrics)
        new_tile = update_visited(visited, cur["pos"])
        if prev is not None:
            for label in classify_step(prev, cur, new_tile):
                counts[BUCKETS[int(label)]] += 1

        done_np = np.asarray(jax.device_get(done))
        current_age += 1
        for i in np.nonzero(done_np)[0]:
            episode_lengths.append(int(current_age[i]))
            current_age[i] = 0
        if done_np.any():
            cov = visited.sum(axis=(1, 2)) / float(64 * 64)
            coverage_peak = np.maximum(coverage_peak, cov)
            visited[done_np] = False

        rewards.append(np.asarray(jax.device_get(reward)))
        metrics_list.append(cur)
        prev = cur
        obs = next_obs

    cov = visited.sum(axis=(1, 2)) / float(64 * 64)
    coverage_peak = np.maximum(coverage_peak, cov)

    stacked = {
        k: np.stack([m[k] for m in metrics_list], axis=0) for k in metrics_list[0]
    }
    flat = {k: v.reshape((-1,) + v.shape[2:]) for k, v in stacked.items()}
    ach_max = stacked["ach"].max(axis=0)
    inv_max = stacked["inv"].max(axis=0).max(axis=0)

    action_probs = action_counts / max(int(action_counts.sum()), 1)
    action_entropy = float(
        -np.sum(action_probs[action_probs > 0] * np.log(action_probs[action_probs > 0]))
    )
    all_lengths = episode_lengths + [int(v) for v in current_age if v > 0]
    total_actions = max(int(action_counts.sum()), 1)

    return {
        "reward_mean": float(np.mean(np.stack(rewards))),
        "reward_last100_mean": float(np.mean(np.stack(rewards)[-100:])),
        "craft_tier_max": int(flat["tier"].max()),
        "craft_tier_mean": float(flat["tier"].mean()),
        "achievements_max": int(ach_max.sum(axis=1).max()),
        "achievements_mean": float(ach_max.sum(axis=1).mean()),
        "kills_max": int(ach_max[:, ACH_COMBAT].sum(axis=1).max()),
        "kills_mean": float(ach_max[:, ACH_COMBAT].sum(axis=1).mean()),
        "resource_max": {
            name: int(value) for name, value in zip(INV_NAMES, inv_max)
        },
        "exploration_coverage_mean": float(coverage_peak.mean()),
        "exploration_coverage_max": float(coverage_peak.max()),
        "survival_steps_mean": float(np.mean(episode_lengths))
        if episode_lengths
        else float("nan"),
        "survival_steps_max": float(np.max(episode_lengths))
        if episode_lengths
        else float("nan"),
        "survival_censored_mean": float(np.mean(all_lengths))
        if all_lengths
        else float("nan"),
        "survival_censored_max": float(np.max(all_lengths))
        if all_lengths
        else float("nan"),
        "final_timestep_mean": float(flat["timestep"].max(axis=0).mean()),
        "episodes_finished": int(len(episode_lengths)),
        "action_counts": [int(v) for v in action_counts],
        "mode_counts": [int(v) for v in mode_counts],
        "noop_fraction": float(action_counts[0] / total_actions),
        "mode_noop_fraction": float(mode_counts[0] / total_actions),
        "action_entropy": action_entropy,
        "policy_entropy_mean": float(
            np.mean(np.concatenate(policy_entropies, axis=0))
        ),
        "event_counts": counts,
        "steps": num_envs * horizon,
    }


def nvidia_smi_used_mb():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        return float("nan")


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "UNKNOWN"


def main():
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    config = build_config(args)

    steps_per_update = args.num_envs * args.rollout_steps
    if args.total_timesteps % steps_per_update != 0:
        raise SystemExit(
            f"total_timesteps must be divisible by {steps_per_update}; "
            f"got {args.total_timesteps}"
        )
    if args.checkpoint_every % steps_per_update != 0:
        raise SystemExit(
            f"checkpoint_every must be divisible by {steps_per_update}"
        )

    print(
        f"PPO sanity: env={args.env} seed={args.seed} steps={args.total_timesteps} "
        f"rollout={args.num_envs}x{args.rollout_steps}",
        flush=True,
    )
    print(f"jax {jax.__version__} devices={jax.devices()}", flush=True)

    t_capture = time.perf_counter()
    update_step, runner_state = capture_official_update(config, args.seed)
    capture_sec = time.perf_counter() - t_capture
    num_updates = int(config["NUM_UPDATES"])
    updates_per_checkpoint = args.checkpoint_every // steps_per_update
    print(
        f"captured official PPO update: updates={num_updates}, "
        f"updates_per_checkpoint={updates_per_checkpoint}, "
        f"capture={capture_sec:.2f}s",
        flush=True,
    )

    def step_state(rs, _):
        rs, _metric = update_step(rs, None)
        return rs, None

    def run_chunk(rs):
        return jax.lax.scan(
            step_state, rs, None, length=updates_per_checkpoint
        )[0]

    chunk_fn = jax.jit(run_chunk)
    step_batched, reset_batched, num_actions = build_eval_env(args.env)
    if num_actions != 17:
        raise SystemExit(f"unexpected action count {num_actions}")

    checkpoints = []
    peak_vram = 0.0
    start = time.perf_counter()
    env_steps = 0

    while env_steps < args.total_timesteps:
        chunk_start = time.perf_counter()
        runner_state = chunk_fn(runner_state)
        jax.block_until_ready(runner_state)
        env_steps += updates_per_checkpoint * steps_per_update
        chunk_sec = time.perf_counter() - chunk_start

        train_state = runner_state[0]
        eval_rng = jax.random.fold_in(jax.random.PRNGKey(args.eval_seed), env_steps)
        eval_start = time.perf_counter()
        metrics = eval_policy(
            train_state.params,
            train_state.apply_fn,
            step_batched,
            reset_batched,
            eval_rng,
            args.eval_envs,
            args.eval_horizon,
        )
        eval_sec = time.perf_counter() - eval_start
        used_mb = nvidia_smi_used_mb()
        if np.isfinite(used_mb):
            peak_vram = max(peak_vram, used_mb)

        record = {
            "env_steps": env_steps,
            "chunk_sec": chunk_sec,
            "eval_sec": eval_sec,
            "nvidia_smi_used_mb": used_mb,
            **metrics,
        }
        checkpoints.append(record)
        print(
            f"[{env_steps:>8d}] reward {metrics['reward_mean']:+.5f} "
            f"tier {metrics['craft_tier_max']} "
            f"ach {metrics['achievements_max']} "
            f"coverage {metrics['exploration_coverage_mean']:.2%} "
            f"noop {metrics['noop_fraction']:.1%} "
            f"entropy {metrics['policy_entropy_mean']:.4f} "
            f"eval {eval_sec:.2f}s",
            flush=True,
        )

    elapsed = time.perf_counter() - start
    final = checkpoints[-1]
    last10 = checkpoints[-10:]
    summary = {
        "env": args.env,
        "seed": args.seed,
        "requested_total_timesteps": args.total_timesteps,
        "actual_total_timesteps": env_steps,
        "num_envs": args.num_envs,
        "rollout_steps": args.rollout_steps,
        "num_updates": num_updates,
        "checkpoint_every": args.checkpoint_every,
        "eval_envs": args.eval_envs,
        "eval_horizon": args.eval_horizon,
        "elapsed_sec": elapsed,
        "steps_per_sec": env_steps / elapsed,
        "capture_sec": capture_sec,
        "peak_nvidia_smi_used_mb": peak_vram,
        "final": final,
        "reward_first10_checkpoint_mean": float(
            np.mean([c["reward_mean"] for c in checkpoints[:10]])
        ),
        "reward_last10_checkpoint_mean": float(
            np.mean([c["reward_mean"] for c in last10])
        ),
        "coverage_last10_checkpoint_mean": float(
            np.mean([c["exploration_coverage_mean"] for c in last10])
        ),
        "noop_last10_checkpoint_mean": float(
            np.mean([c["noop_fraction"] for c in last10])
        ),
        "entropy_last10_checkpoint_mean": float(
            np.mean([c["policy_entropy_mean"] for c in last10])
        ),
        "config": config,
        "versions": {
            "python": sys.version.split()[0],
            "jax": jax.__version__,
            "craftax": package_version("craftax"),
            "craftax_baselines_revision": subprocess.check_output(
                ["git", "-C", str(BASELINES_DIR), "rev-parse", "HEAD"],
                text=True,
            ).strip(),
        },
        "checkpoints": checkpoints,
    }

    out_path = pathlib.Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(
        f"DONE {env_steps} env steps in {elapsed / 60.0:.2f} min "
        f"({elapsed / 3600.0:.2f} h), peak nvidia-smi {peak_vram:.1f} MB",
        flush=True,
    )
    print(
        f"FINAL reward {final['reward_mean']:+.5f}/step "
        f"(last10 checkpoint mean {summary['reward_last10_checkpoint_mean']:+.5f}) "
        f"tier {final['craft_tier_max']} ach {final['achievements_max']} "
        f"coverage {final['exploration_coverage_mean']:.2%} "
        f"noop {final['noop_fraction']:.1%} "
        f"policy entropy {final['policy_entropy_mean']:.4f}",
        flush=True,
    )
    print(f"JSON {out_path}", flush=True)


if __name__ == "__main__":
    main()
