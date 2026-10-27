#!/usr/bin/env python3
"""End-to-end integration smoke test for the Craftax world-model pilot.

Per step this script exercises the full cross-framework chain:

  1. Craftax-Classic-Pixels-v1 rollout (JAX, batched + jitted)
  2. HWC -> CHW + numpy -> torch (cuda)
  3. TWM world-model train step: forward(return_loss=True) + backward + AdamW
  4. TWM imagined rollout: WorldModel.sample(...)
  5. imagined frames -> numpy -> JAX
  6. Craftax_Baselines PPO update (real-env 1-update fallback)

No convergence is expected or measured. The point is mechanism + measured cost.

PPO injection note: `ppo.make_train(config)` creates and rolls out its own
Craftax environment and exposes no API for externally supplied transitions
(source-inspected in Phase A). We therefore time a real-environment 1-update
`train(rng)` call as the PPO cost baseline and mark external injection
UNSUPPORTED.
"""

import os

# Must be set before importing jax / torch.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

import jax
import jax.numpy as jnp

import torch

from craftax.craftax_env import make_craftax_env_from_name

from improving_transformers_world_model.world_model import (
    NearestNeighborTokenizer,
    pack_one_with_inverse,
)

WORKSPACE = pathlib.Path(__file__).resolve().parents[1]
CRAFT_BASELINES = pathlib.Path.home() / "craftax-external" / "Craftax_Baselines"

PHASES = [
    "env sample",
    "jax->torch convert",
    "TWM train step",
    "TWM imagine",
    "torch->jax convert",
    "PPO update",
]


class StableNearestNeighborTokenizer(NearestNeighborTokenizer):
    """Local workaround for improving-transformers-world-model 0.0.59.

    Upstream ``NearestNeighborTokenizer.forward`` ends with
    ``nearest_neighbor_ids.masked_fill_(mask, new_code_ids)`` where
    ``new_code_ids`` is 1-D. torch requires a 0-dim fill value, so the stock
    code raises 'masked_fill_ only supports a 0-dimensional value tensor'
    whenever two or more new patches are discovered in one batch. This subclass
    reproduces the upstream logic verbatim and replaces only that assignment
    with an equivalent index assignment. The third-party package is untouched.
    """

    def forward(self, x, ignore_dist_threshold=None, freeze=False):
        if ignore_dist_threshold is None:
            ignore_dist_threshold = not self.training or self.is_at_max_codes

        x, inverse_pack_one = pack_one_with_inverse(x, "b * d", "b *")

        num_codes = self.num_codes.item()

        if num_codes == 0:
            self.add_codes_(x)
            ids = torch.cdist(x, self.codes).argmin(dim=-1)
            return inverse_pack_one(ids)

        distance_sq = torch.cdist(x, self.codes) ** 2

        if ignore_dist_threshold:
            ids = distance_sq.argmin(dim=-1)
            return inverse_pack_one(ids)

        within_dist_threshold = (distance_sq <= self.distance_threshold).any(dim=-1)
        nearest_neighbor_ids = distance_sq.argmin(dim=-1)
        nearest_neighbor_ids = torch.where(
            within_dist_threshold,
            nearest_neighbor_ids,
            torch.full_like(nearest_neighbor_ids, self.no_code_id),
        )

        if freeze or not self.training:
            return inverse_pack_one(nearest_neighbor_ids)

        all_within_dist_threshold = within_dist_threshold.all()

        if not all_within_dist_threshold:
            new_codes = x[~within_dist_threshold]
            self.add_codes_(new_codes)
            new_code_ids = torch.cdist(new_codes, self.codes).argmin(dim=-1)
            nearest_neighbor_ids = nearest_neighbor_ids.clone()
            nearest_neighbor_ids[~within_dist_threshold] = new_code_ids

        return inverse_pack_one(nearest_neighbor_ids)


def parse_args():
    p = argparse.ArgumentParser(description="Craftax/TWM/PPO integration smoke test")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env", type=str, default="Craftax-Classic-Pixels-v1")
    # env sampling
    p.add_argument("--batch", type=int, default=8,
                   help="number of parallel Craftax environments")
    p.add_argument("--frames", type=int, default=32,
                   help="environment frames collected per step")
    p.add_argument("--wm-time", type=int, default=8,
                   help="frames per sequence fed to the world model (quadratic attention)")
    # TWM
    p.add_argument("--imagine", type=int, default=8,
                   help="K imagined frames per step")
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--reward-bins", type=float, default=65)
    # PPO
    p.add_argument("--ppo-envs", type=int, default=8)
    p.add_argument("--ppo-steps", type=int, default=32)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--ppo-minibatches", type=int, default=8)
    # misc
    p.add_argument("--warmup", type=int, default=2,
                   help="unmeasured warmup steps (compilation/autotune)")
    p.add_argument("--json-out", type=str, default=None)
    return p.parse_args()


def gpu_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def nvidia_smi_used_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        return float("nan")


def make_rollout(env, params, batch, frames):
    def reset_one(key):
        return env.reset(key, params)

    def step_one(key, state, action):
        return env.step(key, state, action, params)

    reset_b = jax.vmap(reset_one)
    step_b = jax.vmap(step_one)

    @jax.jit
    def rollout(rng):
        rng, key = jax.random.split(rng)
        obs, state = reset_b(jax.random.split(key, batch))

        def body(carry, _):
            rng, state, obs = carry
            rng, ka, ks = jax.random.split(rng, 3)
            act_keys = jax.random.split(ka, batch)
            actions = jax.vmap(lambda k: env.action_space(params).sample(k))(act_keys)
            obs, state, reward, done, _info = step_b(
                jax.random.split(ks, batch), state, actions
            )
            return (rng, state, obs), (obs, reward, done, actions)

        (rng, state, obs), (frm, rew, dn, act) = jax.lax.scan(
            body, (rng, state, obs), None, length=frames
        )
        return frm, rew, dn, act

    return rollout


def jax_to_torch_window(frames, rewards, dones, actions, wm_time, device):
    """(t,b,h,w,c) -> (b,c,t,h,w) torch tensors on device."""
    f = np.asarray(jax.device_get(frames[-wm_time:]))       # (t, b, h, w, c)
    r = np.asarray(jax.device_get(rewards[-wm_time:]))      # (t, b)
    d = np.asarray(jax.device_get(dones[-wm_time:]))        # (t, b)
    a = np.asarray(jax.device_get(actions[-wm_time:]))      # (t, b)

    f = np.transpose(f, (1, 4, 0, 2, 3))                    # (b, c, t, h, w)
    x = torch.from_numpy(np.ascontiguousarray(f)).to(device)
    r_t = torch.from_numpy(np.ascontiguousarray(r.T)).to(device)
    d_t = torch.from_numpy(np.ascontiguousarray(d.T)).to(device)
    a_t = torch.from_numpy(np.ascontiguousarray(a.T[..., None])).to(device).long()
    return x, r_t, d_t, a_t


def build_ppo(args):
    sys.path.insert(0, str(CRAFT_BASELINES))
    import ppo

    total = args.ppo_envs * args.ppo_steps
    if total % args.ppo_minibatches != 0:
        raise SystemExit("ppo-envs * ppo-steps must be divisible by ppo-minibatches")

    cfg = dict(
        ENV_NAME=args.env,
        NUM_ENVS=args.ppo_envs,
        TOTAL_TIMESTEPS=total,          # exactly NUM_UPDATES == 1
        LR=2e-4,
        NUM_STEPS=args.ppo_steps,
        UPDATE_EPOCHS=args.ppo_epochs,
        NUM_MINIBATCHES=args.ppo_minibatches,
        GAMMA=0.99,
        GAE_LAMBDA=0.8,
        CLIP_EPS=0.2,
        ENT_COEF=0.01,
        VF_COEF=0.5,
        MAX_GRAD_NORM=1.0,
        ACTIVATION="tanh",
        ANNEAL_LR=True,
        DEBUG=False,
        JIT=True,
        SEED=args.seed,
        USE_WANDB=False,
        SAVE_POLICY=False,
        NUM_REPEATS=1,
        LAYER_SIZE=512,
        WANDB_PROJECT=None,
        WANDB_ENTITY=None,
        USE_OPTIMISTIC_RESETS=False,
        OPTIMISTIC_RESET_RATIO=16,
        EXPLORATION_UPDATE_EPOCHS=4,
        ICM_REWARD_COEFF=1.0,
        TRAIN_ICM=False,
        ICM_LR=3e-4,
        ICM_FORWARD_LOSS_COEF=1.0,
        ICM_INVERSE_LOSS_COEF=1.0,
        ICM_LAYER_SIZE=256,
        ICM_LATENT_SIZE=32,
        E3B_REWARD_COEFF=1.0,
        USE_E3B=False,
        E3B_LAMBDA=0.1,
    )
    return jax.jit(ppo.make_train(cfg))


def main():
    args = parse_args()
    if args.wm_time > args.frames:
        raise SystemExit("--wm-time must be <= --frames")
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA is required for this smoke test")

    print("=" * 78)
    print("Craftax / TWM / PPO integration smoke")
    print("=" * 78)
    print(f"steps={args.steps} seed={args.seed} env={args.env}")
    print(f"env batch={args.batch} frames={args.frames} wm_time={args.wm_time}")
    print(f"imagine K={args.imagine} | TWM dim={args.dim} depth={args.depth}")
    print(f"PPO envs={args.ppo_envs} steps={args.ppo_steps} "
          f"epochs={args.ppo_epochs} minibatches={args.ppo_minibatches}")
    print(f"torch={torch.__version__} jax={jax.__version__} "
          f"devices={jax.devices()}")
    print()

    # ---------------------------------------------------------------- env
    env = make_craftax_env_from_name(args.env, auto_reset=True)
    params = env.default_params
    num_actions = int(env.action_space(params).n)
    obs_shape = tuple(env.observation_space(params).shape)
    print(f"env obs={obs_shape} num_actions={num_actions}")
    if obs_shape != (63, 63, 3):
        raise SystemExit(f"expected (63,63,3) observations, got {obs_shape}")

    rollout = make_rollout(env, params, args.batch, args.frames)

    # ----------------------------------------------------------- world model
    from improving_transformers_world_model import WorldModel

    wm = WorldModel(
        image_size=63,
        patch_size=7,
        channels=3,
        num_actions=num_actions,
        reward_min_value=0.0,
        reward_max_value=1.0,
        reward_num_bins=args.reward_bins,
        transformer=dict(dim=args.dim, depth=args.depth, block_size=81),
        tokenizer=StableNearestNeighborTokenizer(dim=7 * 7 * 3, distance_threshold=0.5),
    ).to(device)
    opt = torch.optim.AdamW(wm.parameters(), lr=args.lr)
    n_wm_params = sum(p.numel() for p in wm.parameters())
    print(f"WorldModel params={n_wm_params/1e6:.2f}M")

    # ------------------------------------------------------------------ PPO
    print("building PPO (jit compile on first call) ...")
    ppo_train = build_ppo(args)
    print("PPO external injection: UNSUPPORTED "
          "(make_train has no external-transition API); using real-env update")

    timings = {p: [] for p in PHASES}
    losses = []

    def run_step(rng):
        # 1. env sample
        gpu_sync()
        t0 = time.perf_counter()
        frames, rewards, dones, actions = rollout(rng)
        jax.block_until_ready((frames, rewards, dones, actions))
        timings["env sample"].append(time.perf_counter() - t0)

        # 2. jax -> torch
        gpu_sync()
        t0 = time.perf_counter()
        x, r_t, d_t, a_t = jax_to_torch_window(
            frames, rewards, dones, actions, args.wm_time, device
        )
        gpu_sync()
        timings["jax->torch convert"].append(time.perf_counter() - t0)

        # 3. world-model train step
        wm.train()
        gpu_sync()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = wm(x, rewards=r_t, actions=a_t, is_terminal=d_t)
        loss.backward()
        opt.step()
        gpu_sync()
        timings["TWM train step"].append(time.perf_counter() - t0)
        losses.append(float(loss.detach()))

        # 4. imagined rollout
        gpu_sync()
        t0 = time.perf_counter()
        imagined = wm.sample(
            x[:, :, -1:],
            time_steps=args.imagine,
            rewards=r_t[:, -1:],
            actions=a_t[:, -1:],
        )
        gpu_sync()
        timings["TWM imagine"].append(time.perf_counter() - t0)

        # 5. torch -> jax
        gpu_sync()
        t0 = time.perf_counter()
        j_img = jnp.asarray(imagined.detach().cpu().numpy())
        j_img.block_until_ready()
        timings["torch->jax convert"].append(time.perf_counter() - t0)

        # 6. PPO update (real-env fallback)
        gpu_sync()
        t0 = time.perf_counter()
        out = ppo_train(rng)
        jax.block_until_ready(out)
        gpu_sync()
        timings["PPO update"].append(time.perf_counter() - t0)

        return float(loss.detach())

    # -------------------------------------------------------- warmup phase
    print(f"warmup: {args.warmup} unmeasured step(s) ...")
    w0 = time.perf_counter()
    rng = jax.random.PRNGKey(args.seed)
    for _ in range(max(1, args.warmup)):
        rng, sub = jax.random.split(rng)
        run_step(sub)
    print(f"warmup wall time: {time.perf_counter() - w0:.2f}s "
          f"(tokenizer codes={int(wm.tokenizer.num_codes.item())})")
    print()

    torch.cuda.reset_peak_memory_stats()
    timings = {p: [] for p in PHASES}
    losses = []

    # ---------------------------------------------------------- measured loop
    loop_start = time.perf_counter()
    for step in range(1, args.steps + 1):
        rng, sub = jax.random.split(rng)
        run_step(sub)

        if step % 100 == 0 or step == args.steps:
            elapsed = time.perf_counter() - loop_start
            print(
                f"step {step:5d}/{args.steps} | elapsed {elapsed:8.2f}s | "
                f"per-step {1000 * elapsed / step:7.2f} ms | "
                f"peak {torch.cuda.max_memory_allocated() / 1e6:7.1f} MB | "
                f"codes {int(wm.tokenizer.num_codes.item())}"
            )
            sys.stdout.flush()

    total = time.perf_counter() - loop_start

    # ---------------------------------------------------------------- report
    print()
    print("=" * 78)
    print("Per-phase cost (measured, warmup excluded)")
    print("=" * 78)
    header = f"{'Phase':<22} | {'Total (s)':>10} | {'Per-step (ms)':>13} | {'% of total':>10}"
    print(header)
    print("-" * len(header))
    for phase in PHASES:
        s = sum(timings[phase])
        print(f"{phase:<22} | {s:>10.3f} | {1000 * s / args.steps:>13.3f} | "
              f"{100 * s / total:>9.2f}%")
    print("-" * len(header))
    print(f"{'TOTAL':<22} | {total:>10.3f} | {1000 * total / args.steps:>13.3f} | "
          f"{100.0:>9.2f}%")
    print()
    print(f"Peak VRAM (torch allocator): {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
    print(f"Peak VRAM (nvidia-smi used): {nvidia_smi_used_mb():.1f} MB")
    print(f"Final TWM loss: {losses[-1]:.4f} (first {losses[0]:.4f}, "
          f"non-finite {sum(1 for v in losses if not np.isfinite(v))})")
    print(f"Tokenizer codes at end: {int(wm.tokenizer.num_codes.item())}")
    print(f"100k extrapolation: {total / args.steps * 100000 / 3600:.2f} h")

    if args.json_out:
        payload = {
            "args": vars(args),
            "phase_total_s": {p: sum(timings[p]) for p in PHASES},
            "phase_per_step_ms": {p: 1000 * sum(timings[p]) / args.steps for p in PHASES},
            "phase_pct": {p: 100 * sum(timings[p]) / total for p in PHASES},
            "total_s": total,
            "per_step_ms": 1000 * total / args.steps,
            "peak_torch_mb": torch.cuda.max_memory_allocated() / 1e6,
            "nvidia_smi_used_mb": nvidia_smi_used_mb(),
            "twm_loss_first": losses[0],
            "twm_loss_last": losses[-1],
            "tokenizer_codes": int(wm.tokenizer.num_codes.item()),
            "wm_params": n_wm_params,
        }
        pathlib.Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"json written to {args.json_out}")


if __name__ == "__main__":
    main()