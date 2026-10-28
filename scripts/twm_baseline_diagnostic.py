#!/usr/bin/env python3
"""TWM baseline diagnostic on Craftax-Classic-Pixels-v1 (1M env steps, seed 0).

Purpose: produce diagnostic data, not a reproduction of the 69.66% anchor.

Pipeline per iteration (B parallel envs x T env steps):
  1. batched Craftax rollout with the official TWM Agent policy
     (Impala -> Actor; same call sequence as Agent.interact_with_env, batched here)
  2. WM train step on recent real frames: forward(return_loss=True) + AdamW
  3. real trajectories -> official Agent.learn memories
  4. one official Agent.forward imagined trajectory (TWM Dyna)
  5. Agent.learn([real, imagined]) -- official policy/world-model coupling

Every 10k env steps: checkpoint WM + policy + optimizers, evaluate held-out
prediction error by mechanic, and run a short current-policy evaluation for
mechanic progress.

Documented deviations (all in this script; no third-party source edits):
  - Real-env interaction is batched here instead of Agent.interact_with_env,
    which is batch-size 1. The policy call sequence is identical:
    impala(frame) -> actor(features, sample_action=True).
  - Dyna ratio is one imagined trajectory (<=23 frames) per 1000 real steps.
    The paper's exact ratio is not exposed by the package; this is OUR choice.
  - Held-out set: 1000 fixed transition windows collected once with a random
    behaviour policy, balanced across mechanic buckets where possible.
  - Mechanic labels come directly from Craftax EnvState (inventory, achievements,
    player/mob positions), which is more direct than parsing the symbolic obs.
    Training still uses Pixels.

Usage:
  python scripts/twm_baseline_diagnostic.py --max-env-steps 1000000
  python scripts/twm_baseline_diagnostic.py --max-env-steps 20000 --holdout-size 80
  python scripts/twm_baseline_diagnostic.py --holdout-only
"""

import os

# Must be set before importing jax / torch.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import collections
import json
import pathlib
import subprocess
import time

import numpy as np

import jax
import jax.numpy as jnp

import torch
import torch.nn.functional as F

from craftax.craftax_env import make_craftax_env_from_name
from calibration_env import make_calibration_env

from improving_transformers_world_model import Agent, WorldModel
from improving_transformers_world_model.agent import (
    MemoriesWithNextState,
    Memory,
    calc_entropy,
    get_log_prob,
)
from improving_transformers_world_model.world_model import (
    NearestNeighborTokenizer,
    pack_one_with_inverse,
)

WORKSPACE = pathlib.Path(__file__).resolve().parents[1]
LOG_DIR = pathlib.Path.home() / "craftax-workspace" / "logs"
DATA_DIR = pathlib.Path.home() / "craftax-workspace" / "data"
CKPT_DIR = DATA_DIR / "diagnostic_checkpoints"

# Craftax achievement indices (22 flags; order verified from installed EnvState).
ACH_CRAFT = [0, 1, 3, 5, 6, 7, 9, 10, 13, 14, 16, 17, 18, 19, 20, 21]
ACH_COMBAT = [8, 12]
ACH_EAT = [2, 11]

BUCKETS = ["normal", "exploration", "combat", "crafting"]
BUCKET_TO_ID = {b: i for i, b in enumerate(BUCKETS)}
INV_NAMES = [
    "wood", "stone", "coal", "iron", "diamond", "sapling",
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
]


class CompatAgent(Agent):
    """Agent with a local critic-loss compatibility shim.

    improving-transformers-world-model 0.0.59 calls
    self.critic.to_value_pred.loss_fn in Agent.critic_loss. The installed
    hl-gauss-pytorch exposes HLGaussLayer.hl_gauss_loss and regress_loss_fn
    instead, so the stock method raises AttributeError. This subclass keeps the
    HL-Gauss critic and reproduces the upstream loss shape, but omits the
    upstream value-clipping branch (which passed scalar values to the HL-Gauss
    logits loss). Third-party source is untouched.

    It also corrects the sign of the PPO surrogate term. TWM 0.0.59 returns
    +min(surr1, surr2) - beta*entropy from Agent.policy_loss; minimizing that
    expression reduces the log-probability of positive-advantage actions and
    drives the policy to a collapsed arbitrary action. The corrected loss is
    -min(surr1, surr2) - beta*entropy. This is a harness-side compatibility fix;
    the installed package source is untouched.
    """

    def policy_loss(
        self,
        states,
        actions,
        old_log_probs,
        values,
        returns,
        world_model_embeds=None,
    ):
        self.actor.train()

        batch = values.shape[0]
        advantages = F.layer_norm(returns - values, (batch,))

        actor_critic_input, _ = self.impala(states)
        action_logits = self.actor(
            actor_critic_input, world_model_embed=world_model_embeds
        )

        prob = action_logits.softmax(dim=-1)
        log_probs = get_log_prob(action_logits, actions)
        ratios = (log_probs - old_log_probs).exp()

        clip = self.actor_eps_clip
        surr1 = ratios * advantages
        surr2 = ratios.clamp(1.0 - clip, 1.0 + clip) * advantages

        action_entropy = calc_entropy(prob)
        return -torch.min(surr1, surr2) - self.actor_beta_s * action_entropy

    def critic_loss(self, states, old_values, returns):
        self.critic.train()
        actor_critic_input, _ = self.impala(states)
        critic_value, critic_logits = self.critic(
            actor_critic_input, return_value_and_logits=True
        )
        layer = self.critic.to_value_pred
        if getattr(layer, "use_classification", True):
            loss = layer.hl_gauss_loss(critic_logits, returns, reduction="none")
        else:
            loss = layer.regress_loss_fn(critic_value, returns, reduction="none")
        return loss.mean()


class StableNearestNeighborTokenizer(NearestNeighborTokenizer):
    """Local workaround for improving-transformers-world-model 0.0.59.

    Upstream NearestNeighborTokenizer.forward calls
    masked_fill_(mask, new_code_ids) with a 1-D fill value, which torch
    rejects when two or more new patches are found in one batch. This subclass
    reproduces the upstream logic and replaces only that assignment. The
    third-party package is untouched. Same workaround as scripts/integration_smoke.py.
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
    p = argparse.ArgumentParser(description="TWM Craftax baseline diagnostic")
    p.add_argument("--env", default="Craftax-Classic-Pixels-v1")
    p.add_argument(
        "--variant",
        choices=("baseline", "reward_shaping", "short_episode", "curriculum"),
        default="baseline",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-env-steps", type=int, default=1_000_000)
    p.add_argument("--envs", type=int, default=50, help="parallel envs")
    p.add_argument("--rollout-steps", type=int, default=20, help="env steps/iteration")
    p.add_argument("--checkpoint-every", type=int, default=10_000)
    p.add_argument("--eval-envs", type=int, default=8)
    p.add_argument("--eval-horizon", type=int, default=250)
    p.add_argument("--eval-seed", type=int, default=4242)
    p.add_argument("--wm-batch", type=int, default=8)
    p.add_argument("--wm-time", type=int, default=8)
    p.add_argument("--wm-lr", type=float, default=1e-4)
    p.add_argument("--wm-grad-steps", type=int, default=1)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--reward-bins", type=int, default=65)
    p.add_argument("--max-codes", type=int, default=100_000,
                   help="tokenizer codebook cap (lower only under memory pressure)")
    p.add_argument("--policy-batch", type=int, default=128)
    p.add_argument("--policy-epochs", type=int, default=1)
    p.add_argument("--imagine-steps", type=int, default=24)
    p.add_argument("--imagine-every", type=int, default=1)
    p.add_argument("--holdout-size", type=int, default=1000)
    p.add_argument("--holdout-per-bucket", type=int, default=250)
    p.add_argument("--holdout-max-steps", type=int, default=200_000)
    p.add_argument("--holdout-envs", type=int, default=32)
    p.add_argument("--holdout-only", action="store_true")
    p.add_argument("--skip-holdout-eval", action="store_true")
    p.add_argument("--holdout-cache", default=str(DATA_DIR / "diagnostic_holdout.npz"))
    p.add_argument("--json-out", default=str(LOG_DIR / "twm_baseline_diagnostic.json"))
    p.add_argument("--ckpt-dir", default=str(CKPT_DIR))
    p.add_argument("--no-save-checkpoints", action="store_true")
    p.add_argument("--debug-timing", action="store_true")
    return p.parse_args()


def build_env(args, initial_env_steps=0, num_envs=None):
    num_envs = args.envs if num_envs is None else num_envs
    env = make_calibration_env(
        args.env,
        auto_reset=True,
        variant=args.variant,
        num_envs=num_envs,
        initial_env_steps=initial_env_steps,
    )
    params = env.default_params
    num_actions = int(env.action_space(params).n)
    obs_shape = tuple(env.observation_space(params).shape)
    if obs_shape != (63, 63, 3):
        raise SystemExit(f"expected (63,63,3) obs, got {obs_shape}")
    if num_actions != 17:
        raise SystemExit(f"expected 17 actions for Craftax-Classic, got {num_actions}")

    def compact_metrics(state, done):
        if hasattr(state, "env_state"):
            state = state.env_state
        inv = state.inventory
        inv_vec = jnp.stack([
            inv.wood, inv.stone, inv.coal, inv.iron, inv.diamond, inv.sapling,
            inv.wood_pickaxe, inv.stone_pickaxe, inv.iron_pickaxe,
            inv.wood_sword, inv.stone_sword, inv.iron_sword,
        ]).astype(jnp.int32)

        tier = jnp.where(
            inv.diamond > 0, 4,
            jnp.where((inv.iron_pickaxe > 0) | (inv.iron_sword > 0), 3,
                      jnp.where((inv.stone_pickaxe > 0) | (inv.stone_sword > 0), 2,
                                jnp.where((inv.wood_pickaxe > 0) | (inv.wood_sword > 0), 1, 0))))
        tier = tier.astype(jnp.int32)

        def min_hostile_dist(mobs):
            d = jnp.max(jnp.abs(mobs.position - state.player_position), axis=-1).astype(jnp.float32)
            alive = mobs.mask & (mobs.health > 0)
            return jnp.min(jnp.where(alive, d, 999.0))

        hostile_dist = jnp.minimum(min_hostile_dist(state.zombies),
                                   min_hostile_dist(state.skeletons))
        mobs_alive = (state.zombies.mask.sum() + state.skeletons.mask.sum()
                      + state.cows.mask.sum()).astype(jnp.int32)

        return dict(
            pos=state.player_position.astype(jnp.int32),
            health=state.player_health.astype(jnp.float32),
            food=state.player_food.astype(jnp.float32),
            drink=state.player_drink.astype(jnp.float32),
            energy=state.player_energy.astype(jnp.float32),
            timestep=state.timestep.astype(jnp.int32),
            inv=inv_vec,
            tier=tier,
            ach=state.achievements,
            hostile_dist=hostile_dist,
            hostile_near=(hostile_dist <= 2.0),
            mobs_alive=mobs_alive,
            done=done,
        )

    def raw_step(key, state, action):
        obs, state, reward, done, _info = env.step(key, state, action, params)
        return obs, state, reward, done, compact_metrics(state, done)

    def reset_one(key):
        return env.reset(key, params)

    step_batched = jax.jit(jax.vmap(raw_step, in_axes=(0, 0, 0)))

    @jax.jit
    def reset_batched(keys):
        return jax.vmap(reset_one)(keys)

    return env, params, num_actions, step_batched, reset_batched


def tree_stack(seq):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *seq)


def tree_numpy(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), tree)


def update_visited(visited, positions):
    """visited: (B,64,64) bool numpy; positions: (B,2) [x,y]."""
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
    """Integer labels (B,) for the target frame of each transition.

    Priority: crafting > combat > exploration > normal.
    """
    inv_delta = np.abs(cur["inv"] - prev["inv"]).sum(axis=-1)
    ach_new = cur["ach"] & ~prev["ach"]

    craft = (inv_delta > 0) | ach_new[:, ACH_CRAFT].any(axis=-1)
    combat = cur["hostile_near"] | ach_new[:, ACH_COMBAT].any(axis=-1)

    labels = np.zeros(inv_delta.shape[0], dtype=np.int8)
    labels[new_tile] = 1
    labels[combat] = 2
    labels[craft] = 3
    return labels


def obs_to_torch(obs_np, device):
    t = torch.from_numpy(np.array(obs_np, copy=True)).to(device, non_blocking=False)
    return t.permute(0, 3, 1, 2).float().contiguous()


def policy_act(agent, obs_np, device, return_logits=False):
    with torch.no_grad():
        x = obs_to_torch(obs_np, device)
        feat, _ = agent.impala(x)
        actions, log_probs = agent.actor(feat, sample_action=True)
        if not return_logits:
            return actions, log_probs
        logits = agent.actor(feat)
    return actions, log_probs, logits


def build_agent(num_actions, device):
    return CompatAgent(
        impala=dict(
            dims=(64, 64, 128), image_size=63, channels=3,
            init_conv_kernel=7, dim_rnn=32, dim_rnn_hidden=32,
        ),
        actor=dict(dim=512, num_actions=num_actions, num_layers=3, expansion_factor=2.0),
        critic=dict(
            dim=512, num_layers=4, expansion_factor=2.0, use_regression=False,
            hl_gauss_loss_kwargs=dict(min_value=-10.0, max_value=10.0, num_bins=64, sigma=0.5),
        ),
        actor_lr=1e-4,
        critic_lr=1e-4,
        max_memories=128_000,
    ).to(device)


def build_world_model(args, num_actions, device):
    wm = WorldModel(
        image_size=63,
        patch_size=7,
        channels=3,
        num_actions=num_actions,
        reward_min_value=-1.0,
        reward_max_value=1.0,
        reward_num_bins=args.reward_bins,
        transformer=dict(dim=args.dim, depth=args.depth, block_size=81),
        tokenizer=StableNearestNeighborTokenizer(
            dim=7 * 7 * 3, distance_threshold=0.5, max_codes=args.max_codes
        ),
    ).to(device)
    opt = torch.optim.AdamW(wm.parameters(), lr=args.wm_lr)
    return wm, opt


def training_rollout(args, agent, device, step_batched, obs, state, rng):
    frames = []
    action_list = []
    logp_list = []
    reward_list = []
    done_list = []
    metrics_list = []

    for _ in range(args.rollout_steps):
        obs_np = np.asarray(jax.device_get(obs))
        actions, log_probs = policy_act(agent, obs_np, device)
        act_np = actions.detach().cpu().numpy().astype(np.int32)

        rng, key = jax.random.split(rng)
        next_obs, state, reward, done, metrics = step_batched(
            jax.random.split(key, args.envs), state, jnp.asarray(act_np)
        )

        frames.append(torch.from_numpy(np.array(obs_np, copy=True)).to(device).permute(0, 3, 1, 2).float().contiguous())
        action_list.append(actions.detach())
        logp_list.append(log_probs.detach())
        reward_list.append(reward)
        done_list.append(done)
        metrics_list.append(metrics)
        obs = next_obs

    frames_t = torch.stack(frames, dim=2)
    actions_t = torch.stack(action_list, dim=1)
    logp_t = torch.stack(logp_list, dim=1)
    rewards_t = jnp.stack(reward_list).T
    dones_t = jnp.stack(done_list).T
    metrics_t = tree_stack(metrics_list)
    next_obs_np = np.asarray(jax.device_get(obs))
    next_frames_t = (
        torch.from_numpy(np.array(next_obs_np, copy=True))
        .to(device)
        .permute(0, 3, 1, 2)
        .float()
        .contiguous()
    )
    return (
        frames_t, actions_t, logp_t, rewards_t, dones_t, metrics_t,
        next_frames_t, obs, state, rng,
    )


def frames_to_wm_batch(frames_t, actions_t, rewards_np, dones_np, batch, seq, rng):
    B, C, T, H, W = frames_t.shape
    if seq > T:
        raise SystemExit(f"wm time {seq} > rollout steps {T}")
    b_idx = rng.integers(0, B, size=batch)
    t_idx = rng.integers(0, T - seq + 1, size=batch)

    x = torch.stack([frames_t[b_idx[i], :, t_idx[i]:t_idx[i] + seq] for i in range(batch)])
    a = torch.stack([
        actions_t[b_idx[i], t_idx[i]:t_idx[i] + seq] for i in range(batch)
    ]).long()[..., None]
    r = torch.as_tensor(
        np.stack([rewards_np[b_idx[i], t_idx[i]:t_idx[i] + seq] for i in range(batch)]),
        device=frames_t.device, dtype=torch.float32,
    )
    d = torch.as_tensor(
        np.stack([dones_np[b_idx[i], t_idx[i]:t_idx[i] + seq] for i in range(batch)]),
        device=frames_t.device, dtype=torch.bool,
    )
    return x, r, a, d


def compute_values(agent, frames_t, chunk=128):
    B, C, T, H, W = frames_t.shape
    flat = frames_t.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    feats = []
    with torch.no_grad():
        for start in range(0, flat.shape[0], chunk):
            feat, _ = agent.impala(flat[start:start + chunk])
            feats.append(feat)
        feat = torch.cat(feats, dim=0)
        values = agent.critic(feat).reshape(B, T)
    return values


def build_real_memories(
    frames_t, actions_t, logp_t, rewards_np, dones_np, values_t, next_frames_t
):
    B, C, T, H, W = frames_t.shape
    frames_cpu = frames_t.detach().cpu()
    next_frames_cpu = next_frames_t.detach().cpu()
    actions_cpu = actions_t.detach().cpu().long()
    logp_cpu = logp_t.detach().cpu().float()
    values_cpu = values_t.detach().cpu().float()
    rewards_cpu = torch.tensor(rewards_np, dtype=torch.float32)
    dones_cpu = torch.tensor(dones_np, dtype=torch.bool)

    out = []
    for b in range(B):
        mems = []
        for t in range(T):
            mems.append(Memory(
                state=frames_cpu[b, :, t],
                action=actions_cpu[b, t],
                action_log_prob=logp_cpu[b, t],
                reward=rewards_cpu[b, t],
                value=values_cpu[b, t],
                done=dones_cpu[b, t],
                world_embed=torch.empty(0),
            ))
        # frames_t ends at the state before the final action. The post-action
        # observation (already auto-reset if terminal) is the true bootstrap
        # state for the last memory in this rollout.
        next_state = next_frames_cpu[b].reshape(C, 1, H, W)
        out.append(MemoriesWithNextState(
            memories=collections.deque(mems),
            next_state=next_state,
            from_real_env=True,
            has_world_model_embed=False,
        ))
    return out


def summarize_rollout(metrics_np, rewards_np):
    T, B = metrics_np["pos"].shape[:2]
    visited = np.zeros((B, 64, 64), dtype=bool)
    mark_initial(visited, metrics_np["pos"][0])

    counts = {b: 0 for b in BUCKETS}
    prev = {k: v[0] for k, v in metrics_np.items()}
    coverage_peak = np.zeros(B)

    for t in range(1, T):
        cur = {k: v[t] for k, v in metrics_np.items()}
        new_tile = update_visited(visited, cur["pos"])
        labels = classify_step(prev, cur, new_tile)
        for lab in labels:
            counts[BUCKETS[int(lab)]] += 1

        done = cur["done"]
        if done.any():
            cov = visited.sum(axis=(1, 2)) / float(64 * 64)
            coverage_peak = np.maximum(coverage_peak, cov)
            visited[done] = False
        prev = cur

    cov = visited.sum(axis=(1, 2)) / float(64 * 64)
    coverage_peak = np.maximum(coverage_peak, cov)

    ach_max = metrics_np["ach"].max(axis=0)
    inv_max = metrics_np["inv"].max(axis=0).max(axis=0)
    return dict(
        reward_mean=float(rewards_np.mean()),
        reward_last=float(rewards_np[:, -1].mean()),
        craft_tier_max=int(metrics_np["tier"].max()),
        craft_tier_mean=float(metrics_np["tier"].mean()),
        achievements_max=int(ach_max.sum(axis=1).max()),
        achievements_mean=float(ach_max.sum(axis=1).mean()),
        kills_max=int(ach_max[:, ACH_COMBAT].sum(axis=1).max()),
        kills_mean=float(ach_max[:, ACH_COMBAT].sum(axis=1).mean()),
        resource_max={n: int(v) for n, v in zip(INV_NAMES, inv_max)},
        exploration_coverage_mean=float(coverage_peak.mean()),
        exploration_coverage_max=float(coverage_peak.max()),
        timestep_max_mean=float(metrics_np["timestep"].max(axis=0).mean()),
        event_counts=counts,
    )


def eval_policy(agent, args, step_batched, reset_batched, device, rng,
                num_envs=None, horizon=None):
    num_envs = num_envs or args.eval_envs
    horizon = horizon or args.eval_horizon

    rng, sub = jax.random.split(rng)
    obs, state = reset_batched(jax.random.split(sub, num_envs))
    visited = np.zeros((num_envs, 64, 64), dtype=bool)
    env_state = state.env_state if hasattr(state, "env_state") else state
    mark_initial(visited, np.asarray(env_state.player_position))
    coverage_peak = np.zeros(num_envs)

    prev = None
    episode_lengths = []
    current_age = np.zeros(num_envs, dtype=np.int64)
    rewards = []
    metrics_list = []
    counts = {b: 0 for b in BUCKETS}
    action_counts = np.zeros(17, dtype=np.int64)
    mode_counts = np.zeros(17, dtype=np.int64)
    policy_entropies = []

    for _ in range(horizon):
        obs_np = np.asarray(jax.device_get(obs))
        actions, _, logits = policy_act(agent, obs_np, device, return_logits=True)
        act_np = actions.detach().cpu().numpy().astype(np.int32)
        action_counts += np.bincount(act_np, minlength=17)
        with torch.no_grad():
            probs = logits.softmax(dim=-1)
            mode_counts += np.bincount(
                probs.argmax(dim=-1).detach().cpu().numpy().astype(np.int32),
                minlength=17,
            )
            policy_entropies.append(
                np.asarray(
                    calc_entropy(probs).detach().cpu().numpy(), dtype=np.float32
                )
            )
        rng, key = jax.random.split(rng)
        next_obs, state, reward, done, metrics = step_batched(
            jax.random.split(key, num_envs), state, jnp.asarray(act_np)
        )
        cur = tree_numpy(metrics)
        new_tile = update_visited(visited, cur["pos"])

        if prev is not None:
            labels = classify_step(prev, cur, new_tile)
            for lab in labels:
                counts[BUCKETS[int(lab)]] += 1

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

    stacked = {k: np.stack([m[k] for m in metrics_list], axis=0) for k in metrics_list[0]}
    flat = {k: v.reshape((-1,) + v.shape[2:]) for k, v in stacked.items()}
    ach_max = stacked["ach"].max(axis=0)
    inv_max = stacked["inv"].max(axis=0).max(axis=0)

    action_probs = action_counts / max(int(action_counts.sum()), 1)
    action_entropy = float(-np.sum(action_probs[action_probs > 0] *
                                   np.log(action_probs[action_probs > 0])))
    mode_probs = mode_counts / max(int(mode_counts.sum()), 1)
    mode_entropy = float(-np.sum(mode_probs[mode_probs > 0] *
                                 np.log(mode_probs[mode_probs > 0])))
    total_actions = max(int(action_counts.sum()), 1)
    all_lengths = episode_lengths + [int(v) for v in current_age if v > 0]

    return dict(
        reward_mean=float(np.mean(np.stack(rewards))),
        reward_last100_mean=float(np.mean(np.stack(rewards)[-100:])),
        craft_tier_max=int(flat["tier"].max()),
        craft_tier_mean=float(flat["tier"].mean()),
        achievements_max=int(ach_max.sum(axis=1).max()),
        achievements_mean=float(ach_max.sum(axis=1).mean()),
        kills_max=int(ach_max[:, ACH_COMBAT].sum(axis=1).max()),
        kills_mean=float(ach_max[:, ACH_COMBAT].sum(axis=1).mean()),
        resource_max={n: int(v) for n, v in zip(INV_NAMES, inv_max)},
        exploration_coverage_mean=float(coverage_peak.mean()),
        exploration_coverage_max=float(coverage_peak.max()),
        survival_steps_mean=float(np.mean(episode_lengths)) if episode_lengths else float("nan"),
        survival_steps_max=float(np.max(episode_lengths)) if episode_lengths else float("nan"),
        survival_censored_mean=float(np.mean(all_lengths)) if all_lengths else float("nan"),
        survival_censored_max=float(np.max(all_lengths)) if all_lengths else float("nan"),
        final_timestep_mean=float(flat["timestep"].max(axis=0).mean()),
        episodes_finished=int(len(episode_lengths)),
        action_counts=[int(v) for v in action_counts],
        mode_counts=[int(v) for v in mode_counts],
        noop_fraction=float(action_counts[0] / total_actions),
        mode_noop_fraction=float(mode_counts[0] / total_actions),
        action_entropy=action_entropy,
        mode_entropy=mode_entropy,
        policy_entropy_mean=float(
            np.mean(np.concatenate(policy_entropies, axis=0))
        ),
        event_counts=counts,
        steps=num_envs * horizon,
    )


def collect_holdout(args, step_batched, reset_batched, device, seed=12345):
    cache = pathlib.Path(args.holdout_cache)
    if cache.exists():
        data = np.load(cache)
        print(f"holdout: loaded cache {cache} ({len(data['labels'])} windows)")
        return {k: data[k] for k in data.files}

    print("holdout: collecting fixed evaluation windows with a random policy ...", flush=True)
    rng = jax.random.PRNGKey(seed)
    n_envs = args.holdout_envs
    rng, sub = jax.random.split(rng)
    obs, state = reset_batched(jax.random.split(sub, n_envs))
    visited = np.zeros((n_envs, 64, 64), dtype=bool)
    env_state = state.env_state if hasattr(state, "env_state") else state
    mark_initial(visited, np.asarray(env_state.player_position))

    pool = {b: [] for b in BUCKETS}
    prev_obs = None
    prev_metrics = None
    prev_action = None
    prev_reward = None
    steps = 0
    max_per = args.holdout_per_bucket

    while steps < args.holdout_max_steps and any(len(pool[b]) < max_per for b in BUCKETS):
        obs_np = np.asarray(jax.device_get(obs))
        rng, ka, ks = jax.random.split(rng, 3)
        act_np = np.asarray(jax.vmap(
            lambda k: jax.random.randint(k, (), 0, 17)
        )(jax.random.split(ka, n_envs)))
        next_obs, state, reward, done, metrics = step_batched(
            jax.random.split(ks, n_envs), state, jnp.asarray(act_np)
        )
        cur = tree_numpy(metrics)
        new_tile = update_visited(visited, cur["pos"])
        reward_np = np.asarray(jax.device_get(reward))
        done_np = np.asarray(jax.device_get(done))

        if prev_obs is not None:
            labels = classify_step(prev_metrics, cur, new_tile)
            for i in range(n_envs):
                if done_np[i]:
                    continue
                label = BUCKETS[int(labels[i])]
                if len(pool[label]) < max_per:
                    pool[label].append(dict(
                        prev_obs=prev_obs[i].copy(),
                        obs=obs_np[i].copy(),
                        prev_action=int(prev_action[i]),
                        prev_reward=float(prev_reward[i]),
                        label=label,
                    ))

        if done_np.any():
            visited[done_np] = False
        prev_obs = obs_np
        prev_metrics = cur
        prev_action = act_np
        prev_reward = reward_np
        obs = next_obs
        steps += n_envs

        if steps % (n_envs * 50) == 0:
            print(f"  holdout {steps:7d} steps | " +
                  " ".join(f"{b}={len(pool[b])}" for b in BUCKETS), flush=True)

    total_want = min(args.holdout_size, sum(len(pool[b]) for b in BUCKETS))
    per_bucket = min(max_per, total_want // len(BUCKETS))
    selected = []
    for b in BUCKETS:
        selected.extend(pool[b][:per_bucket])
    remaining = total_want - len(selected)
    for b in BUCKETS:
        extra = pool[b][per_bucket:per_bucket + remaining]
        selected.extend(extra)
        remaining -= len(extra)
        if remaining <= 0:
            break

    print("holdout selected: " + " ".join(
        f"{b}={sum(1 for s in selected if s['label'] == b)}" for b in BUCKETS), flush=True)

    x = np.stack([
        np.stack([s["prev_obs"], s["obs"]], axis=0) for s in selected
    ]).transpose(0, 4, 1, 2, 3)
    actions = np.stack([
        np.array([s["prev_action"], 0], dtype=np.int64) for s in selected
    ])[:, :, None]
    rewards = np.stack([
        np.array([s["prev_reward"], 0.0], dtype=np.float32) for s in selected
    ])
    dones = np.zeros_like(rewards, dtype=bool)
    labels = np.array([BUCKET_TO_ID[s["label"]] for s in selected], dtype=np.int8)

    data = dict(x=x, actions=actions, rewards=rewards, dones=dones, labels=labels)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **data)
    print(f"holdout: saved {len(selected)} windows to {cache}", flush=True)
    return data


def evaluate_holdout(wm, data, device, chunk=32):
    x_all = torch.as_tensor(data["x"], device=device, dtype=torch.float32)
    a_all = torch.as_tensor(data["actions"], device=device, dtype=torch.long)
    r_all = torch.as_tensor(data["rewards"], device=device, dtype=torch.float32)
    d_all = torch.as_tensor(data["dones"], device=device, dtype=torch.bool)
    labels = data["labels"]

    wm.eval()
    out = {}
    with torch.no_grad():
        for bi, bucket in enumerate(BUCKETS):
            idx = np.nonzero(labels == bi)[0]
            if len(idx) == 0:
                out[bucket] = dict(n=0, state_loss=None, reward_loss=None,
                                   terminal_loss=None, total_loss=None)
                continue
            sums = np.zeros(4, dtype=np.float64)
            n = 0
            for start in range(0, len(idx), chunk):
                sel = idx[start:start + chunk]
                total, (state_loss, reward_loss, term_loss) = wm(
                    x_all[sel], rewards=r_all[sel], actions=a_all[sel],
                    is_terminal=d_all[sel], return_loss_breakdown=True,
                )
                b = len(sel)
                sums += b * np.array([
                    float(state_loss.detach()),
                    float(reward_loss.detach()) if reward_loss is not None else np.nan,
                    float(term_loss.detach()) if term_loss is not None else np.nan,
                    float(total.detach()),
                ])
                n += b
            vals = sums / max(n, 1)
            out[bucket] = dict(
                n=n,
                state_loss=float(vals[0]),
                reward_loss=float(vals[1]) if np.isfinite(vals[1]) else None,
                terminal_loss=float(vals[2]) if np.isfinite(vals[2]) else None,
                total_loss=float(vals[3]),
            )
    wm.train()
    return out


def nvidia_smi_used_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        return float("nan")


def save_checkpoint(path, wm, wm_opt, agent, env_steps, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(
        env_steps=env_steps,
        wm=wm.state_dict(),
        wm_opt=wm_opt.state_dict(),
        impala=agent.impala.state_dict(),
        actor=agent.actor.state_dict(),
        critic=agent.critic.state_dict(),
        actor_optim=agent.actor_optim.state_dict(),
        critic_optim=agent.critic_optim.state_dict(),
        meta=meta,
    ), path)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main():
    args = parse_args()
    if "Symbolic" in args.env:
        raise SystemExit(
            "UNSUPPORTED: TWM WorldModel and Impala in this harness require "
            "63x63x3 pixels; Craftax symbolic observations are flat vectors."
        )
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.variant != "baseline" and args.holdout_cache == str(DATA_DIR / "diagnostic_holdout.npz"):
        args.holdout_cache = str(DATA_DIR / f"diagnostic_holdout_{args.variant}.npz")
    ckpt_dir = pathlib.Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA is required for this diagnostic")

    print("=" * 78)
    print("TWM baseline diagnostic")
    print("=" * 78)
    print(f"env={args.env} seed={args.seed} max_env_steps={args.max_env_steps}")
    print(f"envs={args.envs} rollout_steps={args.rollout_steps} "
          f"({args.envs * args.rollout_steps} env steps/iteration)")
    print(f"torch={torch.__version__} jax={jax.__version__} devices={jax.devices()}")
    print(flush=True)

    env, params, num_actions, step_batched, reset_batched = build_env(args)
    print(f"env obs=(63,63,3) actions={num_actions}", flush=True)

    if args.variant == "curriculum":
        _, _, _, eval_step_batched, eval_reset_batched = build_env(
            args,
            initial_env_steps=args.max_env_steps,
            num_envs=args.eval_envs,
        )
    else:
        eval_step_batched, eval_reset_batched = step_batched, reset_batched

    holdout = collect_holdout(args, step_batched, reset_batched, device)
    if args.holdout_only:
        counts = {b: int((holdout["labels"] == BUCKET_TO_ID[b]).sum()) for b in BUCKETS}
        print("holdout-only counts:", counts)
        return

    agent = build_agent(num_actions, device)
    wm, wm_opt = build_world_model(args, num_actions, device)
    print(f"WM params={sum(p.numel() for p in wm.parameters()) / 1e6:.2f}M", flush=True)

    rng = jax.random.PRNGKey(args.seed)
    rng, sub = jax.random.split(rng)
    obs, state = reset_batched(jax.random.split(sub, args.envs))

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    iterations = args.max_env_steps // (args.envs * args.rollout_steps)
    if iterations < 1:
        raise SystemExit("max-env-steps smaller than one iteration")

    history = dict(
        config=vars(args),
        checkpoints=[],
        iteration_times=[],
        wm_losses=[],
        reward_means=[],
        event_counts=[],
    )
    env_steps = 0
    next_ckpt = args.checkpoint_every
    wm_loss_recent = collections.deque(maxlen=50)

    for it in range(1, iterations + 1):
        it0 = time.perf_counter()
        phases = {}

        (frames_t, actions_t, logp_t, rewards_t, dones_t, metrics_t,
         next_frames_t, obs, state, rng) = training_rollout(
            args, agent, device, step_batched, obs, state, rng
        )
        phases["rollout"] = time.perf_counter() - it0
        env_steps += args.envs * args.rollout_steps

        t0 = time.perf_counter()
        rewards_np = np.asarray(jax.device_get(rewards_t))
        dones_np = np.asarray(jax.device_get(dones_t))
        metrics_np = tree_numpy(metrics_t)
        rollout_summary = summarize_rollout(metrics_np, rewards_np)
        phases["metrics"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        values_t = compute_values(agent, frames_t)
        phases["values"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        wm.train()
        loss_val = float("nan")
        for _ in range(args.wm_grad_steps):
            x, r, a, d = frames_to_wm_batch(
                frames_t, actions_t, rewards_np, dones_np,
                args.wm_batch, args.wm_time, np.random.default_rng(args.seed + it),
            )
            wm_opt.zero_grad(set_to_none=True)
            loss = wm(x, rewards=r, actions=a, is_terminal=d)
            loss.backward()
            wm_opt.step()
            loss_val = float(loss.detach())
        phases["wm_train"] = time.perf_counter() - t0
        wm_loss_recent.append(loss_val)

        t0 = time.perf_counter()
        real_mems = build_real_memories(
            frames_t, actions_t, logp_t, rewards_np, dones_np, values_t,
            next_frames_t,
        )
        phases["memories"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        mem_sets = list(real_mems)
        if args.imagine_every > 0 and (it % args.imagine_every == 0):
            init_idx = int(torch.randint(0, args.envs, (1,)).item())
            init_state = frames_t[init_idx, :, -1]
            imagined = agent.forward(wm, init_state, max_steps=args.imagine_steps)
            mem_sets.append(imagined)
        phases["imagine"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        agent.learn(
            mem_sets,
            batch_size=args.policy_batch,
            epochs=args.policy_epochs,
        )
        phases["learn"] = time.perf_counter() - t0
        if args.debug_timing:
            print("    phases " + " ".join(f"{k}={v:.2f}s" for k, v in phases.items()), flush=True)

        it_time = time.perf_counter() - it0
        history["iteration_times"].append(it_time)
        history["wm_losses"].append(loss_val)
        history["reward_means"].append(rollout_summary["reward_mean"])
        history["event_counts"].append(rollout_summary["event_counts"])

        if it % 10 == 0 or it == iterations:
            elapsed = time.perf_counter() - start
            print(
                f"iter {it:5d}/{iterations} | steps {env_steps:8d} | "
                f"wm_loss {np.mean(list(wm_loss_recent)):7.4f} | "
                f"reward {rollout_summary['reward_mean']:+.4f} | "
                f"it {it_time:6.2f}s | elapsed {elapsed/60:7.2f} min | "
                f"peak {torch.cuda.max_memory_allocated()/1e6:7.1f} MB | "
                f"codes {int(wm.tokenizer.num_codes.item())} | "
                f"tier {rollout_summary['craft_tier_max']} | "
                f"ach {rollout_summary['achievements_max']}",
                flush=True,
            )

        if env_steps >= next_ckpt:
            t_ck = time.perf_counter()
            print(f"\n--- checkpoint @ {env_steps} env steps ---", flush=True)
            eval_rng = jax.random.fold_in(
                jax.random.PRNGKey(args.eval_seed), env_steps
            )
            mech = eval_policy(
                agent,
                args,
                eval_step_batched,
                eval_reset_batched,
                device,
                eval_rng,
            )
            hold = None if args.skip_holdout_eval else evaluate_holdout(wm, holdout, device)
            wall = time.perf_counter() - start
            record = dict(
                env_steps=env_steps,
                wall_s=wall,
                peak_torch_mb=torch.cuda.max_memory_allocated() / 1e6,
                peak_nvidia_smi_mb=nvidia_smi_used_mb(),
                tokenizer_codes=int(wm.tokenizer.num_codes.item()),
                wm_loss_mean_last50=float(np.mean(list(wm_loss_recent))),
                reward_mean_last_iter=rollout_summary["reward_mean"],
                train_mechanics=rollout_summary,
                eval_mechanics=mech,
                holdout=hold,
                checkpoint=None,
            )
            if not args.no_save_checkpoints:
                path = ckpt_dir / f"step_{env_steps:08d}.pt"
                save_checkpoint(path, wm, wm_opt, agent, env_steps, record)
                record["checkpoint"] = str(path)
            history["checkpoints"].append(record)
            write_json(pathlib.Path(args.json_out), history)
            print(
                f"    eval: reward {mech['reward_mean']:+.4f} | "
                f"tier {mech['craft_tier_max']} | ach {mech['achievements_max']} | "
                f"kills {mech['kills_max']} | "
                f"coverage {mech['exploration_coverage_mean']:.3f} | "
                f"survival {mech['survival_steps_mean']:.1f} | "
                f"noop {mech['noop_fraction']:.1%} | "
                f"entropy {mech['policy_entropy_mean']:.2f}",
                flush=True,
            )
            if hold is not None:
                print(
                    "    holdout state loss: " + " | ".join(
                        f"{b} {hold[b]['state_loss']:.3f} (n={hold[b]['n']})"
                        if hold[b]['state_loss'] is not None else f"{b} n/a (n={hold[b]['n']})"
                        for b in BUCKETS
                    ),
                    flush=True,
                )
            print(f"    ckpt wall {time.perf_counter()-t_ck:.1f}s\n", flush=True)
            next_ckpt = (env_steps // args.checkpoint_every + 1) * args.checkpoint_every

    total = time.perf_counter() - start
    print("=" * 78)
    print(f"DONE {env_steps} env steps in {total/60:.2f} min "
          f"({total/3600:.2f} h), peak torch {torch.cuda.max_memory_allocated()/1e6:.1f} MB")
    print(f"json: {args.json_out}")
    print("=" * 78)


if __name__ == "__main__":
    main()
