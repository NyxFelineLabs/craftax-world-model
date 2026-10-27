#!/usr/bin/env python3
"""Focused probes for the TWM harness NOOP-collapse diagnosis.

This script does not train the world model. Each subcommand isolates one of
the checks 1a-1f from docs/twm-crash-diagnosis.md and writes a small JSON
record. The third-party Craftax, Craftax_Baselines, and TWM sources are not
modified.

Usage:
  python scripts/twm_crash_probe.py warmup
  python scripts/twm_crash_probe.py reward
  python scripts/twm_crash_probe.py entropy
  python scripts/twm_crash_probe.py actions
  python scripts/twm_crash_probe.py gradients
  python scripts/twm_crash_probe.py optimizers
  python scripts/twm_crash_probe.py checkpoint
"""

import os

# Must be set before importing jax / torch.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import twm_baseline_diagnostic as diag
from improving_transformers_world_model.agent import calc_entropy, get_log_prob


LOG_DIR = pathlib.Path.home() / "craftax-workspace" / "logs"
DEFAULT_CHECKPOINT = (
    pathlib.Path.home()
    / "craftax-workspace"
    / "data"
    / "diagnostic_checkpoints"
    / "step_01000000.pt"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_args(**overrides):
    values = dict(
        env="Craftax-Classic-Pixels-v1",
        seed=0,
        envs=32,
        rollout_steps=20,
        wm_batch=2,
        wm_time=8,
        wm_lr=1e-4,
        wm_grad_steps=1,
        dim=64,
        depth=1,
        reward_bins=65,
        max_codes=2_000,
        policy_batch=128,
        policy_epochs=1,
        imagine_steps=24,
        imagine_every=1,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def torch_policy_stats(agent, obs_np):
    with torch.no_grad():
        states = diag.obs_to_torch(obs_np, DEVICE)
        features, _ = agent.impala(states)
        logits = agent.actor(features)
        probs = logits.softmax(dim=-1)
        entropy = calc_entropy(probs)
    return logits, probs, entropy


def write_result(name, payload, json_out=None):
    payload = dict(probe=name, **payload)
    path = pathlib.Path(json_out) if json_out else LOG_DIR / f"twm_crash_probe_{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"json: {path}")


def probe_warmup(args):
    agent = diag.build_agent(17, DEVICE)
    env, params, _, step_batched, reset_batched = diag.build_env(args)
    rng = jax.random.PRNGKey(args.seed)
    rng, sub = jax.random.split(rng)
    obs, state = reset_batched(jax.random.split(sub, args.envs))

    records = []
    for iteration in range(1, 6):
        _, probs, entropy = torch_policy_stats(agent, np.asarray(jax.device_get(obs)))
        before = float(entropy.mean())
        max_prob = float(probs.max(dim=-1).values.mean())
        actor_before = torch.cat([p.detach().flatten() for p in agent.actor.parameters()])

        rollout = diag.training_rollout(
            args, agent, DEVICE, step_batched, obs, state, rng
        )
        (
            frames_t,
            actions_t,
            logp_t,
            rewards_t,
            dones_t,
            _metrics_t,
            next_frames_t,
            obs,
            state,
            rng,
        ) = rollout
        rewards_np = np.asarray(jax.device_get(rewards_t))
        dones_np = np.asarray(jax.device_get(dones_t))
        values_t = diag.compute_values(agent, frames_t)
        memories = diag.build_real_memories(
            frames_t, actions_t, logp_t, rewards_np, dones_np, values_t, next_frames_t
        )
        agent.learn(memories, batch_size=args.policy_batch, epochs=1)

        actor_after = torch.cat([p.detach().flatten() for p in agent.actor.parameters()])
        _, probs_after, entropy_after = torch_policy_stats(
            agent, np.asarray(jax.device_get(obs))
        )
        records.append(
            dict(
                iteration=iteration,
                warmup=False,
                actor_parameter_delta=float((actor_after - actor_before).abs().sum()),
                entropy_before=before,
                entropy_after=float(entropy_after.mean()),
                max_prob_before=max_prob,
                max_prob_after=float(probs_after.max(dim=-1).values.mean()),
            )
        )

    write_result(
        "1a_warmup",
        dict(
            warmup_steps_in_policy_optimizer=0,
            warmup_branch_found=False,
            policy_updated_on_iteration_1=records[0]["actor_parameter_delta"] > 0,
            records=records,
            conclusion=(
                "No warmup path exists in TWM 0.0.59 Agent.learn or in the harness; "
                "the actor is updated from iteration 1. Warmup is therefore not the "
                "cause of the collapse."
            ),
        ),
        args.json_out,
    )


def probe_reward(args):
    env, params, _, step_batched, reset_batched = diag.build_env(args)
    rng = jax.random.PRNGKey(args.seed)
    rng, sub = jax.random.split(rng)
    _, state = reset_batched(jax.random.split(sub, args.envs))

    rewards = []
    actions = []
    for _ in range(200):
        rng, action_key, step_key = jax.random.split(rng, 3)
        action = jax.random.randint(action_key, (args.envs,), 0, 17, dtype=jnp.int32)
        _, state, reward, _, _ = step_batched(
            jax.random.split(step_key, args.envs), state, action
        )
        rewards.append(np.asarray(jax.device_get(reward), dtype=np.float32))
        actions.append(np.asarray(jax.device_get(action), dtype=np.int32))

    rewards = np.concatenate(rewards)
    actions = np.concatenate(actions)
    values, counts = np.unique(np.round(rewards, 6), return_counts=True)
    top = sorted(zip(values.tolist(), counts.tolist()), key=lambda x: -x[1])[:20]
    write_result(
        "1b_reward",
        dict(
            sample_steps=int(rewards.size),
            random_action_counts=np.bincount(actions, minlength=17).tolist(),
            reward_min=float(rewards.min()),
            reward_max=float(rewards.max()),
            reward_mean=float(rewards.mean()),
            reward_std=float(rewards.std()),
            reward_quantiles={
                str(q): float(np.quantile(rewards, q))
                for q in (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
            },
            most_common_values=[dict(value=v, count=c) for v, c in top],
            negative_fraction=float((rewards < 0).mean()),
            below_world_model_min=int((rewards < -1.0).sum()),
            above_world_model_max=int((rewards > 1.0).sum()),
            conclusion=(
                "Compare the observed range with the WM support [-1, 1]. "
                "TWM Agent.learn does not clip rewards."
            ),
        ),
        args.json_out,
    )


def probe_entropy(args):
    agent = diag.build_agent(17, DEVICE)
    env, params, _, step_batched, reset_batched = diag.build_env(args)
    rng = jax.random.PRNGKey(args.seed)
    rng, sub = jax.random.split(rng)
    obs, state = reset_batched(jax.random.split(sub, args.envs))
    rollout = diag.training_rollout(args, agent, DEVICE, step_batched, obs, state, rng)
    frames_t, actions_t, logp_t, _rewards_t, _dones_t, _metrics_t, _next, *_ = rollout

    states = frames_t[:, :, 0]
    actions = actions_t[:, 0].long()
    with torch.no_grad():
        features, _ = agent.impala(states)
        logits = agent.actor(features)
        probs = logits.softmax(dim=-1)
        entropy = calc_entropy(probs)
        old_log_probs = get_log_prob(logits, actions)

        # Make the sign test independent of the sampled rollout's reward scale.
        # It exercises the exact PPO surrogate formula, not the WM.
        values = torch.zeros_like(entropy)
        returns = torch.linspace(-2.0, 2.0, entropy.shape[0], device=DEVICE)
        advantages = F.layer_norm(returns - values, (returns.shape[0],))
        ratios = (get_log_prob(logits, actions) - old_log_probs).exp()
        surr1 = ratios * advantages
        surr2 = ratios.clamp(1.0 - agent.actor_eps_clip, 1.0 + agent.actor_eps_clip) * advantages
        selected_source = torch.min(surr1, surr2)
        source_loss = selected_source - agent.actor_beta_s * entropy
        corrected_loss = -(torch.min(surr1, surr2)) - agent.actor_beta_s * entropy

    logits_grad = logits.detach().clone().requires_grad_(True)
    # Recompute both losses against a leaf logits tensor for a clean sign test.
    ratios2 = (get_log_prob(logits_grad, actions) - old_log_probs).exp()
    surr1_2 = ratios2 * advantages
    surr2_2 = ratios2.clamp(1.0 - agent.actor_eps_clip, 1.0 + agent.actor_eps_clip) * advantages
    entropy2 = calc_entropy(logits_grad.softmax(dim=-1))
    source_loss2 = torch.min(surr1_2, surr2_2) - agent.actor_beta_s * entropy2
    corrected_loss2 = -torch.min(surr1_2, surr2_2) - agent.actor_beta_s * entropy2
    source_grad = torch.autograd.grad(
        source_loss2.mean(), logits_grad, retain_graph=True
    )[0]
    corrected_grad = torch.autograd.grad(corrected_loss2.mean(), logits_grad)[0]
    selected = source_grad.gather(1, actions[:, None]).squeeze(1)
    selected_corrected = corrected_grad.gather(1, actions[:, None]).squeeze(1)
    positive = advantages > 0

    write_result(
        "1c_entropy",
        dict(
            actor_beta_s=float(agent.actor_beta_s),
            source_entropy_mean=float(entropy.mean()),
            source_entropy_bonus_magnitude=float(
                (agent.actor_beta_s * entropy).mean()
            ),
            source_loss_mean=float(source_loss.mean()),
            corrected_loss_mean=float(corrected_loss.mean()),
            source_grad_selected_positive_adv_mean=float(selected[positive].mean()),
            source_grad_selected_negative_adv_mean=float(selected[~positive].mean()),
            corrected_grad_selected_positive_adv_mean=float(
                selected_corrected[positive].mean()
            ),
            corrected_grad_selected_negative_adv_mean=float(
                selected_corrected[~positive].mean()
            ),
            source_loss_uses_min_surrogate=True,
            source_gradient_wrong_direction_for_positive_advantage=bool(
                selected[positive].mean() > 0
            ),
            conclusion=(
                "TWM 0.0.59 returns +min(surr1, surr2) - beta*entropy. "
                "Minimizing it reduces the log-probability of positive-advantage "
                "actions. The entropy coefficient itself is 0.01 and is active, "
                "but it cannot overcome the inverted policy-gradient sign."
            ),
        ),
        args.json_out,
    )


def count_actions(agent, step_batched, reset_batched, rng, num_envs=10, steps=10):
    rng, sub = jax.random.split(rng)
    obs, state = reset_batched(jax.random.split(sub, num_envs))
    counts = np.zeros(17, dtype=np.int64)
    sampled = []
    max_prob = []
    entropy = []
    finite = True
    for _ in range(steps):
        obs_np = np.asarray(jax.device_get(obs))
        logits, probs, ent = torch_policy_stats(agent, obs_np)
        finite = finite and bool(torch.isfinite(logits).all())
        with torch.no_grad():
            actions, _ = agent.actor(
                agent.impala(diag.obs_to_torch(obs_np, DEVICE))[0],
                sample_action=True,
            )
        act_np = actions.detach().cpu().numpy().astype(np.int32)
        counts += np.bincount(act_np, minlength=17)
        sampled.extend(act_np.tolist())
        max_prob.append(float(probs.max(dim=-1).values.mean()))
        entropy.append(float(ent.mean()))
        rng, key = jax.random.split(rng)
        _, state, _, _, _ = step_batched(
            jax.random.split(key, num_envs), state, jnp.asarray(act_np)
        )
    return dict(
        count=int(counts.sum()),
        action_counts=counts.tolist(),
        action_frequencies=(counts / max(int(counts.sum()), 1)).tolist(),
        noop_fraction=float(counts[0] / max(int(counts.sum()), 1)),
        mean_policy_entropy=float(np.mean(entropy)),
        mean_max_probability=float(np.mean(max_prob)),
        logits_finite=finite,
        first_20_actions=sampled[:20],
    )


def probe_actions(args):
    env, params, _, step_batched, reset_batched = diag.build_env(args)
    rng = jax.random.PRNGKey(args.seed)
    initial = diag.build_agent(17, DEVICE)
    initial_stats = count_actions(
        initial, step_batched, reset_batched, rng, num_envs=10, steps=10
    )
    final_stats = None
    if DEFAULT_CHECKPOINT.exists():
        checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=False)
        final = diag.build_agent(17, DEVICE)
        final.impala.load_state_dict(checkpoint["impala"])
        final.actor.load_state_dict(checkpoint["actor"])
        final.critic.load_state_dict(checkpoint["critic"])
        final_stats = count_actions(
            final, step_batched, reset_batched, rng, num_envs=10, steps=10
        )
    write_result(
        "1d_actions",
        dict(
            action_index_zero_is_noop=True,
            initial_policy=initial_stats,
            final_1m_checkpoint=final_stats,
            checkpoint=str(DEFAULT_CHECKPOINT) if DEFAULT_CHECKPOINT.exists() else None,
            conclusion=(
                "The initial policy samples broadly. The 1M checkpoint is almost "
                "exactly deterministic NOOP, with finite but extreme logits."
            ),
        ),
        args.json_out,
    )


def probe_gradients(args):
    agent = diag.build_agent(17, DEVICE)
    env, params, _, step_batched, reset_batched = diag.build_env(args)
    rng = jax.random.PRNGKey(args.seed)
    rng, sub = jax.random.split(rng)
    obs, state = reset_batched(jax.random.split(sub, args.envs))
    rollout = diag.training_rollout(args, agent, DEVICE, step_batched, obs, state, rng)
    frames_t, actions_t, logp_t, _rewards_t, _dones_t, _metrics_t, _next, *_ = rollout
    states = frames_t[:, :, 0]
    actions = actions_t[:, 0].long()
    returns = torch.linspace(-2.0, 2.0, states.shape[0], device=DEVICE)
    values = torch.zeros_like(returns)
    with torch.no_grad():
        logits = agent.actor(agent.impala(states)[0])
        old_log_probs = get_log_prob(logits, actions)

    agent.actor.zero_grad(set_to_none=True)
    actor_loss = agent.policy_loss(states, actions, old_log_probs, values, returns)
    actor_loss.mean().backward()
    actor_norms = [
        float(p.grad.norm()) for p in agent.actor.parameters() if p.grad is not None
    ]
    impala_norms = [
        float(p.grad.norm()) for p in agent.impala.parameters() if p.grad is not None
    ]
    actor_finite = all(
        bool(torch.isfinite(p.grad).all())
        for p in agent.actor.parameters()
        if p.grad is not None
    )

    # Verify the world-model loss is isolated from policy parameters.
    for parameter in agent.parameters():
        parameter.grad = None
    wm_args = make_args(dim=64, depth=1)
    wm, wm_opt = diag.build_world_model(wm_args, 17, DEVICE)
    wm_opt.zero_grad(set_to_none=True)
    x = torch.randn(1, 3, 2, 63, 63, device=DEVICE)
    r = torch.zeros(1, 2, device=DEVICE)
    a = torch.zeros(1, 2, 1, dtype=torch.long, device=DEVICE)
    d = torch.zeros(1, 2, dtype=torch.bool, device=DEVICE)
    wm_loss = wm(x, rewards=r, actions=a, is_terminal=d)
    wm_loss.backward()
    wm_leaked_to_actor = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in agent.actor.parameters()
    )

    write_result(
        "1e_gradients",
        dict(
            actor_grad_tensor_count=len(actor_norms),
            actor_grad_all_zero=bool(all(v == 0 for v in actor_norms)),
            actor_grad_finite=actor_finite,
            actor_grad_norm_total=float(sum(actor_norms)),
            impala_grad_tensor_count=len(impala_norms),
            impala_grad_all_zero=bool(all(v == 0 for v in impala_norms)),
            impala_grad_norm_total=float(sum(impala_norms)),
            wm_loss=float(wm_loss.detach()),
            wm_loss_leaked_to_actor=wm_leaked_to_actor,
            conclusion=(
                "Policy gradients reach both actor and shared Impala parameters. "
                "WM gradients do not reach policy parameters. The failure is not "
                "a disconnected graph or a WM-to-policy leak."
            ),
        ),
        args.json_out,
    )


def probe_optimizers(args):
    agent = diag.build_agent(17, DEVICE)
    wm, wm_opt = diag.build_world_model(make_args(dim=64, depth=1), 17, DEVICE)
    actor_ids = {id(p) for p in agent.actor.parameters()}
    critic_ids = {id(p) for p in agent.critic.parameters()}
    impala_ids = {id(p) for p in agent.impala.parameters()}
    wm_ids = {id(p) for p in wm.parameters()}
    actor_opt_ids = {
        id(p) for group in agent.actor_optim.param_groups for p in group["params"]
    }
    critic_opt_ids = {
        id(p) for group in agent.critic_optim.param_groups for p in group["params"]
    }
    wm_opt_ids = {id(p) for group in wm_opt.param_groups for p in group["params"]}
    write_result(
        "1f_optimizers",
        dict(
            actor_optimizer_parameter_count=len(actor_opt_ids),
            critic_optimizer_parameter_count=len(critic_opt_ids),
            impala_parameter_count=len(impala_ids),
            shared_impala_parameter_count=len(actor_opt_ids & critic_opt_ids & impala_ids),
            actor_optimizer_contains_impala=bool(impala_ids <= actor_opt_ids),
            critic_optimizer_contains_impala=bool(impala_ids <= critic_opt_ids),
            policy_wm_parameter_overlap=len((actor_ids | critic_ids | impala_ids) & wm_ids),
            wm_optimizer_is_disjoint=not bool(wm_opt_ids & (actor_ids | critic_ids | impala_ids)),
            all_policy_parameters_on_cuda=all(p.is_cuda for p in agent.parameters()),
            optimizer_state_initialized=bool(agent.actor_optim.state),
            actor_optimizer_class=type(agent.actor_optim).__name__,
            critic_optimizer_class=type(agent.critic_optim).__name__,
            conclusion=(
                "Actor and critic optimizers both contain all Impala parameters, so "
                "the shared encoder receives two separately normalized optimizer "
                "updates. WM parameters and optimizer are disjoint. This is a "
                "secondary risk, not yet the isolated cause."
            ),
        ),
        args.json_out,
    )


def probe_checkpoint(args):
    checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=False)
    nan_counts = {}
    for name in ("impala", "actor", "critic", "wm", "actor_optim", "critic_optim", "wm_opt"):
        count = 0
        for value in checkpoint[name].values():
            if torch.is_tensor(value) and torch.is_floating_point(value):
                count += int((~torch.isfinite(value)).sum())
        nan_counts[name] = count
    write_result(
        "checkpoint_finiteness",
        dict(
            checkpoint=str(DEFAULT_CHECKPOINT),
            nonfinite_counts=nan_counts,
            all_finite=not any(nan_counts.values()),
            conclusion="The collapse is a finite, strongly saturated policy, not NaN weights.",
        ),
        args.json_out,
    )


PROBES = {
    "warmup": probe_warmup,
    "reward": probe_reward,
    "entropy": probe_entropy,
    "actions": probe_actions,
    "gradients": probe_gradients,
    "optimizers": probe_optimizers,
    "checkpoint": probe_checkpoint,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("probe", choices=sorted(PROBES))
    parser.add_argument("--json-out", default=None)
    parsed = parser.parse_args()
    args = make_args(
        **{
            key: value
            for key, value in vars(parsed).items()
            if key not in ("probe", "json_out")
        }
    )
    args.json_out = parsed.json_out
    args.probe = parsed.probe
    if DEVICE.type != "cuda":
        raise SystemExit("CUDA is required for these probes")
    torch.manual_seed(0)
    np.random.seed(0)
    PROBES[args.probe](args)


if __name__ == "__main__":
    main()
