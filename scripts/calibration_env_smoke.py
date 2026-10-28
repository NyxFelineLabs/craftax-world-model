#!/usr/bin/env python3
"""Fast JIT smoke for project-local task calibration environments."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import jax
import jax.numpy as jnp

from calibration_env import make_calibration_env


def main():
    for variant in ("reward_shaping", "short_episode", "curriculum"):
        env = make_calibration_env(
            "Craftax-Classic-Pixels-v1",
            auto_reset=True,
            variant=variant,
            num_envs=2,
            initial_env_steps=200_000 if variant == "curriculum" else 0,
        )
        params = env.default_params
        obs, state = env.reset(jax.random.PRNGKey(0), params)
        states = jax.tree.map(lambda value: jnp.stack((value, value)), state)
        keys = jax.random.split(jax.random.PRNGKey(1), 2)
        step = jax.jit(jax.vmap(lambda key, item, action: env.step(key, item, action, params)))
        next_obs, next_state, reward, done, info = step(
            keys, states, jnp.asarray([0, 1], dtype=jnp.int32)
        )
        jax.block_until_ready(reward)
        assert next_obs.shape == (2, 63, 63, 3)
        assert reward.shape == (2,)
        assert jnp.isfinite(reward).all()
        print(
            variant,
            "obs", tuple(next_obs.shape),
            "reward", tuple(float(value) for value in reward),
            "done", tuple(bool(value) for value in done),
        )
        if variant == "short_episode":
            assert int(params.max_timesteps) == 500
        if variant == "curriculum":
            print(
                "curriculum counters",
                tuple(int(value) for value in next_state.env_steps),
                "progress",
                tuple(float(value) for value in info["curriculum_progress"]),
            )
            assert all(int(value) == 100_001 for value in next_state.env_steps)
            assert all(float(value) == 1.0 for value in info["curriculum_progress"])


if __name__ == "__main__":
    main()
