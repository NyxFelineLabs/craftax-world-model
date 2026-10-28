"""Project-local Craftax variants used by the task difficulty calibration."""

from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

from craftax.craftax_env import make_craftax_env_from_name


SUPPORTED_VARIANTS = {
    "baseline",
    "reward_shaping",
    "short_episode",
    "curriculum",
}


@struct.dataclass
class ShapingState:
    env_state: object
    visited: jax.Array


@struct.dataclass
class CurriculumState:
    env_state: object
    env_steps: jax.Array
    previous_position: jax.Array


class CalibrationEnv:
    def __init__(self, env, variant, num_envs=1, initial_env_steps=0):
        self._env = env
        self.variant = variant
        self.num_envs = int(num_envs)
        self.initial_env_steps = int(initial_env_steps)
        params = env.default_params
        if variant == "short_episode":
            params = params.replace(max_timesteps=500)
        self._default_params = params

    @property
    def default_params(self):
        return self._default_params

    def __getattr__(self, name):
        return getattr(self._env, name)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        if params is None:
            params = self.default_params
        obs, env_state = self._env.reset(key, params)
        if self.variant not in ("reward_shaping", "curriculum"):
            return obs, env_state
        if self.variant == "curriculum":
            return obs, CurriculumState(
                env_state=env_state,
                env_steps=jnp.asarray(
                    self.initial_env_steps // self.num_envs, dtype=jnp.int32
                ),
                previous_position=env_state.player_position,
            )
        visited = jnp.zeros((64, 64), dtype=jnp.bool_)
        x, y = env_state.player_position
        visited = visited.at[jnp.clip(y, 0, 63), jnp.clip(x, 0, 63)].set(True)
        return obs, ShapingState(env_state=env_state, visited=visited)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        if params is None:
            params = self.default_params
        if self.variant == "curriculum":
            step_key, reset_key = jax.random.split(key)
            obs, env_state, base_reward, done, info = self._env.step(
                step_key, state.env_state, action, params
            )
            moved = jnp.any(env_state.player_position != state.previous_position)
            env_steps = state.env_steps + 1
            global_steps = env_steps.astype(jnp.float32) * self.num_envs
            progress = jnp.clip(
                (global_steps - 50_000.0) / 150_000.0, 0.0, 1.0
            )
            reward = 0.01 + 0.08 * moved.astype(base_reward.dtype)
            reward = reward + progress * base_reward
            info["curriculum_progress"] = progress

            reset_obs, reset_state = self._env.reset(reset_key, params)
            env_state = jax.tree.map(
                lambda reset_value, next_value: jax.lax.select(
                    done, reset_value, next_value
                ),
                reset_state,
                env_state,
            )
            obs = jax.tree.map(
                lambda reset_value, next_value: jax.lax.select(
                    done, reset_value, next_value
                ),
                reset_obs,
                obs,
            )
            return obs, CurriculumState(
                env_state=env_state,
                env_steps=env_steps,
                previous_position=env_state.player_position,
            ), reward, done, info

        if self.variant != "reward_shaping":
            return self._env.step(key, state, action, params)

        step_key, reset_key = jax.random.split(key)
        obs, env_state, reward, done, info = self._env.step(
            step_key, state.env_state, action, params
        )
        x, y = env_state.player_position
        x = jnp.clip(x, 0, 63)
        y = jnp.clip(y, 0, 63)
        is_new = ~state.visited[y, x]
        reward = reward + 0.1 * is_new.astype(reward.dtype)
        visited = state.visited.at[y, x].set(True)

        reset_obs, reset_state = self._env.reset(reset_key, params)
        env_state = jax.tree.map(
            lambda reset_value, next_value: jax.lax.select(
                done, reset_value, next_value
            ),
            reset_state,
            env_state,
        )
        obs = jax.tree.map(
            lambda reset_value, next_value: jax.lax.select(
                done, reset_value, next_value
            ),
            reset_obs,
            obs,
        )
        reset_x, reset_y = env_state.player_position
        reset_visited = jnp.zeros_like(visited).at[
            jnp.clip(reset_y, 0, 63), jnp.clip(reset_x, 0, 63)
        ].set(True)
        visited = jax.lax.select(done, reset_visited, visited)
        return obs, ShapingState(env_state=env_state, visited=visited), reward, done, info


def make_calibration_env(
    name,
    auto_reset,
    variant="baseline",
    num_envs=1,
    initial_env_steps=0,
):
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"unsupported calibration variant: {variant}")
    env = make_craftax_env_from_name(
        name,
        auto_reset=False
        if variant in ("reward_shaping", "curriculum")
        else auto_reset,
    )
    if variant == "baseline":
        return env
    return CalibrationEnv(
        env,
        variant,
        num_envs=num_envs,
        initial_env_steps=initial_env_steps,
    )
