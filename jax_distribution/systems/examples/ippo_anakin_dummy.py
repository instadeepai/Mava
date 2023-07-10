"""
An example following the Anakin podracer example found here:
https://colab.research.google.com/drive/1974D-qP17fd5mLxy6QZv-ic4yxlPJp-G?usp=sharing#scrollTo=myLN2J47oNGq
"""

import timeit
from typing import NamedTuple, Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jumanji
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal


class TimeIt:
    def __init__(self, tag, frames=None):
        self.tag = tag
        self.frames = frames

    def __enter__(self):
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args):
        self.elapsed_secs = timeit.default_timer() - self.start
        msg = self.tag + (": Elapsed time=%.2fs" % self.elapsed_secs)
        if self.frames:
            msg += ", FPS=%.2e" % (self.frames / self.elapsed_secs)
        print(msg)


    # COLLECT TRAJECTORIES
    def setup_env(self, env, config):
        def _env_step(runner_state, unused):
            params, env_state, last_timestep, rng = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)

            # TODO: Take a random action
            action = jax.random.randint(_rng, shape=(4,), minval=0, maxval=4)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            env_state, next_timestep = env.step(env_state, action)

            done, reward = jax.tree_map(
                lambda x: jnp.repeat(x, 4).reshape(-1),
                [next_timestep.last(), next_timestep.reward],
            )

            runner_state = (env_state, rng)
            return runner_state, None

        runner_state = (env_state, outer_rng)
        runner_state, _ = jax.lax.scan(
            _env_step, runner_state, None, config["ROLLOUT_LENGTH"]
        )
    
def run_experiment(env, config):
    cores_count = len(jax.devices())
    

    rng = jax.random.PRNGKey(config["SEED"])
    rng, rng_env, rng_params = jax.random.split(rng, 3)

    init_obs = env.observation_spec().generate_value()
    init_obs = jax.tree_util.tree_map(
        lambda x: x[None, ...],
        init_obs,
    )

    # TODO: Complete this
    learn = setup_env(
        env,
        config,
    )

    learn = jax.pmap(learn, axis_name="i")  # replicate over multiple cores.

    broadcast = lambda x: jnp.broadcast_to(
        x, (cores_count, config["BATCH_SIZE"]) + x.shape
    )
  
    rng, *env_rngs = jax.random.split(rng, cores_count * config["BATCH_SIZE"] + 1)
    env_states, env_timesteps = jax.vmap(env.reset)(jnp.stack(env_rngs))  # init envs.
    rng, *step_rngs = jax.random.split(rng, cores_count * config["BATCH_SIZE"] + 1)

    reshape = lambda x: x.reshape((cores_count, config["BATCH_SIZE"]) + x.shape[1:])
    step_rngs = reshape(jnp.stack(step_rngs))  # add dimension to pmap over.
    env_states = jax.tree_util.tree_map(
        reshape,
        env_states,
    )  # add dimension to pmap over.
    env_timesteps = jax.tree_util.tree_map(
        reshape,
        env_timesteps,
    )
    # env_timesteps = reshape(env_timesteps)  # add dimension to pmap over.

    with TimeIt(tag="COMPILATION"):
        out = learn(step_rngs, env_states, env_timesteps)  # compiles
        jax.block_until_ready(out)

    num_frames = (
        cores_count
        * config["ITERATIONS"]
        * config["ROLLOUT_LENGTH"]
        * config["BATCH_SIZE"]
    )
    with TimeIt(tag="EXECUTION", frames=num_frames):
        out = learn(  # runs compiled fn
            step_rngs, env_states, env_timesteps, 
        )
        jax.block_until_ready(out)

if __name__ == "__main__":
    config = {
        "LR": 5e-3,
        "ENV_NAME": "RobotWarehouse-v0",
        "ACTIVATION": "relu",
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 8,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "BATCH_SIZE": 1280, # Parallel updates / environmnents
        "ROLLOUT_LENGTH": 20, # Length of each rollout
        "ITERATIONS": 100, # Number of training updates 
        "SEED": 42,
    }

    run_experiment(jumanji.make(config["ENV_NAME"]), config)
