"""
An example following the Anakin podracer example found here: 
https://colab.research.google.com/drive/1974D-qP17fd5mLxy6QZv-ic4yxlPJp-G?usp=sharing#scrollTo=myLN2J47oNGq
"""

import timeit
import jax 
import jax.numpy as jnp 
import numpy as np 
from typing import NamedTuple, Sequence
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
import jumanji
import optax

class TimeIt():

  def __init__(self, tag, frames=None):
    self.tag = tag
    self.frames = frames

  def __enter__(self):
    self.start = timeit.default_timer()
    return self

  def __exit__(self, *args):
    self.elapsed_secs = timeit.default_timer() - self.start
    msg = self.tag + (': Elapsed time=%.2fs' % self.elapsed_secs)
    if self.frames:
      msg += ', FPS=%.2e' % (self.frames / self.elapsed_secs)
    print(msg)

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    observation: jnp.ndarray
    info: jnp.ndarray

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, observation):
        
        x = observation.agents_view

        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        masked_logits = jnp.where(
            observation.action_mask,
            actor_mean,
            jnp.finfo(jnp.float32).min,
        )

        pi = distrax.Categorical(logits=masked_logits)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)

def get_learner_fn(env, forward_pass, opt_update, config):
    
    def loss_fn():
        


def run_experiment(env, config):
    cores_count = len(jax.devices())
    num_actions = int(env.action_spec().num_values[0])
    network = ActorCritic(num_actions, activation=config["ACTIVATION"])
    optim = optax.chain(
                 optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                 optax.adam(config["LR"], eps=1e-5),
             )
   
    rng = jax.random.PRNGKey(config["SEED"])
    rng, rng_env, rng_params = jax.random.split(rng, 3)

    init_obs = env.observation_spec().generate_value()
    init_obs = jax.tree_util.tree_map(
        lambda x: x[None, ...], 
        init_obs, 
    )

    network_params = network.init(rng_params, init_obs)
    opt_state = optim.init(network_params)

    learn = get_learner_fn(
        env, 
        network.apply,
        optim.update,
        config, 
    )

   
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
    "BATCH_SIZE": 128, 
    "ROLLOUT_LENGTH": 16,
    "ITERATIONS": 100,
    "SEED": 42,
}
run_experiment(jumanji.make(config["ENV_NAME"]), config)