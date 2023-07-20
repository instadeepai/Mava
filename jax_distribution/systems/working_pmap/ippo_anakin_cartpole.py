import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import gymnax
from jax_distribution.wrappers.purejaxrl import LogWrapper, FlattenObservationWrapper
import time

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

def run_experiment(env, batch_size, rollout_len, step_size, iterations, seed):
  """Runs experiment."""
  cores_count = len(jax.devices())  # get available TPU cores.
  network = ActorCritic(env.num_actions, "relu")  # define network.
  optim = optax.adam(step_size)  # define optimiser.

  rng, rng_e, rng_p = random.split(random.PRNGKey(seed), num=3)  # prng keys.
  dummy_obs = env.render(env.initial_state(rng_e))[None,]  # dummy for net init.
  params = network.init(rng_p, dummy_obs)  # initialise params.
  opt_state = optim.init(params)  # initialise optimiser stats.

  learn = get_learner_fn(  # get batched iterated update.
      env, network.apply, optim.update, rollout_len=rollout_len,
      agent_discount=1, lambda_=0.99, iterations=iterations)
  learn = jax.pmap(learn, axis_name='i')  # replicate over multiple cores.

  broadcast = lambda x: jnp.broadcast_to(x, (cores_count, batch_size) + x.shape)
  params = jax.tree_map(broadcast, params)  # broadcast to cores and batch.
  opt_state = jax.tree_map(broadcast, opt_state)  # broadcast to cores and batch

  rng, *env_rngs = jax.random.split(rng, cores_count * batch_size + 1)
  env_states = jax.vmap(env.initial_state)(jnp.stack(env_rngs))  # init envs.
  rng, *step_rngs = jax.random.split(rng, cores_count * batch_size + 1)

  reshape = lambda x: x.reshape((cores_count, batch_size) + x.shape[1:])
  step_rngs = reshape(jnp.stack(step_rngs))  # add dimension to pmap over.
  env_states = reshape(env_states)  # add dimension to pmap over.

  with TimeIt(tag='COMPILATION'):
    learn(params, opt_state, step_rngs, env_states)  # compiles

  num_frames = cores_count * iterations * rollout_len * batch_size
  with TimeIt(tag='EXECUTION', frames=num_frames):
    params, opt_state, step_rngs, env_states = learn(  # runs compiled fn
        params, opt_state, step_rngs, env_states)