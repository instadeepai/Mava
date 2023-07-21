import timeit
from typing import Any, NamedTuple, Sequence, Tuple

import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jumanji
import numpy as np
import optax
from flax import struct
from flax.linen.initializers import constant, orthogonal
from jumanji.environments.routing.robot_warehouse.types import State
from jumanji.types import TimeStep
from jumanji.wrappers import AutoResetWrapper, Wrapper


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


@struct.dataclass
class LogEnvState:
    env_state: State
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int


class LogWrapper(Wrapper):
    """Log the episode returns and lengths."""

    def reset(self, key: chex.PRNGKey) -> Tuple[LogEnvState, TimeStep]:
        state, timestep = self._env.reset(key)
        # timestep.extras = {}
        state = LogEnvState(state, 0.0, 0, 0.0, 0)
        return state, timestep

    def step(
        self,
        state: State,
        action: jnp.ndarray,
    ) -> Tuple[State, TimeStep]:

        env_state, timestep = self._env.step(state.env_state, action)

        new_episode_return = state.episode_returns + jnp.mean(timestep.reward)
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - timestep.last()),
            episode_lengths=new_episode_length * (1 - timestep.last()),
            returned_episode_returns=state.returned_episode_returns
            * (1 - timestep.last())
            + new_episode_return * timestep.last(),
            returned_episode_lengths=state.returned_episode_lengths
            * (1 - timestep.last())
            + new_episode_length * timestep.last(),
        )
        return state, timestep

class Flatten(nn.Module):
    def __call__(self, x):
        return jnp.reshape(x, (x.shape[0],x.shape[1], -1))

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    time_limit: int = 100

    @nn.compact
    def __call__(self, observation):
        
        # TODO: Include grid embedding to the network. 
        # Something like this. 
        # observation.grid (B, 10, 10) -> CNN -> (B, 1, H) -> (B, 3, H)
        # observation.agents_locations (B, 3, 2)

        # x = concat(grid_embedding, agent_locations) (B, 3, H+2)
        
        """num_conv_channels=[4, 4, 1]
        conv_layers = [
            [
                nn.Conv(output_channels, (3, 3)),
                jax.nn.relu,
            ]
            for output_channels in num_conv_channels
        ]
        torso = nn.Sequential(
            [
                *[layer for conv_layer in conv_layers for layer in conv_layer],
                Flatten()
            ]
        )
        obs =jax.vmap(process_obs)(observation)
        embedding=torso(obs) # (B, N, W*H)
        num_agents = obs.shape[1]
        normalised_step_count = jnp.repeat(
            jnp.expand_dims(observation.step_count, axis=(-1)) / self.time_limit,
            num_agents,
            axis=1,
        )  # (B, N, 1)
        output = jnp.concatenate(
            [embedding, normalised_step_count], axis=-1
        )
        x = output"""

        # Define convolutional an dense layers
        grid=jnp.expand_dims(observation.grid, axis=-1)
        conv_output = nn.Conv(32, kernel_size=(3, 3), strides=(2, 2))(grid)
        conv_output = jax.nn.relu(conv_output)
        conv_output = nn.Conv(32, kernel_size=(3, 3), strides=(2, 2))(conv_output)
        conv_output = jax.nn.relu(conv_output)
        conv_output = nn.Conv(1, kernel_size=(1, 1), strides=(1, 1))(conv_output)
        conv_output = jax.nn.relu(conv_output)
        # Flatten the output
        conv_output = jnp.reshape(conv_output, (observation.agents_locations.shape[0],observation.agents_locations.shape[1], -1))
        # Pass through dense layers
        grid_embedding = nn.Dense(8)(conv_output)
        x =  jnp.concatenate(
            [grid_embedding, observation.agents_locations], axis=-1
        )

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


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: dict


def get_learner_fn(env, apply_fn, update_fn, config):
    def _update_step(runner_state, unused_target):
        def _env_step(runner_state, unused):
            # runner_state = (params, opt_state, rng, env_state, obs)
            params, opt_state, rng, env_state, last_timestep = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            pi, value = apply_fn(params, last_timestep.observation)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            done, reward = jax.tree_util.tree_map(
                lambda x: jnp.repeat(x, config["NUM_AGENTS"]).reshape(
                    config["NUM_ENVS"], -1
                ),
                (timestep.last(), timestep.reward),
            )
            info = {
                "returned_episode_returns": env_state.returned_episode_returns,
                "returned_episode_lengths": env_state.returned_episode_lengths,
            }

            transition = Transition(
                done, action, value, reward, log_prob, last_timestep.observation, info
            )
            runner_state = (params, opt_state, rng, env_state, timestep)
            return runner_state, transition

        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, config["ROLLOUT_LENGTH"]
        )

        # CALCULATE ADVANTAGE
        params, opt_state, rng, env_state, last_timestep = runner_state
        _, last_val = apply_fn(params, last_timestep.observation)

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, last_val)

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            def _update_minbatch(train_state, batch_info):
                params, opt_state = train_state
                traj_batch, advantages, targets = batch_info

                def _loss_fn(params, opt_state, traj_batch, gae, targets):
                    # RERUN NETWORK
                    pi, value = apply_fn(params, traj_batch.obs)
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )

                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"],
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    entropy = pi.entropy().mean()

                    total_loss = (
                        loss_actor
                        + config["VF_COEF"] * value_loss
                        - config["ENT_COEF"] * entropy
                    )
                    return total_loss, (value_loss, loss_actor, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    params, opt_state, traj_batch, advantages, targets
                )

                grads, total_loss = jax.lax.pmean(
                    (grads, total_loss), axis_name="batch"
                )
                grads, total_loss = jax.lax.pmean(
                    (grads, total_loss), axis_name="device"
                )

                updates, new_opt_state = update_fn(grads, opt_state)
                new_params = optax.apply_updates(params, updates)

                return (new_params, new_opt_state), total_loss

            params, opt_state, traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)

            # TODO: Set this properly
            batch_size = config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
            # batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
            # assert (
            #     batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
            # ), "batch size must be equal to number of steps * number of envs"
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )
            (params, opt_state), total_loss = jax.lax.scan(
                _update_minbatch, (params, opt_state), minibatches
            )
            update_state = (params, opt_state, traj_batch, advantages, targets, rng)
            return update_state, total_loss

        update_state = (params, opt_state, traj_batch, advantages, targets, rng)

        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
        )

        params, opt_state, traj_batch, advantages, targets, rng = update_state
        runner_state = (params, opt_state, rng, env_state, last_timestep)
        metric = traj_batch.info

        return runner_state, metric

    def learner_fn(params, opt_state, rng, env_state, timesteps):
        runner_state = (params, opt_state, rng, env_state, timesteps)

        batched_update_step = jax.vmap(
            _update_step, in_axes=(0, None), axis_name="batch"
        )

        runner_state, metric = jax.lax.scan(
            batched_update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return learner_fn


def run_experiment(env, config):
    """Runs experiment."""
    cores_count = len(jax.devices())  # get available TPU cores.
    num_actions = int(env.action_spec().num_values[0])
    num_agents = env.action_spec().shape[0]
    config["NUM_AGENTS"] = num_agents
    network = ActorCritic(num_actions, config["ACTIVATION"])  # define network.
    optim = optax.adam(config["LR"])  # define optimiser.

    rng, rng_e, rng_p = jax.random.split(
        jax.random.PRNGKey(config["SEED"]), num=3
    )  # prng keys.
    init_x = env.observation_spec().generate_value()
    init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)
    params = network.init(rng_p, init_x)
    opt_state = optim.init(params)  # initialise optimiser stats.

    learn = get_learner_fn(  # get batched iterated update.
        env, network.apply, optim.update, config
    )
    learn = jax.pmap(learn, axis_name="device")  # replicate over multiple cores.

    broadcast = lambda x: jnp.broadcast_to(
        x, (cores_count, config["BATCH_SIZE"]) + x.shape
    )
    params = jax.tree_map(broadcast, params)  # broadcast to cores and batch.
    opt_state = jax.tree_map(broadcast, opt_state)  # broadcast to cores and batch

    rng, *env_rngs = jax.random.split(
        rng, cores_count * config["BATCH_SIZE"] * config["NUM_ENVS"] + 1
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
        jnp.stack(env_rngs),
    )  # init envs.
    rng, *step_rngs = jax.random.split(rng, cores_count * config["BATCH_SIZE"] + 1)

    reshape_step_rngs = lambda x: x.reshape(
        (cores_count, config["BATCH_SIZE"]) + x.shape[1:]
    )
    step_rngs = reshape_step_rngs(jnp.stack(step_rngs))  # add dimension to pmap over.

    reshape_states = lambda x: x.reshape(
        (cores_count, config["BATCH_SIZE"], config["NUM_ENVS"]) + x.shape[1:]
    )
    env_states = jax.tree_util.tree_map(
        reshape_states, env_states
    )  # add dimension to pmap over.
    timesteps = jax.tree_util.tree_map(
        reshape_states, timesteps
    )  # add dimension to pmap over.

    with TimeIt(tag="COMPILATION"):
        learn(params, opt_state, step_rngs, env_states, timesteps)  # compiles

    num_frames = (
        cores_count
        * config["NUM_UPDATES"]
        * config["ROLLOUT_LENGTH"]
        * config["BATCH_SIZE"]
        * config["NUM_ENVS"]
    )
    with TimeIt(tag="EXECUTION", frames=num_frames):
        output = learn(  # runs compiled fn
            params, opt_state, step_rngs, env_states, timesteps
        )

    return output


config = {
    "LR": 2.5e-4,
    "BATCH_SIZE": 4,
    "ROLLOUT_LENGTH": 128,
    "NUM_UPDATES": 1000,
    "NUM_ENVS": 64,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 8,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "relu",
    "ENV_NAME": "Cleaner-v0",
    "SEED": 42,
}

env = jumanji.make(config["ENV_NAME"])
env = AutoResetWrapper(env)
env = LogWrapper(env)

output = run_experiment(env, config)
print(f"MEAN RETURN: {output['metrics']['returned_episode_returns'].mean()}")
print(f"MAX RETURN: {output['metrics']['returned_episode_returns'].max()}")
x = 0
