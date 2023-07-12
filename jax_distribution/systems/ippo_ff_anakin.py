"""
An example following the Anakin podracer example found here:
https://colab.research.google.com/drive/1974D-qP17fd5mLxy6QZv-ic4yxlPJp-G?usp=sharing#scrollTo=myLN2J47oNGq
"""

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
from flax.core.frozen_dict import FrozenDict
from flax.linen.initializers import constant, orthogonal
from jumanji.environments.routing.multi_cvrp.generator import UniformRandomGenerator
from jumanji.environments.routing.multi_cvrp.types import State
from jumanji.types import TimeStep
from jumanji.wrappers import AutoResetWrapper, Wrapper


class TimeIt:
    """Context manager for timing execution."""

    def __init__(self, tag: str, frames=None) -> None:
        """Initialise the context manager."""
        self.tag = tag
        self.frames = frames

    def __enter__(self) -> "TimeIt":
        """Start the timer."""
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args: Any) -> None:
        """Print the elapsed time."""
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

    def __init__(self, env_name: str, num_agents=None) -> None:
        self._env_name = env_name
        if "MultiCVRP" in env_name:
            if num_agents is None:
                num_agents = 3
            generator = UniformRandomGenerator(num_vehicles=num_agents, num_customers=6)
            env = jumanji.make(env_name, generator=generator)
        elif "RobotWarehouse" in env_name:
            env = jumanji.make(env_name)
        self._env = AutoResetWrapper(env)

    def get_num_agents(self) -> int:
        """Get the number of agents in the environment"""
        if "MultiCVRP" in self._env_name:
            num_agents = self._env.num_vehicles
        elif "RobotWarehouse" in self._env_name:
            num_agents = self._env.num_agents
        else:
            raise NotImplementedError("This environment is not supported")
        return num_agents

    def get_num_actions(self) -> int:
        """Get the number of actions in the environment"""
        if "MultiCVRP" in self._env_name:
            num_actions = int(self._env.action_spec().maximum)
        elif "RobotWarehouse" in self._env_name:
            num_actions = int(self._env.action_spec().num_values[0])
        else:
            raise NotImplementedError("This environment is not supported")
        return num_actions

    def reset(self, key: chex.PRNGKey) -> Tuple[LogEnvState, TimeStep]:
        """Reset the environment."""
        state, timestep = self._env.reset(key)
        state = LogEnvState(state, 0.0, 0, 0.0, 0)
        return state, timestep

    def step(
        self,
        state: State,
        action: jnp.ndarray,
    ) -> Tuple[State, TimeStep]:
        """Step the environment."""
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

    @staticmethod
    def process_observation(observation: Any, env_name: str) -> Any:
        if "MultiCVRP" in env_name:
            observation = observation.vehicles.coordinates
        elif "RobotWarehouse" in env_name:
            observation = observation.agents_view
        else:
            raise NotImplementedError("This environment is not supported")
        return observation


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    observation: jnp.ndarray
    info: jnp.ndarray


class ActorCritic(nn.Module):
    """Actor Critic Network."""

    action_dim: Sequence[int]
    activation: str = "tanh"
    env_name: str = "RobotWarehouse-v0"

    @nn.compact
    def __call__(self, observation) -> Tuple[distrax.Categorical, jnp.ndarray]:
        """Forward pass."""
        x = LogWrapper.process_observation(observation, self.env_name)

        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_output = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)

        actor_output = activation(actor_output)
        actor_output = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_output)
        actor_output = activation(actor_output)

        actor_output = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_output)

        masked_logits = jnp.where(
            observation.action_mask,
            actor_output,
            jnp.finfo(jnp.float32).min,
        )

        pi = distrax.Categorical(logits=masked_logits)

        critic_output = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic_output = activation(critic_output)
        critic_output = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic_output)
        critic_output = activation(critic_output)
        critic_output = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic_output)

        return pi, jnp.squeeze(critic_output, axis=-1)


def get_learner_fn(
    env: jumanji.Environment,
    forward_pass: nn.Module,
    opt_update: optax.GradientTransformation,
    config: dict,
) -> callable:
    """Get the learner function."""

    def update_step_fn(
        params: FrozenDict,
        opt_state: optax.OptState,
        outer_rng: chex.PRNGKey,
        env_state: LogEnvState,
        timestep: TimeStep,
    ) -> Tuple:
        """Update the network."""
        # COLLECT TRAJECTORIES
        def _env_step(runner_state: Tuple, unused: Any) -> Tuple:
            """Step the environment."""
            params, env_state, last_timestep, rng = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            pi, value = forward_pass(params, last_timestep.observation)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            env_state, next_timestep = env.step(env_state, action)

            done, reward = jax.tree_map(
                lambda x: jnp.repeat(x, env.get_num_agents()).reshape(-1),
                [next_timestep.last(), next_timestep.reward],
            )

            transition = Transition(
                done,
                action,
                value,
                reward,
                log_prob,
                last_timestep.observation,
                {
                    "returned_episode_returns": jnp.repeat(
                        env_state.returned_episode_returns, env.get_num_agents()
                    ).reshape(-1),
                    "returned_episode_lengths": jnp.repeat(
                        env_state.returned_episode_lengths, env.get_num_agents()
                    ).reshape(-1),
                    "returned_episode": jnp.repeat(
                        next_timestep.last(), env.get_num_agents()
                    ).reshape(-1),
                },
            )

            runner_state = (params, env_state, next_timestep, rng)
            return runner_state, transition

        runner_state = (params, env_state, timestep, outer_rng)
        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, config["ROLLOUT_LENGTH"]
        )

        # CALCULATE ADVANTAGE
        params, env_state, last_timestep, rng = runner_state
        _, last_val = forward_pass(params, last_timestep.observation)

        def _calculate_gae(
            traj_batch: Transition, last_val: jnp.ndarray
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Calculate the GAE."""

            def _get_advantages(
                gae_and_next_value: Tuple, transition: Transition
            ) -> Tuple:
                """Calculate the GAE for a single transition."""
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
        def _update_epoch(update_state: Tuple, unused: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minbatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""
                traj_batch, advantages, targets = batch_info

                def _loss_fn(
                    params: FrozenDict,
                    traj_batch: Transition,
                    gae: jnp.ndarray,
                    targets: jnp.ndarray,
                ) -> Tuple:
                    """Calculate the loss."""
                    # RERUN NETWORK
                    pi, value = forward_pass(params, traj_batch.observation)
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
                total_loss, grads = grad_fn(params, traj_batch, advantages, targets)
                # pmean
                total_loss = jax.lax.pmean(total_loss, axis_name="devices")
                grads = jax.lax.pmean(grads, axis_name="j")
                grads = jax.lax.pmean(grads, axis_name="devices")
                updates, new_opt_state = opt_update(grads, opt_state)
                new_params = optax.apply_updates(params, updates)
                return (new_params, new_opt_state), total_loss

            (params, opt_state), traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)
            batch_size = config["ROLLOUT_LENGTH"]
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[1:]), batch
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

            update_state = ((params, opt_state), traj_batch, advantages, targets, rng)
            return update_state, total_loss

        update_state = ((params, opt_state), traj_batch, advantages, targets, rng)

        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
        )

        ((params, opt_state), traj_batch, advantages, targets, rng) = update_state

        metric_info = jax.tree_util.tree_map(lambda x: x[:, 0], traj_batch.info)
        return params, opt_state, rng, env_state, last_timestep, metric_info

    def update_fn(
        params: FrozenDict,
        opt_state: optax.OptState,
        rng: chex.PRNGKey,
        env_state: LogEnvState,
        timestep: TimeStep,
    ) -> Tuple:
        """Compute a gradient update from a single trajectory."""
        rng, loss_rng = jax.random.split(rng)
        (
            new_params,
            new_opt_state,
            rng,
            new_env_state,
            new_timestep,
            metric_info,
        ) = update_step_fn(
            params,
            opt_state,
            rng,
            env_state,
            timestep,
        )

        return (
            new_params,
            new_opt_state,
            rng,
            new_env_state,
            new_timestep,
        ), metric_info

    def learner_fn(
        params: FrozenDict,
        opt_state: optax.OptState,
        rngs: chex.Array,
        env_states: LogEnvState,
        timesteps: TimeStep,
    ) -> Tuple:
        """Vectorise and repeat the update."""
        batched_update_fn = jax.vmap(
            update_fn, axis_name="j"
        )  # vectorize across batch.

        def iterate_fn(val: Tuple, unused: Any) -> Tuple:
            """Repeat the update function."""
            params, opt_state, rngs, env_states, timesteps = val
            return batched_update_fn(params, opt_state, rngs, env_states, timesteps)

        runner_state = (params, opt_state, rngs, env_states, timesteps)
        runner_state, metric = jax.lax.scan(
            iterate_fn, runner_state, None, config["ITERATIONS"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return learner_fn


def run_experiment(env_name, config):
    env = LogWrapper(env_name)
    cores_count = len(jax.devices())

    # Number of iterations
    timesteps_per_iteration = (
        cores_count * config["ROLLOUT_LENGTH"] * config["BATCH_SIZE"]
    )
    config["ITERATIONS"] = (
        config["TOTAL_TIMESTEPS"] // timesteps_per_iteration
    )  # Number of training updates
    total_time_steps = config["TOTAL_TIMESTEPS"]

    # Create the network
    network = ActorCritic(
        env.get_num_actions(), activation=config["ACTIVATION"], env_name=env_name
    )
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

    params = network.init(rng_params, init_obs)
    opt_state = optim.init(params)

    # Create the learner function.
    learn = get_learner_fn(
        env,
        network.apply,
        optim.update,
        config,
    )

    learn = jax.pmap(learn, axis_name="devices")  # replicate over multiple cores.

    broadcast = lambda x: jnp.broadcast_to(
        x, (cores_count, config["BATCH_SIZE"]) + x.shape
    )
    params = jax.tree_map(broadcast, params)  # broadcast to cores and batch.
    opt_state = jax.tree_map(broadcast, opt_state)  # broadcast to cores and batch

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

    # Run the experiment.
    with TimeIt(tag="COMPILATION"):
        out = learn(params, opt_state, step_rngs, env_states, env_timesteps)  # compiles
        jax.block_until_ready(out)

    with TimeIt(tag="EXECUTION", frames=total_time_steps):
        out = learn(  # runs compiled fn
            params,
            opt_state,
            step_rngs,
            env_states,
            env_timesteps,
        )
        jax.block_until_ready(out)
    val = out["metrics"]["returned_episode_returns"].mean()
    print(f"Mean Episode Return: {val}")


if __name__ == "__main__":
    config = {
        "LR": 5e-3,
        "ENV_NAME": "RobotWarehouse-v0",  # [RobotWarehouse-v0, MultiCVRP-v0]
        "ACTIVATION": "relu",
        "UPDATE_EPOCHS": 1,
        "NUM_MINIBATCHES": 1,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "BATCH_SIZE": 4,  # Parallel updates / environmnents
        "ROLLOUT_LENGTH": 128,  # Length of each rollout
        "TOTAL_TIMESTEPS": 204800,  # Number of training timesteps
        "SEED": 42,
    }

    run_experiment(config["ENV_NAME"], config)
