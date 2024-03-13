# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import time
from pprint import pprint
from typing import Any, Callable, Dict, Tuple  # noqa

import chex
import flashbax as fbx
import hydra
import jax
import jax.lax as lax
import jax.numpy as jnp
import optax
from chex import PRNGKey
from colorama import Fore, Style
from flashbax.buffers.flat_buffer import TrajectoryBuffer
from flax.core.scope import FrozenVariableDict
from jax import Array
from jumanji.env import Environment
from omegaconf import DictConfig, OmegaConf

from mava.evaluator import make_eval_fns
from mava.networks import RecQNetwork, ScannedRNN
from mava.systems.q_learning.types import (
    ActionSelectionState,
    DDQNParams,
    InteractionState,
    LearnerState,
    Metrics,
    TrainState,
    Transition,
)
from mava.types import Observation
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.jax import unreplicate_batch_dim, unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger
from mava.wrappers import episode_metrics


def init(
    cfg: DictConfig,
) -> Tuple[
    Tuple[Environment, Environment],
    RecQNetwork,
    optax.GradientTransformation,
    TrajectoryBuffer,
    LearnerState,
    MavaLogger,
    PRNGKey,
]:
    """Initialize system by creating the envs, networks etc.

    Args:
        cfg: System configuration.

    Returns:
        Tuple containing:
            Tuple[Environment, Environment]: The environment and evaluation environment.
            RecQNetwork: Recurrent Q network
            optax.GradientTransformation: Optimiser for RecQNetwork.
            TrajectoryBuffer: The replay buffer.
            LearnerState: The initial learner state.
            MavaLogger: The logger.
            PRNGKey: The random key.
    """
    logger = MavaLogger(cfg)

    key = jax.random.PRNGKey(cfg.system.seed)
    devices = jax.devices()

    def replicate(x: Any) -> Any:
        """First replicate the update batch dim then put on devices."""
        x = jax.tree_map(lambda y: jnp.broadcast_to(y, (cfg.system.update_batch_size, *y.shape)), x)
        return jax.device_put_replicated(x, devices)

    env, eval_env = environments.make(cfg)

    action_dim = env.action_dim
    num_agents = env.num_agents

    key, q_key = jax.random.split(key, 2)
    # Shape legend:
    # T: Time
    # B: Batch
    # A: Agent
    # Make dummy inputs to init recurrent Q network -> need shape (T, B, A, ...)
    init_obs = env.observation_spec().generate_value()  # (A, ...)
    init_obs_batched = jax.tree_util.tree_map(
        lambda x: jnp.repeat(x[jnp.newaxis, ...], cfg.arch.num_envs, axis=0),
        init_obs,
    )  # (B, A, ...)
    init_obs_batched = jax.tree_util.tree_map(
        lambda x: x[jnp.newaxis, ...], init_obs_batched
    )  # add time dim (1, B, A, ...)
    init_done = jnp.zeros((1, cfg.arch.num_envs, 1), dtype=bool)  # (1, B, 1)
    init_x = (init_obs_batched, init_done)  # pack the RNN dummy inputs
    init_hidden_state = ScannedRNN.initialize_carry(
        (cfg.arch.num_envs, num_agents), cfg.network.hidden_state_dim
    )  # (B, A, ...)

    # Making recurrent Q network
    pre_torso = hydra.utils.instantiate(cfg.network.q_network.pre_torso)
    post_torso = hydra.utils.instantiate(cfg.network.q_network.post_torso)
    q_net = RecQNetwork(
        pre_torso,
        post_torso,
        action_dim,
        cfg.network.hidden_state_dim,
    )
    q_params = q_net.init(q_key, init_hidden_state, init_x)  # epsilon defaults to 0
    q_target_params = q_net.init(q_key, init_hidden_state, init_x)  # ensure parameters are separate

    # Pack Q network params
    params = DDQNParams(q_params, q_target_params)

    # Making optimiser and state
    if cfg.system.clip_optimiser:
        opt = optax.chain(
            optax.clip_by_global_norm(cfg.system.global_norm),
            optax.adam(learning_rate=cfg.system.q_lr, eps=1e-5),
        )
    else:
        opt = optax.adam(learning_rate=cfg.system.q_lr)
    opt_state = opt.init(params.online)

    # Distribute params, opt states and hidden states across all devices
    params = replicate(params)
    opt_state = replicate(opt_state)
    init_hidden_state = replicate(init_hidden_state)

    # Create dummy transition
    init_acts = env.action_spec().generate_value()  # (A,)
    init_transition = Transition(
        obs=Observation(*init_obs),  # (A, ...)
        action=init_acts,
        reward=jnp.zeros((num_agents,), dtype=float),
        done=jnp.zeros((1,), dtype=bool),  # one flag for all agents
    )

    # Initialise trajectory buffer
    rb = fbx.make_trajectory_buffer(
        # n transitions gives n-1 full data points
        sample_sequence_length=cfg.system.recurrent_chunk_size + 1,
        period=1,  # sample any unique trajectory
        add_batch_size=cfg.arch.num_envs,
        sample_batch_size=cfg.system.sample_batch_size,
        max_length_time_axis=cfg.system.buffer_size,
        min_length_time_axis=cfg.system.min_buffer_size,
    )
    buffer_state = rb.init(init_transition)
    buffer_state = replicate(buffer_state)

    # Keys to reset env
    n_keys = cfg.arch.num_envs * cfg.arch.n_devices * cfg.system.update_batch_size
    key_shape = (cfg.arch.n_devices, cfg.system.update_batch_size, cfg.arch.num_envs, -1)
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, n_keys)
    reset_keys = jnp.reshape(reset_keys, key_shape)

    # Get initial state and timestep per-device
    env_state, first_timestep = jax.pmap(  # devices
        jax.vmap(  # update_batch_size
            jax.vmap(env.reset),  # num_envs
            axis_name="batch",
        ),
        axis_name="device",
    )(reset_keys)
    first_obs = first_timestep.observation
    first_done = first_timestep.last()[..., jnp.newaxis]

    # Initialise env steps and training steps
    t0 = jnp.zeros((cfg.arch.n_devices, cfg.system.update_batch_size), dtype=int)
    t0_train = jnp.zeros((cfg.arch.n_devices, cfg.system.update_batch_size), dtype=int)

    # Keys passed to learner
    first_keys = jax.random.split(key, (cfg.arch.n_devices * cfg.system.update_batch_size))
    first_keys = first_keys.reshape((cfg.arch.n_devices, cfg.system.update_batch_size, -1))

    # Initial learner state.
    learner_state = LearnerState(
        first_obs,
        first_done,
        init_hidden_state,
        env_state,
        t0,
        t0_train,
        opt_state,
        buffer_state,
        params,
        first_keys,
    )

    return (env, eval_env), q_net, opt, rb, learner_state, logger, key


def make_update_fns(
    cfg: DictConfig,
    env: Environment,
    q_net: RecQNetwork,
    opt: optax.GradientTransformation,
    rb: TrajectoryBuffer,
) -> Callable[[LearnerState], Tuple[LearnerState, Tuple[Metrics, Metrics]]]:
    """Create the update function for the Q-learner.

    Args:
        cfg: System configuration.
        env: Learning environment.
        q_net: Recurrent q network.
        opt: Optimiser for the recurrent Q network.
        rb: The replay buffer.

    Returns:
        The update function.
    """

    # INTERACT LEVEL 2
    def select_eps_greedy_action(
        action_selection_state: ActionSelectionState, obs: Observation, done: Array
    ) -> Tuple[ActionSelectionState, Array]:
        """Select action to take in epsilon-greedy way. Batch and agent dims are included.

            Args:
            action_selection_state: Tuple of online parameters, previous hidden state,
                environment timestep (used to calculate epsilon) and a random key.
            obs: The observation from the previous timestep.
            done: The flag timestep.last() from the previous timestep.

        Returns:
            A tuple of the updated action selection state and the chosen action.
        """
        params, hidden_state, t, key = action_selection_state

        eps = jax.numpy.maximum(
            cfg.system.eps_min, 1 - (t / cfg.system.eps_decay) * (1 - cfg.system.eps_min)
        )

        obs = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], obs)
        done = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], done)

        next_hidden_state, eps_greedy_dist = q_net.apply(params, hidden_state, (obs, done), eps)

        new_key, explore_key = jax.random.split(key, 2)

        action = eps_greedy_dist.sample(seed=explore_key)
        action = jnp.squeeze(action)

        next_action_selection_state = ActionSelectionState(
            params, next_hidden_state, t + cfg.arch.num_envs, new_key
        )

        return next_action_selection_state, action

    # INTERACT LEVEL 1
    def interaction_step(
        interaction_state: InteractionState, _: Any
    ) -> Tuple[InteractionState, Dict]:
        """Selects action, steps global env, stores timesteps in rb and repacks the parameters."""
        # light unpacking
        action_selection_state, env_state, buffer_state, obs, done = interaction_state

        # select the actions to take
        next_action_selection_state, action = select_eps_greedy_action(
            action_selection_state, obs, done
        )

        # step env with selected actions
        next_env_state, next_timestep = jax.vmap(env.step)(env_state, action)

        # Get reward
        reward = next_timestep.reward
        meaned_reward = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x[:, jnp.newaxis, ...], cfg.system.num_agents, axis=1),
            jnp.mean(next_timestep.reward, axis=-1),
        )

        reward = jax.lax.select(
            (jnp.zeros_like(reward) + cfg.system.mean_reward).astype(int), meaned_reward, reward
        )

        transition = Transition(obs, action, reward, done)
        transition = jax.tree_util.tree_map(
            lambda x: x[:, jnp.newaxis, ...], transition
        )  # Add dummy time dim
        next_buffer_state = rb.add(buffer_state, transition)

        # Next obs and done for learner state
        next_obs = next_timestep.observation
        next_done = next_timestep.last()[
            ..., jnp.newaxis
        ]  # make compatible with network input and transition storage in next step

        # Repack
        new_interact_state = InteractionState(
            next_action_selection_state, next_env_state, next_buffer_state, next_obs, next_done
        )

        return new_interact_state, next_timestep.extras["episode_metrics"]

    def switch_leading_axes(arr: chex.Array) -> chex.Array:
        """Switches the first two axes, generally used for BT -> TB."""
        arr: Dict[str, chex.Array] = jax.tree_map(lambda x: jax.numpy.swapaxes(x, 0, 1), arr)
        return arr

    # TRAIN LEVEL 3 and 4
    def prep_inputs_to_scannedrnn(obs: Observation, done: chex.Array) -> chex.Array:
        """
        Prepares the inputs to the RNN network for either
        getting q values or the eps-greedy distribution.
        """

        hidden_state = ScannedRNN.initialize_carry(
            (cfg.system.sample_batch_size, obs.agents_view.shape[2]), cfg.network.hidden_state_dim
        )
        obs = switch_leading_axes(obs)  # (B, T) -> (T, B)
        done = switch_leading_axes(done)  # (B, T) -> (T, B)
        obs_done = (obs, done)

        return hidden_state, obs_done

    # TRAIN LEVEL 3
    def q_loss_fn(
        q_online_params: FrozenVariableDict, obs: Array, done: Array, action: Array, target: Array
    ) -> Tuple[Array, Metrics]:
        """The portion of the calculation to grad, namely online apply and mse with target."""

        hidden_state, obs_done = prep_inputs_to_scannedrnn(
            obs, done
        )  # axes switched here to scan over time

        _, q_online = q_net.apply(
            q_online_params, hidden_state, obs_done, method="get_q_values"
        )  # get online q values of all actions
        q_online = switch_leading_axes(q_online)  # (T, B, ...) -> (B, T, ...)
        q_online = jnp.squeeze(
            jnp.take_along_axis(q_online, action[..., jnp.newaxis], axis=-1), axis=-1
        )  # get the q values of the taken actions and remove extra dim
        q_loss = jnp.mean((q_online - target) ** 2)  # mse

        # pack metrics for logging
        loss_info = {
            "q_loss": q_loss,
            "mean_q": jnp.mean(q_online),
            "max_q_error": jnp.max(jnp.abs(q_online - target) ** 2),
            "min_q_error": jnp.min(jnp.abs(q_online - target) ** 2),
            "mean_target": jnp.mean(target),
        }

        return q_loss, loss_info

    # Standardise the update function inputs for a cond
    def hard_update(
        next_online_params: FrozenVariableDict, target_params: FrozenVariableDict, t_train: int
    ) -> FrozenVariableDict:
        next_target_params = optax.periodic_update(
            next_online_params, target_params, t_train, cfg.system.update_period
        )
        return next_target_params

    def soft_update(
        next_online_params: FrozenVariableDict, target_params: FrozenVariableDict, t_train: int
    ) -> FrozenVariableDict:
        next_target_params = optax.incremental_update(
            next_online_params, target_params, cfg.system.tau
        )
        return next_target_params

    # TRAIN LEVEL 2
    def update_q(
        params: DDQNParams, opt_states: optax.OptState, data: Transition, t_train: int
    ) -> Tuple[DDQNParams, optax.OptState, Metrics]:
        """Update the Q parameters."""

        # Get the data associated with obs
        data_first = jax.tree_map(lambda x: x[:, :-1, ...], data)
        # Get the data associated with next_obs
        data_next = jax.tree_map(lambda x: x[:, 1:, ...], data)

        first_reward = data_first.reward
        next_done = data_next.done

        # Scan over each sample and discard first timestep
        hidden_state, obs_done = prep_inputs_to_scannedrnn(data.obs, data.done)

        _, next_online_greedy_dist = q_net.apply(
            params.online, hidden_state, obs_done
        )  # eps defaults to 0

        _, next_q_vals_target = q_net.apply(
            params.target, hidden_state, obs_done, method="get_q_values"
        )

        # Skip first timestep, since we are working with "next" information
        next_action = next_online_greedy_dist.mode()[1:, ...]  # (T, B, ...)
        next_q_vals_target = next_q_vals_target[1:, ...]  # (T, B, ...)

        # Double q-value selection
        next_q_val = jnp.squeeze(
            jnp.take_along_axis(next_q_vals_target, next_action[..., jnp.newaxis], axis=-1), axis=-1
        )

        next_q_val = switch_leading_axes(next_q_val)  # (T, B, ...) -> (B, T, ...)

        # TD Target
        target_q_val = (
            first_reward
            + (1.0 - jnp.array(next_done, dtype="float32")) * cfg.system.gamma * next_q_val
        )

        # Update Q function.
        q_grad_fn = jax.grad(q_loss_fn, has_aux=True)
        q_grads, q_loss_info = q_grad_fn(
            params.online, data_first.obs, data_first.done, data_first.action, target_q_val
        )

        # Mean over the device and batch dimension.
        q_grads, q_loss_info = lax.pmean((q_grads, q_loss_info), axis_name="device")
        q_grads, q_loss_info = lax.pmean((q_grads, q_loss_info), axis_name="batch")
        q_updates, next_opt_state = opt.update(q_grads, opt_states)
        next_online_params = optax.apply_updates(params.online, q_updates)

        target_update_params = (next_online_params, params.target, t_train)

        # Depending on choice of target network strategy, update target weights.
        next_target_params = jax.lax.cond(
            cfg.system.hard_update,
            hard_update,
            soft_update,
            *target_update_params,
        )

        # Repack params and opt_states.
        next_params = DDQNParams(next_online_params, next_target_params)

        return next_params, next_opt_state, q_loss_info

    # TRAIN LEVEL 1
    def train(train_state: TrainState, _: Any) -> Tuple[TrainState, Metrics]:
        """Sample, train and repack."""

        # unpack and get keys
        buffer_state, params, opt_states, t_train, key = train_state
        next_key, buff_key = jax.random.split(key, 2)

        # sample
        data = rb.sample(buffer_state, buff_key).experience

        # learn
        next_params, next_opt_states, q_loss_info = update_q(params, opt_states, data, t_train)

        # Repack.
        next_train_state = TrainState(
            buffer_state, next_params, next_opt_states, t_train + 1, next_key
        )

        return next_train_state, q_loss_info

    # INTERACT-TRAIN LOOP

    scanned_interact = lambda state: lax.scan(
        interaction_step, state, None, length=cfg.system.rollout_length
    )
    scanned_train = lambda state: lax.scan(train, state, None, length=cfg.system.epochs)

    def update_step(
        learner_state: LearnerState, _: Any
    ) -> Tuple[LearnerState, Tuple[Metrics, Metrics]]:
        """Interact, then learn."""

        # unpack and get random keys
        (
            obs,
            done,
            hidden_state,
            env_state,
            time_steps,
            train_steps,
            opt_state,
            buffer_state,
            params,
            key,
        ) = learner_state
        new_key, interact_key, train_key = jax.random.split(key, 3)

        # Select actions, step env and store transitions
        action_selection_state = ActionSelectionState(
            params.online, hidden_state, time_steps, interact_key
        )
        interaction_state = InteractionState(
            action_selection_state, env_state, buffer_state, obs, done
        )
        final_interaction_state, metrics = scanned_interact(interaction_state)

        # Sample and learn
        train_state = TrainState(
            final_interaction_state.buffer_state, params, opt_state, train_steps, train_key
        )
        final_train_state, losses = scanned_train(train_state)

        next_learner_state = LearnerState(
            final_interaction_state.obs,
            final_interaction_state.done,
            final_interaction_state.action_selection_state.hidden_state,
            final_interaction_state.env_state,
            final_interaction_state.action_selection_state.time_steps,
            final_train_state.train_steps,
            final_train_state.opt_state,
            final_interaction_state.buffer_state,
            final_train_state.params,
            new_key,
        )

        return next_learner_state, (metrics, losses)

    pmaped_update_step = jax.pmap(
        jax.vmap(
            lambda state: lax.scan(update_step, state, None, length=cfg.system.scan_steps),
            axis_name="batch",
        ),
        axis_name="device",
        donate_argnums=0,
    )

    return pmaped_update_step  # type:ignore


def run_experiment(cfg: DictConfig) -> float:
    # Add runtime variables to config
    cfg.arch.n_devices = len(jax.devices())

    # Number of env steps before evaluating/logging.
    steps_per_rollout = int(cfg.system.total_timesteps // cfg.arch.num_evaluation)
    # Multiplier for a single env/learn step in an anakin system
    anakin_steps = cfg.arch.n_devices * cfg.system.update_batch_size
    # Number of env steps in one anakin style update.
    anakin_act_steps = anakin_steps * cfg.arch.num_envs * cfg.system.rollout_length
    # Number of steps to do in the scanned update method (how many anakin steps).
    cfg.system.scan_steps = int(steps_per_rollout / anakin_act_steps)

    pprint(OmegaConf.to_container(cfg, resolve=True))

    # Initialise system and make learning/evaluation functions
    (env, eval_env), q_net, opt, rb, learner_state, logger, key = init(cfg)
    update = make_update_fns(cfg, env, q_net, opt, rb)

    cfg.system.num_agents = env.num_agents

    key, eval_key = jax.random.split(key)
    evaluator, absolute_metric_evaluator = make_eval_fns(
        eval_env=eval_env,
        network_apply_fn=q_net.apply,
        config=cfg,
        use_recurrent_net=True,
        scanned_rnn=ScannedRNN(cfg.network.hidden_state_dim),
    )

    if cfg.logger.checkpointing.save_model:
        checkpointer = Checkpointer(
            metadata=cfg,  # Save all config as metadata in the checkpoint
            model_name=cfg.logger.system_name,
            **cfg.logger.checkpointing.save_args,  # Checkpoint args
        )

    max_episode_return = -jnp.inf
    start_time = time.time()

    # Main loop:
    for eval_idx, t in enumerate(
        range(steps_per_rollout, int(cfg.system.total_timesteps), steps_per_rollout)
    ):
        # Learn loop:
        learner_state, (metrics, losses) = update(learner_state)
        jax.block_until_ready(learner_state)

        # Log:
        # Add learn steps here because anakin steps per second is learn + act steps
        # But we also want to make sure we're counting env steps correctly so
        # learn steps is not included in the loop counter.
        learn_steps = anakin_steps * cfg.system.epochs
        eps = jax.numpy.maximum(
            cfg.system.eps_min, 1 - (t / cfg.system.eps_decay) * (1 - cfg.system.eps_min)
        )
        sps = (t + learn_steps) / (time.time() - start_time)

        final_metrics, ep_completed = episode_metrics.get_final_step_metrics(metrics)
        loss_metrics = losses
        logger.log({"step": t, "steps_per_second": sps}, t, eval_idx, LogEvent.MISC)
        if ep_completed:
            logger.log(final_metrics, t, eval_idx, LogEvent.ACT)
        logger.log(loss_metrics, t, eval_idx, LogEvent.TRAIN)
        logger.log({"epsilon": eps}, t, eval_idx, LogEvent.MISC)

        # Evaluate:
        key, eval_key = jax.random.split(key)
        eval_keys = jax.random.split(eval_key, cfg.arch.n_devices)
        # essentially squeeze/unreplicate batch dim
        eval_params = unreplicate_batch_dim(learner_state.params.online)
        eval_output = evaluator(eval_params, eval_keys)
        jax.block_until_ready(eval_output)

        # Log:
        episode_return = jnp.mean(eval_output.episode_metrics["episode_return"])
        logger.log(eval_output.episode_metrics, t, eval_idx, LogEvent.EVAL)

        # Save best actor params.
        if cfg.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(eval_params)
            max_episode_return = episode_return

        # Checkpoint:
        if cfg.logger.checkpointing.save_model:
            # Save checkpoint of learner state
            unreplicated_learner_state = unreplicate_n_dims(learner_state)  # type: ignore
            checkpointer.save(
                timestep=t,
                unreplicated_learner_state=unreplicated_learner_state,
                episode_return=episode_return,
            )

    # Measure absolute metric.
    if cfg.arch.absolute_metric:
        eval_keys = jax.random.split(key, cfg.arch.n_devices)

        eval_output = absolute_metric_evaluator(best_params, eval_keys)
        jax.block_until_ready(eval_output)

        logger.log(eval_output.episode_metrics, t, eval_idx, LogEvent.ABSOLUTE)

    logger.stop()

    return float(max_episode_return)


@hydra.main(config_path="../../configs", config_name="default_rec_iql.yaml", version_base="1.2")
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    final_return = run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}IDQN experiment completed{Style.RESET_ALL}")

    return float(final_return)


if __name__ == "__main__":
    hydra_entry_point()
