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
from typing import Any, Callable, Dict, Tuple

import chex
import flashbax as fbx
import hydra
import jax
import jax.lax as lax
import jax.numpy as jnp
import optax
from colorama import Fore, Style
from flashbax.buffers.flat_buffer import TrajectoryBuffer
from flax.core.scope import FrozenVariableDict
from flax.linen import FrozenDict
from jax import Array, tree
from jumanji.types import TimeStep
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from mava.evaluator import ActorState, get_eval_fn, get_num_eval_envs
from mava.networks import RecQNetwork, ScannedRNN
from mava.networks.base import QMixingNetwork
from mava.systems.q_learning.types import (
    ActionSelectionState,
    ActionState,
    LearnerState,
    Metrics,
    QMIXParams,
    TrainState,
    Transition,
)
from mava.types import MarlEnv, MavaObservation
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps
from mava.utils.jax_utils import (
    switch_leading_axes,
    unreplicate_batch_dim,
    unreplicate_n_dims,
)
from mava.utils.logger import LogEvent, MavaLogger
from mava.wrappers import episode_metrics


def init(
    cfg: DictConfig,
) -> Tuple[
    Tuple[MarlEnv, MarlEnv],
    RecQNetwork,
    QMixingNetwork,
    optax.GradientTransformation,
    TrajectoryBuffer,
    LearnerState,
    MavaLogger,
    chex.PRNGKey,
]:
    """Initialize system by creating the envs, networks etc."""
    logger = MavaLogger(cfg)

    # init key, get devices available
    key = jax.random.PRNGKey(cfg.system.seed)
    devices = jax.devices()

    def replicate(x: Any) -> Any:
        """First replicate the update batch dim then put on devices."""
        x = tree.map(lambda y: jnp.broadcast_to(y, (cfg.system.update_batch_size, *y.shape)), x)
        return jax.device_put_replicated(x, devices)

    env, eval_env = environments.make(cfg, add_global_state=True)

    action_dim = env.action_dim
    num_agents = env.num_agents

    key, q_key = jax.random.split(key, 2)

    # Shape legend:
    # T: Time
    # B: Batch
    # N: Agent

    # Make dummy inputs to init recurrent Q network -> need shape (T, B, N, ...)
    init_obs = env.observation_spec.generate_value()  # (N, ...)
    # (B, T, N, ...)
    init_obs_batched = tree.map(lambda x: x[jnp.newaxis, jnp.newaxis, ...], init_obs)
    init_term_or_trunc = jnp.zeros((1, 1, 1), dtype=bool)  # (T, B, 1)
    init_x = (init_obs_batched, init_term_or_trunc)
    # (B, N, ...)
    init_hidden_state = ScannedRNN.initialize_carry(
        (cfg.arch.num_envs, num_agents), cfg.network.hidden_state_dim
    )

    # Making recurrent Q network
    pre_torso = hydra.utils.instantiate(cfg.network.q_network.pre_torso)
    post_torso = hydra.utils.instantiate(cfg.network.q_network.post_torso)
    q_net = RecQNetwork(
        pre_torso=pre_torso,
        post_torso=post_torso,
        num_actions=action_dim,
        hidden_state_dim=cfg.network.hidden_state_dim,
    )
    q_params = q_net.init(q_key, init_hidden_state, init_x)
    q_target_params = q_net.init(q_key, init_hidden_state, init_x)

    # Make Mixer Network
    dummy_agent_qs = jnp.zeros(
        (
            cfg.system.sample_batch_size,
            cfg.system.sample_sequence_length - 1,
            num_agents,
        ),
        dtype=float,
    )
    global_env_state_shape = (
        env.observation_spec.generate_value().global_state[0, :].shape
    )  # NOTE: Env wrapper currently duplicates env state for each agent
    dummy_global_env_state = jnp.zeros(
        (
            cfg.system.sample_batch_size,
            cfg.system.sample_sequence_length - 1,
            *global_env_state_shape,
        ),
        dtype=float,
    )
    q_mixer = hydra.utils.instantiate(
        cfg.network.mixer_network,
        num_actions=action_dim,
        num_agents=num_agents,
        embed_dim=cfg.system.qmix_embed_dim,
    )
    mixer_online_params = q_mixer.init(q_key, dummy_agent_qs, dummy_global_env_state)
    mixer_target_params = q_mixer.init(q_key, dummy_agent_qs, dummy_global_env_state)

    # Pack params
    params = QMIXParams(q_params, q_target_params, mixer_online_params, mixer_target_params)

    # Optimiser
    opt = optax.chain(
        optax.adam(learning_rate=cfg.system.q_lr),
    )
    opt_state = opt.init((params.online, params.mixer_online))

    # Distribute params, opt states and hidden states across all devices
    params = replicate(params)
    opt_state = replicate(opt_state)
    init_hidden_state = replicate(init_hidden_state)

    init_acts = env.action_spec.generate_value()

    # NOTE: term_or_trunc refers to the the joint done, ie. when all agents are done or when the
    # episode horizon has been reached. We use this exclusively in QMIX.
    # Terminal refers to individual agent dones. We keep this here for consistency with IQL.
    init_transition = Transition(
        obs=init_obs,  # (N, ...)
        action=init_acts,  # (N,)
        reward=jnp.zeros((1,), dtype=float),
        terminal=jnp.zeros((1,), dtype=bool),
        term_or_trunc=jnp.zeros((1,), dtype=bool),
        next_obs=init_obs,
    )

    # Initialise trajectory buffer
    rb = fbx.make_trajectory_buffer(
        # n transitions gives n-1 full data points
        sample_sequence_length=cfg.system.sample_sequence_length,
        period=1,  # sample any unique trajectory
        add_batch_size=cfg.arch.num_envs,
        sample_batch_size=cfg.system.sample_batch_size,
        max_length_time_axis=cfg.system.buffer_size,
        min_length_time_axis=cfg.system.min_buffer_size,
    )
    buffer_state = rb.init(init_transition)
    buffer_state = replicate(buffer_state)

    # Reset env
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
    first_term_or_trunc = first_timestep.last()[..., jnp.newaxis]
    first_term = (1 - first_timestep.discount[..., 0, jnp.newaxis]).astype(bool)

    # Initialise env steps and training steps
    t0_act = jnp.zeros((cfg.arch.n_devices, cfg.system.update_batch_size), dtype=int)
    t0_train = jnp.zeros((cfg.arch.n_devices, cfg.system.update_batch_size), dtype=int)

    # Keys passed to learner
    first_keys = jax.random.split(key, (cfg.arch.n_devices * cfg.system.update_batch_size))
    first_keys = first_keys.reshape((cfg.arch.n_devices, cfg.system.update_batch_size, -1))

    # Initial learner state.
    learner_state = LearnerState(
        first_obs,
        first_term,
        first_term_or_trunc,
        init_hidden_state,
        env_state,
        t0_act,
        t0_train,
        opt_state,
        buffer_state,
        params,
        first_keys,
    )

    return (env, eval_env), q_net, q_mixer, opt, rb, learner_state, logger, key


def make_update_fns(
    cfg: DictConfig,
    env: MarlEnv,
    q_net: RecQNetwork,
    mixer: QMixingNetwork,
    opt: optax.GradientTransformation,
    rb: TrajectoryBuffer,
) -> Callable[[LearnerState[QMIXParams]], Tuple[LearnerState[QMIXParams], Tuple[Metrics, Metrics]]]:
    def select_eps_greedy_action(
        action_selection_state: ActionSelectionState,
        obs: MavaObservation,
        term_or_trunc: Array,
    ) -> Tuple[ActionSelectionState, Array]:
        """Select action to take in eps-greedy way. Batch and agent dims are included."""

        params, hidden_state, t, key = action_selection_state

        eps = jnp.maximum(
            cfg.system.eps_min, 1 - (t / cfg.system.eps_decay) * (1 - cfg.system.eps_min)
        )

        obs = tree.map(lambda x: x[jnp.newaxis, ...], obs)
        term_or_trunc = tree.map(lambda x: x[jnp.newaxis, ...], term_or_trunc)

        next_hidden_state, eps_greedy_dist = q_net.apply(
            params, hidden_state, (obs, term_or_trunc), eps
        )

        new_key, explore_key = jax.random.split(key, 2)

        action = eps_greedy_dist.sample(seed=explore_key)
        action = action[0, ...]  # (1, B, N) -> (B, N)

        # repack new selection params
        next_action_selection_state = ActionSelectionState(
            params, next_hidden_state, t + cfg.arch.num_envs, new_key
        )
        return next_action_selection_state, action

    def action_step(action_state: ActionState, _: Any) -> Tuple[ActionState, Dict]:
        """Selects an action, steps global env, stores timesteps in global rb and repacks the
        parameters for the next step.
        """

        action_selection_state, env_state, buffer_state, obs, terminal, term_or_trunc = action_state

        next_action_selection_state, action = select_eps_greedy_action(
            action_selection_state, obs, term_or_trunc
        )

        next_env_state, next_timestep = jax.vmap(env.step)(env_state, action)

        # Get reward
        # NOTE: Combine agent rewards, since QMIX is cooperative.
        reward = jnp.mean(next_timestep.reward, axis=-1, keepdims=True)

        transition = Transition(
            obs, action, reward, terminal, term_or_trunc, next_timestep.extras["real_next_obs"]
        )
        # Add dummy time dim
        transition = tree.map(lambda x: x[:, jnp.newaxis, ...], transition)
        next_buffer_state = rb.add(buffer_state, transition)

        next_obs = next_timestep.observation
        # Make compatible with network input and transition storage in next step
        next_terminal = (1 - next_timestep.discount[..., 0, jnp.newaxis]).astype(bool)
        next_term_or_trunc = next_timestep.last()[..., jnp.newaxis]

        new_act_state = ActionState(
            next_action_selection_state,
            next_env_state,
            next_buffer_state,
            next_obs,
            next_terminal,
            next_term_or_trunc,
        )

        return new_act_state, next_timestep.extras["episode_metrics"]

    def prep_inputs_to_scannedrnn(obs: MavaObservation, term_or_trunc: chex.Array) -> chex.Array:
        """Prepares the inputs to the RNN network for either getting q values or the
        eps-greedy distribution.

        Mostly swaps leading axes because the replay buffer outputs (B, T, ... )
        and the RNN takes in (T, B, ...).
        """
        hidden_state = ScannedRNN.initialize_carry(
            (cfg.system.sample_batch_size, obs.agents_view.shape[2]), cfg.network.hidden_state_dim
        )
        # the rb outputs (B, T, ... ) the RNN takes in (T, B, ...)
        obs = switch_leading_axes(obs)  # (B, T) -> (T, B)
        term_or_trunc = switch_leading_axes(term_or_trunc)  # (B, T) -> (T, B)
        obs_term_or_trunc = (obs, term_or_trunc)

        return hidden_state, obs_term_or_trunc

    def q_loss_fn(
        online_params: FrozenVariableDict,
        obs: Array,
        term_or_trunc: Array,
        action: Array,
        target: Array,
    ) -> Tuple[Array, Metrics]:
        """The portion of the calculation to grad, namely online apply and mse with target."""
        q_online_params, online_mixer_params = online_params

        # Axes switched to scan over time
        hidden_state, obs_term_or_trunc = prep_inputs_to_scannedrnn(obs, term_or_trunc)

        # Get online q values of all actions
        _, q_online = q_net.apply(
            q_online_params, hidden_state, obs_term_or_trunc, method="get_q_values"
        )
        q_online = switch_leading_axes(q_online)  # (T, B, ...) -> (B, T, ...)
        # Get the q values of the taken actions and remove extra dim
        q_online = jnp.squeeze(
            jnp.take_along_axis(q_online, action[..., jnp.newaxis], axis=-1), axis=-1
        )

        # NOTE: States are replicated over agents so we take only take first one
        q_online = mixer.apply(
            online_mixer_params, q_online, obs.global_state[:, :, 0, ...]
        )  # (B, T, N, ...) -> (B , T, 1 , ...)

        q_loss = jnp.mean((q_online - target) ** 2)

        q_error = q_online - target
        loss_info = {
            "q_loss": q_loss,
            "mean_q": jnp.mean(q_online),
            "max_q_error": jnp.max(jnp.abs(q_error) ** 2),
            "min_q_error": jnp.min(jnp.abs(q_error) ** 2),
            "mean_target": jnp.mean(target),
        }

        return q_loss, loss_info

    def update_q(
        params: QMIXParams, opt_states: optax.OptState, data_full: Transition, t_train: int
    ) -> Tuple[QMIXParams, optax.OptState, Metrics]:
        """Update the Q parameters."""

        # Get data aligned with current/next timestep
        data = tree.map(lambda x: x[:, :-1, ...], data_full)  # (B, T, ...)
        data_next = tree.map(lambda x: x[:, 1:, ...], data_full)  # (B, T, ...)

        reward = data.reward
        next_done = data_next.term_or_trunc

        # Get the greedy action using the distribution.
        # Epsilon defaults to 0.
        hidden_state, next_obs_term_or_trunc = prep_inputs_to_scannedrnn(
            data_full.obs, data_full.term_or_trunc
        )  # (T, B, ...)
        _, next_greedy_dist = q_net.apply(params.online, hidden_state, next_obs_term_or_trunc)
        next_action = next_greedy_dist.mode()  # (T, B, ...)
        next_action = switch_leading_axes(next_action)  # (T, B, ...) -> (B, T, ...)
        next_action = next_action[:, 1:, ...]  # (B, T, ...)

        hidden_state, next_obs_term_or_trunc = prep_inputs_to_scannedrnn(
            data_full.obs, data_full.term_or_trunc
        )  # (T, B, ...)

        _, next_q_vals_target = q_net.apply(
            params.target, hidden_state, next_obs_term_or_trunc, method="get_q_values"
        )
        next_q_vals_target = switch_leading_axes(next_q_vals_target)  # (T, B, ...) -> (B, T, ...)
        next_q_vals_target = next_q_vals_target[:, 1:, ...]  # (B, T, ...)

        # Double q-value selection
        next_q_val = jnp.squeeze(
            jnp.take_along_axis(next_q_vals_target, next_action[..., jnp.newaxis], axis=-1), axis=-1
        )

        next_q_val = mixer.apply(
            params.mixer_target, next_q_val, data_next.obs.global_state[:, :, 0, ...]
        )  # (B, T, N, ...) -> (B , T, 1 , ...)

        # TD Target
        target_q_val = reward + (1.0 - next_done) * cfg.system.gamma * next_q_val

        q_grad_fn = jax.grad(q_loss_fn, has_aux=True)
        q_grads, q_loss_info = q_grad_fn(
            (params.online, params.mixer_online),
            data.obs,
            data.term_or_trunc,
            data.action,
            target_q_val,
        )
        q_loss_info["mean_reward_t0"] = jnp.mean(reward)
        q_loss_info["mean_next_qval"] = jnp.mean(next_q_val)
        q_loss_info["done"] = jnp.mean(data_full.term_or_trunc)

        # Mean over the device and batch dimension.
        q_grads, q_loss_info = lax.pmean((q_grads, q_loss_info), axis_name="device")
        q_grads, q_loss_info = lax.pmean((q_grads, q_loss_info), axis_name="batch")
        q_updates, next_opt_state = opt.update(q_grads, opt_states)
        (next_online_params, next_mixer_params) = optax.apply_updates(
            (params.online, params.mixer_online), q_updates
        )

        # Target network update.
        if cfg.system.hard_update:
            next_target_params = optax.periodic_update(
                next_online_params, params.target, t_train, cfg.system.update_period
            )
            next_mixer_target_params = optax.periodic_update(
                next_mixer_params, params.mixer_target, t_train, cfg.system.update_period
            )
        else:
            next_target_params = optax.incremental_update(
                next_online_params, params.target, cfg.system.tau
            )
            next_mixer_target_params = optax.incremental_update(
                next_mixer_params, params.mixer_target, cfg.system.tau
            )
        # Repack params and opt_states.
        next_params = QMIXParams(
            next_online_params,
            next_target_params,
            next_mixer_params,
            next_mixer_target_params,
        )

        return next_params, next_opt_state, q_loss_info

    def train(
        train_state: TrainState[QMIXParams], _: Any
    ) -> Tuple[TrainState[QMIXParams], Metrics]:
        """Sample, train and repack."""

        buffer_state, params, opt_states, t_train, key = train_state
        next_key, buff_key = jax.random.split(key, 2)

        data = rb.sample(buffer_state, buff_key).experience

        # Learn
        next_params, next_opt_states, q_loss_info = update_q(params, opt_states, data, t_train)

        next_train_state = TrainState(
            buffer_state, next_params, next_opt_states, t_train + 1, next_key
        )

        return next_train_state, q_loss_info

    # ---- Act-train loop ----
    scanned_act = lambda state: lax.scan(action_step, state, None, length=cfg.system.rollout_length)
    scanned_train = lambda state: lax.scan(train, state, None, length=cfg.system.epochs)

    # Act and train
    def update_step(
        learner_state: LearnerState[QMIXParams], _: Any
    ) -> Tuple[LearnerState[QMIXParams], Tuple[Metrics, Metrics]]:
        """Act, then learn."""

        (
            obs,
            terminal,
            term_or_trunc,
            hidden_state,
            env_state,
            time_steps,
            train_steps,
            opt_state,
            buffer_state,
            params,
            key,
        ) = learner_state
        new_key, act_key, train_key = jax.random.split(key, 3)

        # Select actions, step env and store transitions
        action_selection_state = ActionSelectionState(
            params.online, hidden_state, time_steps, act_key
        )
        action_state = ActionState(
            action_selection_state, env_state, buffer_state, obs, terminal, term_or_trunc
        )
        final_action_state, metrics = scanned_act(action_state)

        # Sample and learn
        train_state = TrainState(
            final_action_state.buffer_state, params, opt_state, train_steps, train_key
        )
        final_train_state, losses = scanned_train(train_state)

        next_learner_state = LearnerState(
            final_action_state.obs,
            final_action_state.terminal,
            final_action_state.term_or_trunc,
            final_action_state.action_selection_state.hidden_state,
            final_action_state.env_state,
            final_action_state.action_selection_state.time_steps,
            final_train_state.train_steps,
            final_train_state.opt_state,
            final_action_state.buffer_state,
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

    return pmaped_update_step


def run_experiment(cfg: DictConfig) -> float:
    cfg.logger.system_name = "rec_qmix"
    cfg.arch.n_devices = len(jax.devices())
    cfg = check_total_timesteps(cfg)

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
    (env, eval_env), q_net, q_mixer, opts, rb, learner_state, logger, key = init(cfg)
    update = make_update_fns(cfg, env, q_net, q_mixer, opts, rb)

    cfg.system.num_agents = env.num_agents

    key, eval_key = jax.random.split(key)

    def eval_act_fn(
        params: FrozenDict, timestep: TimeStep, key: chex.PRNGKey, actor_state: ActorState
    ) -> Tuple[chex.Array, ActorState]:
        """The acting function that get's passed to the evaluator.
        A custom function is needed for epsilon-greedy acting.
        """
        hidden_state = actor_state["hidden_state"]

        term_or_trunc = timestep.last()
        net_input = (timestep.observation, term_or_trunc[..., jnp.newaxis])
        net_input = tree.map(lambda x: x[jnp.newaxis], net_input)  # add batch dim to obs
        next_hidden_state, eps_greedy_dist = q_net.apply(params, hidden_state, net_input)
        action = eps_greedy_dist.sample(seed=key).squeeze(0)
        return action, {"hidden_state": next_hidden_state}

    evaluator = get_eval_fn(eval_env, eval_act_fn, cfg, absolute_metric=False)

    if cfg.logger.checkpointing.save_model:
        checkpointer = Checkpointer(
            metadata=cfg,  # Save all config as metadata in the checkpoint
            model_name=cfg.logger.system_name,
            **cfg.logger.checkpointing.save_args,  # Checkpoint args
        )

    # Create an initial hidden state used for resetting memory for evaluation
    eval_batch_size = get_num_eval_envs(cfg, absolute_metric=False)
    eval_hs = ScannedRNN.initialize_carry(
        (jax.device_count(), eval_batch_size, cfg.system.num_agents),
        cfg.network.hidden_state_dim,
    )

    max_episode_return = -jnp.inf
    best_params = copy.deepcopy(unreplicate_batch_dim(learner_state.params.online))

    # Main loop:
    for eval_idx, t in enumerate(
        range(steps_per_rollout, int(cfg.system.total_timesteps + 1), steps_per_rollout)
    ):
        # Learn loop:
        start_time = time.time()
        learner_state, (metrics, losses) = update(learner_state)
        jax.block_until_ready(learner_state)

        # Log:
        elapsed_time = time.time() - start_time
        eps = jnp.maximum(
            cfg.system.eps_min, 1 - (t / cfg.system.eps_decay) * (1 - cfg.system.eps_min)
        )
        final_metrics, ep_completed = episode_metrics.get_final_step_metrics(metrics)
        final_metrics["steps_per_second"] = steps_per_rollout / elapsed_time
        loss_metrics = losses
        logger.log({"timestep": t, "epsilon": eps}, t, eval_idx, LogEvent.MISC)
        if ep_completed:
            logger.log(final_metrics, t, eval_idx, LogEvent.ACT)
        logger.log(loss_metrics, t, eval_idx, LogEvent.TRAIN)

        # Evaluate:
        key, eval_key = jax.random.split(key)
        eval_keys = jax.random.split(eval_key, cfg.arch.n_devices)
        eval_params = unreplicate_batch_dim(learner_state.params.online)
        eval_metrics = evaluator(eval_params, eval_keys, {"hidden_state": eval_hs})
        jax.block_until_ready(eval_metrics)
        logger.log(eval_metrics, t, eval_idx, LogEvent.EVAL)
        episode_return = jnp.mean(eval_metrics["episode_return"])

        # Save best actor params.
        if cfg.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(eval_params)
            max_episode_return = episode_return

        # Checkpoint:
        if cfg.logger.checkpointing.save_model:
            # Save checkpoint of learner state
            unreplicated_learner_state = unreplicate_n_dims(learner_state)
            checkpointer.save(
                timestep=t,
                unreplicated_learner_state=unreplicated_learner_state,
                episode_return=episode_return,
            )

    eval_performance = float(jnp.mean(eval_metrics[cfg.env.eval_metric]))

    # Measure absolute metric.
    if cfg.arch.absolute_metric:
        eval_keys = jax.random.split(key, cfg.arch.n_devices)
        eval_batch_size = get_num_eval_envs(cfg, absolute_metric=True)
        eval_hs = ScannedRNN.initialize_carry(
            (jax.device_count(), eval_batch_size, cfg.system.num_agents),
            cfg.network.hidden_state_dim,
        )

        abs_metric_evaluator = get_eval_fn(eval_env, eval_act_fn, cfg, absolute_metric=True)
        eval_metrics = abs_metric_evaluator(best_params, eval_keys, {"hidden_state": eval_hs})
        logger.log(eval_metrics, t, eval_idx, LogEvent.ABSOLUTE)

    logger.stop()

    return eval_performance


@hydra.main(
    config_path="../../../configs/default/",
    config_name="rec_qmix.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)
    # Run experiment.
    eval_performance = run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}QMIX experiment completed{Style.RESET_ALL}")

    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
