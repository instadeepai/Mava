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
from jax import Array, tree
from jumanji.env import State
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from mava.evaluator import get_eval_fn, make_ff_eval_act_fn
from mava.networks import FeedForwardActor as Actor
from mava.networks import FeedForwardQNet as QNetwork
from mava.systems.sac.types import (
    BufferState,
    LearnerState,
    Metrics,
    Networks,
    Optimisers,
    OptStates,
    QVals,
    QValsAndTarget,
    SacParams,
    Transition,
)
from mava.types import MarlEnv, ObservationGlobalState
from mava.utils import make_env as environments
from mava.utils.centralised_training import get_joint_action, get_updated_joint_actions
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps
from mava.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.network_utils import get_action_head
from mava.wrappers import episode_metrics


def init(
    cfg: DictConfig,
) -> Tuple[
    Tuple[MarlEnv, MarlEnv],
    Networks,
    Optimisers,
    TrajectoryBuffer,
    LearnerState,
    Array,
    MavaLogger,
    chex.PRNGKey,
]:
    """Initialize system by creating the envs, networks etc.

    Args:
    ----
        cfg: System configuration.

    Returns:
    -------
        Tuple containing:
            Tuple[Environment, Environment]: The environment and evaluation environment.
            Networks: Tuple of actor and critic networks.
            Optimisers: Tuple of actor, critic and alpha optimisers.
            TrajectoryBuffer: The replay buffer.
            LearnerState: The initial learner state.
            Array: The target entropy.
            MavaLogger: The logger.
            PRNGKey: The random key.

    """
    logger = MavaLogger(cfg)

    key = jax.random.PRNGKey(cfg.system.seed)
    devices = jax.devices()

    def replicate(x: Any) -> Any:
        """First replicate the update batch dim then put on devices."""
        x = tree.map(lambda y: jnp.broadcast_to(y, (cfg.system.update_batch_size, *y.shape)), x)
        return jax.device_put_replicated(x, devices)

    env, eval_env = environments.make(cfg, add_global_state=True)

    n_agents = env.num_agents
    action_dim = env.action_dim

    key, actor_key, q1_key, q2_key, q1_target_key, q2_target_key = jax.random.split(key, 6)

    acts = env.action_spec.generate_value()  # all agents actions
    act_single = acts[0]  # single agents action
    joint_acts = jnp.concatenate([act_single for _ in range(n_agents)], axis=0)
    joint_acts_batched = joint_acts[jnp.newaxis, ...]  # joint actions with a batch dim
    obs = env.observation_spec.generate_value()
    obs_single_batched = tree.map(lambda x: x[0][jnp.newaxis, ...], obs)

    # Making actor network
    actor_torso = hydra.utils.instantiate(cfg.network.actor_network.pre_torso)
    action_head, _ = get_action_head(env.action_spec)
    actor_action_head = hydra.utils.instantiate(
        action_head, action_dim=env.action_dim, independent_std=False
    )
    actor_network = Actor(actor_torso, actor_action_head)
    actor_params = actor_network.init(actor_key, obs_single_batched)

    # Making Q networks
    critic_torso = hydra.utils.instantiate(cfg.network.critic_network.pre_torso)
    q_network = QNetwork(critic_torso, centralised_critic=True)
    q1_params = q_network.init(q1_key, obs_single_batched, joint_acts_batched)
    q2_params = q_network.init(q2_key, obs_single_batched, joint_acts_batched)
    q1_target_params = q_network.init(q1_target_key, obs_single_batched, joint_acts_batched)
    q2_target_params = q_network.init(q2_target_key, obs_single_batched, joint_acts_batched)

    # Automatic entropy tuning
    target_entropy = -cfg.system.target_entropy_scale * action_dim
    target_entropy = jnp.repeat(target_entropy, n_agents).astype(float)
    # making sure we have shape=(B, A) so broacasting works fine
    target_entropy = target_entropy[jnp.newaxis, :]
    if cfg.system.autotune:
        log_alpha = jnp.zeros_like(target_entropy)
    else:
        log_alpha = jnp.log(cfg.system.init_alpha)
        log_alpha = jnp.broadcast_to(log_alpha, target_entropy.shape)

    # Pack params
    online_q_params = QVals(q1_params, q2_params)
    target_q_params = QVals(q1_target_params, q2_target_params)
    params = SacParams(actor_params, QValsAndTarget(online_q_params, target_q_params), log_alpha)

    # Make opt states.
    grad_clip = optax.clip_by_global_norm(cfg.system.max_grad_norm)

    actor_opt = optax.chain(grad_clip, optax.adam(cfg.system.policy_lr))
    actor_opt_state = actor_opt.init(params.actor)

    q_opt = optax.chain(grad_clip, optax.adam(cfg.system.q_lr))
    q_opt_state = q_opt.init(params.q.online)

    alpha_opt = optax.chain(grad_clip, optax.adam(cfg.system.alpha_lr))
    alpha_opt_state = alpha_opt.init(params.log_alpha)

    # Pack opt states
    opt_states = OptStates(actor_opt_state, q_opt_state, alpha_opt_state)

    # Distribute params and opt states across all devices
    params = replicate(params)
    opt_states = replicate(opt_states)

    # Create replay buffer
    init_transition = Transition(
        obs=obs,
        action=acts,
        reward=jnp.zeros((n_agents,), dtype=float),
        done=jnp.zeros((n_agents,), dtype=bool),
        next_obs=obs,
    )

    rb = fbx.make_item_buffer(
        max_length=int(cfg.system.buffer_size),
        min_length=int(cfg.system.explore_steps),
        sample_batch_size=int(cfg.system.batch_size),
        add_batches=True,
    )
    buffer_state = replicate(rb.init(init_transition))

    networks = (actor_network, q_network)
    optims = (actor_opt, q_opt, alpha_opt)

    # Reset env.
    n_keys = cfg.arch.num_envs * cfg.arch.n_devices * cfg.system.update_batch_size
    key_shape = (cfg.arch.n_devices, cfg.system.update_batch_size, cfg.arch.num_envs, -1)
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, n_keys)
    reset_keys = jnp.reshape(reset_keys, key_shape)

    # Keys passed to learner
    first_keys = jax.random.split(key, (cfg.arch.n_devices * cfg.system.update_batch_size))
    first_keys = first_keys.reshape((cfg.arch.n_devices, cfg.system.update_batch_size, -1))

    env_state, first_timestep = jax.pmap(  # devices
        jax.vmap(  # update_batch_size
            jax.vmap(env.reset),  # num_envs
            axis_name="batch",
        ),
        axis_name="device",
    )(reset_keys)
    first_obs = first_timestep.observation

    t = jnp.zeros((cfg.arch.n_devices, cfg.system.update_batch_size), dtype=int)

    # Initial learner state.
    learner_state = LearnerState(
        first_obs, env_state, buffer_state, params, opt_states, t, first_keys
    )
    return (env, eval_env), networks, optims, rb, learner_state, target_entropy, logger, key


def make_update_fns(
    cfg: DictConfig,
    env: MarlEnv,
    networks: Networks,
    optims: Optimisers,
    rb: TrajectoryBuffer,
    target_entropy: chex.Array,
) -> Tuple[
    Callable[[LearnerState], Tuple[LearnerState, Metrics]],
    Callable[[LearnerState], Tuple[LearnerState, Tuple[Metrics, Metrics]]],
]:
    """Create the update functions for the learner.

    Args:
    ----
        cfg: System configuration.
        env: The environment.
        networks: Tuple of actor and critic networks.
        optims: Tuple of actor, critic and alpha optimisers.
        rb: The replay buffer.
        target_entropy: The target entropy.

    Returns:
    -------
        Tuple of (explore_fn, update_fn).
        Explore function is used for initial exploration with random actions.
        Update function is the main learning function, it both acts and learns.

    """
    actor_net, q_net = networks
    actor_opt, q_opt, alpha_opt = optims

    full_action_shape = (cfg.arch.num_envs, *env.action_spec.shape)

    # losses:
    def q_loss_fn(
        q_params: QVals, obs: Array, action: Array, target: Array
    ) -> Tuple[Array, Metrics]:
        q1_params, q2_params = q_params
        joint_action = get_joint_action(action)

        q1_a_values = q_net.apply(q1_params, obs, joint_action)
        q2_a_values = q_net.apply(q2_params, obs, joint_action)

        q1_loss = jnp.mean(jnp.square(q1_a_values - target))
        q2_loss = jnp.mean(jnp.square(q2_a_values - target))

        loss = q1_loss + q2_loss
        loss_info = {
            "loss": loss,
            "q1_loss": q1_loss,
            "q2_loss": q2_loss,
            "q1_a_vals": q1_a_values,
            "q2_a_vals": q2_a_values,
        }

        return loss, loss_info

    def actor_loss_fn(
        actor_params: FrozenVariableDict,
        obs: ObservationGlobalState,
        actions: Array,
        alpha: Array,
        q_params: QVals,
        key: chex.PRNGKey,
    ) -> Array:
        pi = actor_net.apply(actor_params, obs)
        new_actions = pi.sample(seed=key)
        log_prob = pi.log_prob(new_actions)

        # Updated joint actions are done so that each agent's central critic sees what all
        # other agents did in the past, but it sees how its agent's policy is currently acting.
        # This is done by placing new_action[i] in joint_actions[i].
        joint_actions = get_updated_joint_actions(actions, new_actions)

        qval_1 = q_net.apply(q_params.q1, obs, joint_actions)
        qval_2 = q_net.apply(q_params.q2, obs, joint_actions)
        min_q_val = jnp.minimum(qval_1, qval_2)

        return ((alpha * log_prob) - min_q_val).mean()

    def alpha_loss_fn(log_alpha: Array, log_pi: Array, target_entropy: Array) -> Array:
        return jnp.mean(-jnp.exp(log_alpha) * (log_pi + target_entropy))

    # Update functions:
    def update_q(
        params: SacParams, opt_states: OptStates, data: Transition, key: chex.PRNGKey
    ) -> Tuple[SacParams, OptStates, Metrics]:
        """Update the Q parameters."""
        # Calculate Q target values.
        pi = actor_net.apply(params.actor, data.next_obs)
        next_action = pi.sample(seed=key)
        next_log_prob = pi.log_prob(next_action)

        joint_next_actions = get_joint_action(next_action)
        next_q1_val = q_net.apply(params.q.targets.q1, data.next_obs, joint_next_actions)
        next_q2_val = q_net.apply(params.q.targets.q2, data.next_obs, joint_next_actions)
        next_q_val = jnp.minimum(next_q1_val, next_q2_val)
        next_q_val = next_q_val - jnp.exp(params.log_alpha) * next_log_prob

        target_q_val = data.reward + (1.0 - data.done) * cfg.system.gamma * next_q_val

        # Update Q function.
        q_grad_fn = jax.grad(q_loss_fn, has_aux=True)
        q_grads, q_loss_info = q_grad_fn(params.q.online, data.obs, data.action, target_q_val)
        # Mean over the device and batch dimension.
        q_grads, q_loss_info = lax.pmean((q_grads, q_loss_info), axis_name="device")
        q_grads, q_loss_info = lax.pmean((q_grads, q_loss_info), axis_name="batch")
        q_updates, new_q_opt_state = q_opt.update(q_grads, opt_states.q)
        new_online_q_params = optax.apply_updates(params.q.online, q_updates)

        # Target network polyak update.
        new_target_q_params = optax.incremental_update(
            new_online_q_params, params.q.targets, cfg.system.tau
        )

        # Repack params and opt_states.
        q_and_target = QValsAndTarget(new_online_q_params, new_target_q_params)
        params = params._replace(q=q_and_target)
        opt_states = opt_states._replace(q=new_q_opt_state)

        return params, opt_states, q_loss_info

    def update_actor_and_alpha(
        params: SacParams, opt_states: OptStates, data: Transition, key: chex.PRNGKey
    ) -> Tuple[SacParams, OptStates, Metrics]:
        """Update the actor and alpha parameters. Compensated for the delay in policy updates."""
        # compensate for the delay by doing `policy_frequency` updates instead of 1.
        assert cfg.system.policy_update_delay > 0, "Need to have a policy update delay > 0."
        for _ in range(cfg.system.policy_update_delay):
            actor_key, alpha_key = jax.random.split(key)

            # Update actor.
            actor_grad_fn = jax.value_and_grad(actor_loss_fn)
            actor_loss, act_grads = actor_grad_fn(
                params.actor,
                data.obs,
                data.action,
                jnp.exp(params.log_alpha),
                params.q.online,
                actor_key,
            )
            # Mean over the device and batch dimensions.
            actor_loss, act_grads = lax.pmean((actor_loss, act_grads), axis_name="device")
            actor_loss, act_grads = lax.pmean((actor_loss, act_grads), axis_name="batch")
            actor_updates, new_actor_opt_state = actor_opt.update(act_grads, opt_states.actor)
            new_actor_params = optax.apply_updates(params.actor, actor_updates)

            params = params._replace(actor=new_actor_params)
            opt_states = opt_states._replace(actor=new_actor_opt_state)

            # Update alpha if autotuning
            alpha_loss = 0.0  # loss is 0 if autotune is off
            if cfg.system.autotune:
                # Get log prob for alpha loss
                pi = actor_net.apply(params.actor, data.obs)
                action = pi.sample(seed=key)
                log_prob = pi.log_prob(action)

                alpha_grad_fn = jax.value_and_grad(alpha_loss_fn)
                alpha_loss, alpha_grads = alpha_grad_fn(params.log_alpha, log_prob, target_entropy)
                alpha_loss, alpha_grads = lax.pmean((alpha_loss, alpha_grads), axis_name="device")
                alpha_loss, alpha_grads = lax.pmean((alpha_loss, alpha_grads), axis_name="batch")
                alpha_updates, new_alpha_opt_state = alpha_opt.update(alpha_grads, opt_states.alpha)
                new_log_alpha = optax.apply_updates(params.log_alpha, alpha_updates)

                params = params._replace(log_alpha=new_log_alpha)
                opt_states = opt_states._replace(alpha=new_alpha_opt_state)

        loss_info = {"actor_loss": actor_loss, "alpha_loss": alpha_loss}
        return params, opt_states, loss_info

    # Act/learn loops:
    def train(
        carry: Tuple[BufferState, SacParams, OptStates, int, chex.PRNGKey], _: Any
    ) -> Tuple[Tuple[BufferState, SacParams, OptStates, int, chex.PRNGKey], Metrics]:
        """Update the Q function and optionally policy/alpha with TD3 delayed update."""
        buffer_state, params, opt_states, t, key = carry
        key, buff_key, q_key, actor_key = jax.random.split(key, 4)

        # sample
        data = rb.sample(buffer_state, buff_key).experience

        # learn
        params, opt_states, q_loss_info = update_q(params, opt_states, data, q_key)
        params, opt_states, act_loss_info = lax.cond(
            t % cfg.system.policy_update_delay == 0,  # TD 3 Delayed update support
            update_actor_and_alpha,
            # just return same params and opt_states and 0 for losses
            lambda params, opt_states, *_: (
                params,
                opt_states,
                {"actor_loss": 0.0, "alpha_loss": 0.0},
            ),
            params,
            opt_states,
            data,
            actor_key,
        )

        losses = q_loss_info | act_loss_info

        return (buffer_state, params, opt_states, t, key), losses

    # Acting
    def step(
        action: Array, obs: ObservationGlobalState, env_state: State, buffer_state: BufferState
    ) -> Tuple[Array, State, BufferState, Dict]:
        """Given an action, step the environment and add to the buffer."""
        env_state, timestep = jax.vmap(env.step)(env_state, action)
        next_obs = timestep.observation
        rewards = timestep.reward
        terms = ~timestep.discount.astype(bool)
        infos = timestep.extras

        real_next_obs = infos["real_next_obs"]

        transition = Transition(obs, action, rewards, terms, real_next_obs)
        buffer_state = rb.add(buffer_state, transition)

        return next_obs, env_state, buffer_state, infos["episode_metrics"]

    def act(
        carry: Tuple[FrozenVariableDict, Array, State, BufferState, chex.PRNGKey], _: Any
    ) -> Tuple[Tuple[FrozenVariableDict, Array, State, BufferState, chex.PRNGKey], Dict]:
        """Acting loop: select action, step env, add to buffer."""
        actor_params, obs, env_state, buffer_state, key = carry
        key, act_key = jax.random.split(key)

        pi = actor_net.apply(actor_params, obs)
        action = pi.sample(seed=act_key)

        next_obs, env_state, buffer_state, metrics = step(action, obs, env_state, buffer_state)
        return (actor_params, next_obs, env_state, buffer_state, key), metrics

    def explore(carry: LearnerState, _: Any) -> Tuple[LearnerState, Metrics]:
        """Take random actions to fill up buffer at the start of training."""
        obs, env_state, buffer_state, _, _, t, key = carry
        # mypy thinks it's Observation | ObservationGlobalState
        assert isinstance(obs, ObservationGlobalState)

        key, explore_key = jax.random.split(key)
        action = jax.random.uniform(explore_key, full_action_shape)
        next_obs, env_state, buffer_state, metrics = step(action, obs, env_state, buffer_state)

        t += cfg.arch.num_envs
        learner_state = carry._replace(
            obs=next_obs, env_state=env_state, buffer_state=buffer_state, t=t, key=key
        )
        return learner_state, metrics

    scanned_train = lambda state: lax.scan(train, state, None, length=cfg.system.epochs)
    scanned_act = lambda state: lax.scan(act, state, None, length=cfg.system.rollout_length)

    # Act loop -> sample -> update loop
    def update_step(carry: LearnerState, _: Any) -> Tuple[LearnerState, Tuple[Metrics, Metrics]]:
        """Act, sample, learn. The body of the main SAC loop."""
        obs, env_state, buffer_state, params, opt_states, t, key = carry
        key, act_key, learn_key = jax.random.split(key, 3)
        # Act
        act_state = (params.actor, obs, env_state, buffer_state, act_key)
        (_, next_obs, env_state, buffer_state, _), metrics = scanned_act(act_state)

        # Sample and learn
        learn_state = (buffer_state, params, opt_states, t, learn_key)
        (buffer_state, params, opt_states, _, _), losses = scanned_train(learn_state)

        t += cfg.arch.num_envs * cfg.system.rollout_length
        return (
            LearnerState(next_obs, env_state, buffer_state, params, opt_states, t, key),
            (metrics, losses),
        )

    # pmap and scan over explore and update_step
    # Make sure to not do num_envs explore steps (could fill up the buffer too much).
    explore_steps = cfg.system.explore_steps // cfg.arch.num_envs
    pmaped_explore = jax.pmap(
        jax.vmap(
            lambda state: lax.scan(explore, state, None, length=explore_steps),
            axis_name="batch",
        ),
        axis_name="device",
        donate_argnums=0,
    )
    pmaped_update_step = jax.pmap(
        jax.vmap(
            lambda state: lax.scan(update_step, state, None, length=cfg.system.scan_steps),
            axis_name="batch",
        ),
        axis_name="device",
        donate_argnums=0,
    )

    return pmaped_explore, pmaped_update_step


def run_experiment(cfg: DictConfig) -> float:
    # Add runtime variables to config
    cfg.logger.system_name = "ff_masac"
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

    # Initialize system and make learning functions.
    (env, eval_env), networks, optims, rb, learner_state, target_entropy, logger, key = init(cfg)
    explore, update = make_update_fns(cfg, env, networks, optims, rb, target_entropy)

    actor, _ = networks
    key, eval_key = jax.random.split(key)
    eval_act_fn = make_ff_eval_act_fn(actor.apply, cfg)
    evaluator = get_eval_fn(eval_env, eval_act_fn, cfg, absolute_metric=False)

    if cfg.logger.checkpointing.save_model:
        checkpointer = Checkpointer(
            metadata=cfg,  # Save all config as metadata in the checkpoint
            model_name=cfg.logger.system_name,
            **cfg.logger.checkpointing.save_args,  # Checkpoint args
        )

    max_episode_return = -jnp.inf
    start_time = time.time()

    # Fill up buffer/explore.
    learner_state, metrics = explore(learner_state)

    # Log explore metrics.
    t = int(jnp.sum(learner_state.t))
    sps = t / (time.time() - start_time)
    logger.log({"step": t}, t, 0, LogEvent.MISC)

    # Don't mind if episode isn't completed here, nice to have the graphs start near 0.
    # So we ignore the second return value.
    final_metrics, _ = episode_metrics.get_final_step_metrics(metrics)
    final_metrics["steps_per_second"] = sps
    logger.log(final_metrics, cfg.system.explore_steps, 0, LogEvent.ACT)

    # Main loop:
    start = cfg.system.explore_steps
    stop = int(cfg.system.total_timesteps + 1)
    for eval_idx, t in enumerate(range(start, stop, steps_per_rollout)):
        # Learn loop:
        start_time = time.time()
        learner_state, (metrics, losses) = update(learner_state)
        jax.block_until_ready(learner_state)
        t += steps_per_rollout  # Completed rollout so add to step count.

        # Log:
        elapsed_time = time.time() - start_time
        final_metrics, ep_completed = episode_metrics.get_final_step_metrics(metrics)
        final_metrics["steps_per_second"] = steps_per_rollout / elapsed_time
        loss_metrics = losses | {"log_alpha": learner_state.params.log_alpha}

        logger.log({"timestep": t}, t, eval_idx, LogEvent.MISC)
        if ep_completed:
            logger.log(final_metrics, t, eval_idx, LogEvent.ACT)
        logger.log(loss_metrics, t, eval_idx, LogEvent.TRAIN)

        # Evaluate:
        key, eval_key = jax.random.split(key)
        eval_keys = jax.random.split(eval_key, cfg.arch.n_devices)
        eval_metrics = evaluator(unreplicate_batch_dim(learner_state.params.actor), eval_keys, {})
        logger.log(eval_metrics, t, eval_idx, LogEvent.EVAL)
        episode_return = jnp.mean(eval_metrics["episode_return"])

        # Save best actor params.
        if cfg.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(unreplicate_batch_dim(learner_state.params.actor))
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

    # Record the performance for the final evaluation run.
    eval_performance = float(jnp.mean(eval_metrics[cfg.env.eval_metric]))

    # Measure absolute metric.
    if cfg.arch.absolute_metric:
        eval_keys = jax.random.split(key, cfg.arch.n_devices)

        abs_metric_evaluator = get_eval_fn(eval_env, eval_act_fn, cfg, absolute_metric=True)
        eval_metrics = abs_metric_evaluator(best_params, eval_keys, {})

        logger.log(eval_metrics, t, eval_idx, LogEvent.ABSOLUTE)

    logger.stop()

    return eval_performance


@hydra.main(
    config_path="../../../configs/default",
    config_name="ff_masac.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    final_return = run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}MASAC experiment completed{Style.RESET_ALL}")

    return float(final_return)


if __name__ == "__main__":
    hydra_entry_point()
