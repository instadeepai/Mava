# Copyright 2022 InstaD
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
from jax import Array
from jumanji.env import Environment, State
from omegaconf import DictConfig, OmegaConf

from mava.evaluator import make_eval_fns
from mava.networks import FeedForwardActor as Actor
from mava.networks import FeedForwardQNet as QNetwork
from mava.systems.sac.types import (
    BufferState,
    LearnerState,
    Metrics,
    Networks,
    Optimisers,
    OptStates,
    Qs,
    QsAndTarget,
    SacParams,
    Transition,
)
from mava.types import Observation, ObservationGlobalState
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.jax import unreplicate_batch_dim, unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger
from mava.wrappers import episode_metrics


def get_joint_action(actions: Array) -> Array:
    batch_size, num_agents, _ = actions.shape
    repeated_action = jnp.tile(actions[:, jnp.newaxis, ...], (1, num_agents, 1, 1))
    joint_action = jnp.reshape(repeated_action, (batch_size, num_agents, -1))

    return joint_action


def init(
    cfg: DictConfig,
) -> Tuple[
    Tuple[Environment, Environment],
    Networks,
    Optimisers,
    TrajectoryBuffer,
    LearnerState,
    Array,
    MavaLogger,
    chex.PRNGKey,
]:
    logger = MavaLogger(cfg)

    key = jax.random.PRNGKey(cfg.system.seed)
    devices = jax.devices()

    def replicate(x: Any) -> Any:
        """First replicate the update batch dim then put on devices."""
        x = jax.tree_map(lambda y: jnp.broadcast_to(y, (cfg.system.update_batch_size, *y.shape)), x)
        return jax.device_put_replicated(x, devices)

    env, eval_env = environments.make(cfg, add_global_state=True)

    n_agents = env.action_spec().shape[0]
    action_dim = env.action_spec().shape[1]

    key, actor_key, q1_key, q2_key, q1_target_key, q2_target_key = jax.random.split(key, 6)

    acts = env.action_spec().generate_value()  # all agents actions
    act_single = acts[0]  # single agents action
    concat_acts = jnp.concatenate([act_single for _ in range(n_agents)], axis=0)
    concat_acts_batched = concat_acts[jnp.newaxis, ...]  # batch + concat of all agents actions
    obs = env.observation_spec().generate_value()
    obs_single_batched = jax.tree_map(lambda x: x[0][jnp.newaxis, ...], obs)

    # Making actor network
    actor_torso = hydra.utils.instantiate(cfg.network.actor_network.pre_torso)
    actor_action_head = hydra.utils.instantiate(
        cfg.network.action_head, action_dim=env.action_dim, independent_std=False
    )
    actor = Actor(actor_torso, actor_action_head)
    actor_params = actor.init(actor_key, obs_single_batched)

    # Making Q networks
    critic_torso = hydra.utils.instantiate(cfg.network.critic_network.pre_torso)
    q = QNetwork(critic_torso, centralised_critic=True)
    q1_params = q.init(q1_key, obs_single_batched, concat_acts_batched)
    q2_params = q.init(q2_key, obs_single_batched, concat_acts_batched)
    q1_target_params = q.init(q1_target_key, obs_single_batched, concat_acts_batched)
    q2_target_params = q.init(q2_target_key, obs_single_batched, concat_acts_batched)

    # Automatic entropy tuning
    target_entropy = -cfg.system.target_entropy_scale * action_dim
    target_entropy = jnp.repeat(target_entropy, n_agents).astype(float)
    # making sure we have dim=3 so broacasting works fine
    target_entropy = target_entropy[jnp.newaxis, :]
    if cfg.system.autotune:
        log_alpha = jnp.zeros_like(target_entropy)
    else:
        log_alpha = jnp.log(cfg.system.init_alpha)
        log_alpha = jnp.broadcast_to(log_alpha, target_entropy.shape)

    # Pack params
    online_q_params = Qs(q1_params, q2_params)
    target_q_params = Qs(q1_target_params, q2_target_params)
    params = SacParams(actor_params, QsAndTarget(online_q_params, target_q_params), log_alpha)

    # Make opt states.
    actor_opt = optax.adam(cfg.system.policy_lr)
    actor_opt_state = actor_opt.init(params.actor)

    q_opt = optax.adam(cfg.system.q_lr)
    q_opt_state = q_opt.init(params.q.online)

    alpha_opt = optax.adam(cfg.system.alpha_lr)
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
        max_length=cfg.system.buffer_size,
        min_length=cfg.system.explore_steps,
        sample_batch_size=cfg.system.batch_size,
        add_batches=True,
    )
    buffer_state = replicate(rb.init(init_transition))

    nns = (actor, q)
    opts = (actor_opt, q_opt, alpha_opt)

    # Reset env.
    n_keys = cfg.system.n_envs * cfg.arch.n_devices * cfg.system.update_batch_size
    key_shape = (cfg.arch.n_devices, cfg.system.update_batch_size, cfg.system.n_envs, -1)
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, n_keys)
    reset_keys = jnp.reshape(reset_keys, key_shape)

    # Keys passed to learner
    first_keys = jax.random.split(key, (cfg.arch.n_devices * cfg.system.update_batch_size))
    first_keys = first_keys.reshape((cfg.arch.n_devices, cfg.system.update_batch_size, -1))

    env_state, first_timestep = jax.pmap(  # devices
        jax.vmap(  # update_batch_size
            jax.vmap(env.reset),  # n_envs
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
    return (env, eval_env), nns, opts, rb, learner_state, target_entropy, logger, key


def make_update_fns(
    cfg: DictConfig,
    env: Environment,
    nns: Networks,
    opts: Optimisers,
    rb: TrajectoryBuffer,
    target_entropy: chex.Array,
) -> Tuple[
    Callable[[LearnerState], Tuple[LearnerState, Metrics]],
    Callable[[LearnerState], Tuple[LearnerState, Tuple[Metrics, Metrics]]],
]:
    actor, q = nns
    actor_opt, q_opt, alpha_opt = opts

    full_action_shape = (cfg.system.n_envs, *env.action_spec().shape)

    n_agents = env.action_spec().shape[0]

    def step(
        action: Array, obs: Observation, env_state: State, buffer_state: BufferState
    ) -> Tuple[Array, State, BufferState, Dict]:
        """Given an action, step the environment and add to the buffer."""
        env_state, timestep = jax.vmap(env.step)(env_state, action)
        next_obs = timestep.observation
        rewards = timestep.reward
        terms = ~(timestep.discount).astype(bool)
        infos = timestep.extras

        real_next_obs = infos["real_next_obs"]

        transition = Transition(obs, action, rewards, terms, real_next_obs)
        buffer_state = rb.add(buffer_state, transition)

        return next_obs, env_state, buffer_state, infos["episode_metrics"]

    # losses:
    def q_loss_fn(q_params: Qs, obs: Array, action: Array, target: Array) -> Tuple[Array, Metrics]:
        q1_params, q2_params = q_params
        joint_action = get_joint_action(action)

        q1_a_values = q.apply(q1_params, obs, joint_action)
        q2_a_values = q.apply(q2_params, obs, joint_action)

        q1_loss = jnp.mean((q1_a_values - target) ** 2)
        q2_loss = jnp.mean((q2_a_values - target) ** 2)

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
        q_params: Qs,
        key: chex.PRNGKey,
    ) -> Array:
        pi = actor.apply(actor_params, obs)
        new_action = pi.sample(seed=key)
        log_prob = pi.log_prob(new_action)

        # Repeat the actions from the replay buffer such that you have (B, Ag, Ag, Ac).
        # This gives you n_agent joint actions with the action dim kept separate.
        # Then replace along the diagonal with the new action from the policy.
        # This replacement means that joint_action_i will have the new action for agent_i.
        actions_repeated = jnp.tile(actions[:, jnp.newaxis, ...], (1, n_agents, 1, 1))
        inds = jnp.diag_indices_from(actions_repeated[0, ..., 0])
        spliced_actions = actions_repeated.at[:, inds[0], inds[1], :].set(new_action)
        joint_actions = spliced_actions.reshape((*spliced_actions.shape[:2], -1))

        qf1_pi = q.apply(q_params.q1, obs, joint_actions)
        qf2_pi = q.apply(q_params.q2, obs, joint_actions)

        min_qf_pi = jnp.minimum(qf1_pi, qf2_pi)

        return ((alpha * log_prob) - min_qf_pi).mean()

    def alpha_loss_fn(log_alpha: Array, log_pi: Array, target_entropy: Array) -> Array:
        return jnp.mean(-jnp.exp(log_alpha) * (log_pi + target_entropy))

    # Update functions:
    def update_q(
        params: SacParams, opt_states: OptStates, data: Transition, key: chex.PRNGKey
    ) -> Tuple[SacParams, OptStates, Metrics]:
        """Update the Q parameters."""
        # Calculate Q target values.
        pi = actor.apply(params.actor, data.next_obs)
        next_action = pi.sample(seed=key)
        next_log_prob = pi.log_prob(next_action)

        joint_next_actions = get_joint_action(next_action)
        next_q1_val = q.apply(params.q.targets.q1, data.next_obs, joint_next_actions)
        next_q2_val = q.apply(params.q.targets.q2, data.next_obs, joint_next_actions)
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
        q_and_target = QsAndTarget(new_online_q_params, new_target_q_params)
        params = params._replace(q=q_and_target)
        opt_states = opt_states._replace(q=new_q_opt_state)

        return params, opt_states, q_loss_info

    def update_actor_and_alpha(
        params: SacParams, opt_states: OptStates, data: Transition, key: chex.PRNGKey
    ) -> Tuple[SacParams, OptStates, Metrics]:
        """Update the actor and alpha parameters. Compensated for the delay in policy updates."""
        # compensate for the delay by doing `policy_frequency` updates instead of 1.
        assert cfg.system.policy_frequency > 0, "Need to have a policy frequency > 0."
        for _ in range(cfg.system.policy_frequency):
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
                pi = actor.apply(params.actor, data.obs)
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
    def update_epoch(
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
            t % cfg.system.policy_frequency == 0,  # TD 3 Delayed update support
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

    def act(
        carry: Tuple[FrozenVariableDict, Array, State, BufferState, chex.PRNGKey], _: Any
    ) -> Tuple[Tuple[FrozenVariableDict, Array, State, BufferState, chex.PRNGKey], Dict]:
        """Acting loop: select action, step env, add to buffer."""
        actor_params, obs, env_state, buffer_state, key = carry

        pi = actor.apply(actor_params, obs)
        action = pi.sample(seed=key)

        next_obs, env_state, buffer_state, metrics = step(action, obs, env_state, buffer_state)
        return (actor_params, next_obs, env_state, buffer_state, key), metrics

    def explore(carry: LearnerState, _: Any) -> Tuple[LearnerState, Metrics]:
        """Take random actions to fill up buffer at the start of training."""
        obs, env_state, buffer_state, _, _, t, key = carry
        key, explore_key = jax.random.split(key)
        action = jax.random.uniform(explore_key, full_action_shape)
        next_obs, env_state, buffer_state, metrics = step(action, obs, env_state, buffer_state)

        t += cfg.system.n_envs
        learner_state = carry._replace(
            obs=next_obs, env_state=env_state, buffer_state=buffer_state, t=t, key=key
        )
        return learner_state, metrics

    scanned_update = lambda state: lax.scan(update_epoch, state, None, length=cfg.system.epochs)
    scanned_act = lambda state: lax.scan(act, state, None, length=cfg.system.rollout_length)

    # Act loop -> sample -> update loop
    def update_step(carry: LearnerState, _: Any) -> Tuple[LearnerState, Tuple[Metrics, Metrics]]:
        """Act, sample, learn."""

        obs, env_state, buffer_state, params, opt_states, t, key = carry
        key, act_key, learn_key = jax.random.split(key, 3)
        # Act
        act_state = (params.actor, obs, env_state, buffer_state, act_key)
        (_, next_obs, env_state, buffer_state, _), metrics = scanned_act(act_state)

        # Sample and learn
        learn_state = (buffer_state, params, opt_states, t, learn_key)
        (buffer_state, params, opt_states, _, _), losses = scanned_update(learn_state)

        t += cfg.system.n_envs * cfg.system.rollout_length
        return (
            LearnerState(next_obs, env_state, buffer_state, params, opt_states, t, key),
            (metrics, losses),
        )

    # pmap and scan over explore and update_step
    # Make sure to not do n_envs explore steps (could fill up the buffer too much).
    explore_steps = cfg.system.explore_steps // cfg.system.n_envs
    pmaped_explore = jax.pmap(
        jax.vmap(
            lambda state: lax.scan(explore, state, None, length=explore_steps),
            axis_name="batch",
        ),
        axis_name="device",
        donate_argnums=0,
    )
    pmaped_updated_step = jax.pmap(
        jax.vmap(
            lambda state: lax.scan(update_step, state, None, length=cfg.system.scan_steps),
            axis_name="batch",
        ),
        axis_name="device",
        donate_argnums=0,
    )

    return pmaped_explore, pmaped_updated_step


def run_experiment(cfg: DictConfig) -> float:
    # Add runtime variables to config
    cfg.arch.n_devices = len(jax.devices())

    # Number of env steps before evaluating/logging.
    steps_per_rollout = int(cfg.system.total_timesteps // cfg.arch.num_evaluation)
    # Multiplier for a single env/learn step in an anakin system
    anakin_steps = cfg.arch.n_devices * cfg.system.update_batch_size
    # Number of env steps in one anakin style update.
    anakin_act_steps = anakin_steps * cfg.system.n_envs * cfg.system.rollout_length
    # Number of steps to do in the scanned update method (how many anakin steps).
    cfg.system.scan_steps = int(steps_per_rollout / anakin_act_steps)

    (env, eval_env), nns, opts, rb, learner_state, target_entropy, logger, key = init(cfg)
    explore, update = make_update_fns(cfg, env, nns, opts, rb, target_entropy)

    actor, _ = nns
    key, eval_key = jax.random.split(key)
    evaluator, absolute_metric_evaluator = make_eval_fns(
        eval_env=eval_env,
        network=actor,
        config=cfg,
    )

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
    final_metrics, ep_completed = episode_metrics.get_final_step_metrics(metrics)

    logger.log({"step": t, "steps_per_second": sps}, t, 0, LogEvent.MISC)
    if ep_completed:
        logger.log(final_metrics, cfg.system.explore_steps, 0, LogEvent.ACT)

    # Main loop:
    # We want start to align with the final step of the first pmaped_learn,
    # where we've done explore_steps and 1 full learn step.
    start = cfg.system.explore_steps + steps_per_rollout
    for eval_idx, t in enumerate(range(start, int(cfg.system.total_timesteps), steps_per_rollout)):
        # Learn loop:
        learner_state, (metrics, losses) = update(learner_state)
        jax.block_until_ready(learner_state)

        # Log:
        # Add learn steps here because anakin steps per second is learn + act steps
        # But we also want to make sure we're counting env steps correctly so
        # learn steps is not included in the loop counter.
        learn_steps = anakin_steps * cfg.system.epochs
        sps = (t + learn_steps) / (time.time() - start_time)
        final_metrics, ep_completed = episode_metrics.get_final_step_metrics(metrics)
        loss_metrics = losses | {"log_alpha": learner_state.params.log_alpha}

        logger.log({"step": t, "steps_per_second": sps}, t, eval_idx, LogEvent.MISC)
        if ep_completed:
            logger.log(final_metrics, t, eval_idx, LogEvent.ACT)
        logger.log(loss_metrics, t, eval_idx, LogEvent.TRAIN)

        # Evaluate:
        key, eval_key = jax.random.split(key)
        eval_keys = jax.random.split(eval_key, cfg.arch.n_devices)
        eval_output = evaluator(unreplicate_batch_dim(learner_state.params.actor), eval_keys)
        jax.block_until_ready(eval_output)

        # Log:
        episode_return = jnp.mean(eval_output.episode_metrics["episode_return"])
        logger.log(eval_output.episode_metrics, t, eval_idx, LogEvent.EVAL)

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

    # Measure absolute metric.
    if cfg.arch.absolute_metric:
        eval_keys = jax.random.split(key, cfg.arch.n_devices)

        eval_output = absolute_metric_evaluator(best_params, eval_keys)
        jax.block_until_ready(eval_output)

        logger.log(eval_output.episode_metrics, t, eval_idx, LogEvent.ABSOLUTE)

    logger.stop()

    return float(max_episode_return)


@hydra.main(config_path="../../configs", config_name="default_ff_isac.yaml", version_base="1.2")
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    final_return = run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}ISAC experiment completed{Style.RESET_ALL}")

    return float(final_return)


if __name__ == "__main__":
    hydra_entry_point()
