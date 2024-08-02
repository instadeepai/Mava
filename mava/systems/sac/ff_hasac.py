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
from rich.pretty import pprint

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
    QVals,
    QValsAndTarget,
    SacParams,
    Transition,
)
from mava.types import ObservationGlobalState
from mava.utils import make_env as environments
from mava.utils.centralised_training import get_joint_action
from mava.utils.checkpointing import Checkpointer
from mava.utils.jax_utils import (
    tree_at_set,
    tree_slice,
    unreplicate_batch_dim,
    unreplicate_n_dims,
)
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.total_timestep_checker import check_total_timesteps
from mava.wrappers import episode_metrics


def get_action(actor_params, actor_net, keys, env, obs, batch_size):
    actions = jnp.zeros((batch_size, env.num_agents, env.action_dim))
    log_std = jnp.zeros((batch_size, env.num_agents))

    for agent in range(env.num_agents):
        actor_params_per_agent = jax.tree_util.tree_map(
            lambda x, agent=agent: x[agent], actor_params
        )
        obs_per_agent = jax.tree_util.tree_map(lambda x, agent=agent: x[:, agent], obs)

        pi = actor_net.apply(actor_params_per_agent, obs_per_agent)
        action = pi.sample(seed=keys[agent])
        actions.at[:, agent].set(action)
        log_std.at[:, agent].set(pi.log_prob(action))

    return actions, log_std


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
    """Initialize system by creating the envs, networks etc.

    Args:
        cfg: System configuration.

    Returns:
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
        x = jax.tree_map(lambda y: jnp.broadcast_to(y, (cfg.system.update_batch_size, *y.shape)), x)
        return jax.device_put_replicated(x, devices)

    env, eval_env = environments.make(cfg, add_global_state=True)

    n_agents = env.num_agents
    action_dim = env.action_dim

    key, actor_key, q1_key, q2_key, q1_target_key, q2_target_key = jax.random.split(key, 6)
    actor_keys = jax.random.split(actor_key, n_agents)

    acts = env.action_spec().generate_value()  # all agents actions
    act_single = acts[0]  # single agents action
    concat_acts = jnp.concatenate([act_single for _ in range(n_agents)], axis=0)
    concat_acts_batched = concat_acts[jnp.newaxis, ...]  # batch + concat of all agents actions
    obs = env.observation_spec().generate_value()
    obs_single_batched = jax.tree_map(lambda x: x[0][jnp.newaxis, ...], obs)

    # Making actor network
    actor_torso = hydra.utils.instantiate(cfg.network.actor_network.pre_torso)
    actor_action_head = hydra.utils.instantiate(
        cfg.network.action_head, action_dim=action_dim, independent_std=False
    )
    actor_network = Actor(actor_torso, actor_action_head)
    actor_params = jax.vmap(actor_network.init, in_axes=(0, None))(actor_keys, obs_single_batched)

    # Making Q networks
    critic_torso = hydra.utils.instantiate(cfg.network.critic_network.pre_torso)
    q_network = QNetwork(critic_torso, centralised_critic=True)
    q1_params = q_network.init(q1_key, obs_single_batched, concat_acts_batched)
    q2_params = q_network.init(q2_key, obs_single_batched, concat_acts_batched)
    q1_target_params = q_network.init(q1_target_key, obs_single_batched, concat_acts_batched)
    q2_target_params = q_network.init(q2_target_key, obs_single_batched, concat_acts_batched)

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
    actor_opt = optax.adam(cfg.system.policy_lr)
    actor_opt_state = jax.vmap(actor_opt.init)(params.actor)

    q_opt = optax.adam(cfg.system.q_lr)
    q_opt_state = q_opt.init(params.q.online)

    alpha_opt = optax.adam(cfg.system.alpha_lr)
    alpha_opt_state = jax.vmap(alpha_opt.init, in_axes=1)(params.log_alpha)

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
    env: Environment,
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
        cfg: System configuration.
        env: The environment.
        networks: Tuple of actor and critic networks.
        optims: Tuple of actor, critic and alpha optimisers.
        rb: The replay buffer.
        target_entropy: The target entropy.

    Returns:
        Tuple of (explore_fn, update_fn).
        Explore function is used for initial exploration with random actions.
        Update function is the main learning function, it both acts and learns.
    """
    actor_net, q_net = networks
    actor_opt, q_opt, alpha_opt = optims

    full_action_shape = (cfg.arch.num_envs, *env.action_spec().shape)

    # todo: move this to where stepping loop is
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
        agent_id,
    ) -> Array:
        B, A, _ = actions.shape
        pi = actor_net.apply(actor_params, obs)
        new_actions = pi.sample(seed=key)
        log_prob = pi.log_prob(new_actions)

        # Updated joint actions are done so that each agent's central critic sees what all
        # other agents did in the past, but it sees how its agent's policy is currently acting.
        # This is done by placing new_action[i] in joint_actions[i].
        # [32, 4, 2] -> insert -> [32, 8]
        joint_actions = actions.at[:, agent_id, :].set(new_actions).reshape(B, -1)
        # joint_actions = get_updated_joint_actions(actions, new_actions)

        qval_1 = q_net.apply(q_params.q1, obs, joint_actions)
        qval_2 = q_net.apply(q_params.q2, obs, joint_actions)
        min_q_val = jnp.minimum(qval_1, qval_2)

        # todo: hasac uses only 1 alpha!
        return ((alpha[:, agent_id] * log_prob) - min_q_val).mean()

    def alpha_loss_fn(log_alpha: Array, log_pi: Array, target_entropy: Array) -> Array:
        return jnp.mean(-jnp.exp(log_alpha) * (log_pi + target_entropy))

    # Update functions:
    def update_q(
        params: SacParams, opt_states: OptStates, data: Transition, key: chex.PRNGKey
    ) -> Tuple[SacParams, OptStates, Metrics]:
        """Update the Q parameters."""
        # Calculate Q target values.
        # pi = actor_net.apply(params.actor, data.next_obs)
        # next_action = pi.sample(seed=key)
        # next_log_prob = pi.log_prob(next_action)
        act_keys = jax.random.split(key, env.num_agents)
        next_action, next_log_prob = get_action(
            params.actor, actor_net, act_keys, env, data.next_obs, cfg.system.batch_size
        )

        joint_next_actions = get_joint_action(next_action)
        next_q1_val = q_net.apply(params.q.targets.q1, data.next_obs, joint_next_actions)
        next_q2_val = q_net.apply(params.q.targets.q2, data.next_obs, joint_next_actions)
        next_q_val = jnp.minimum(next_q1_val, next_q2_val)
        # todo: look into this -> hasac does 1 alpha and sums the log probs
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
            key, act_key, agent_order_key = jax.random.split(key, 3)
            act_keys = jax.random.split(act_key, env.num_agents)
            agent_ids = jax.random.permutation(agent_order_key, env.num_agents)

            # todo: we can almost certainly get this from the buffer, we just need the log probs for alpha :/
            actions, log_probs = get_action(
                params.actor, actor_net, act_keys, env, data.obs, cfg.system.batch_size
            )

            # TODO: sequential update - this is not working yet!
            # Everything other than the actor/alpha update should be done
            # What needs to happen here:
            # Do an actor update where we compute the action as above but only for the current agent.
            # Add this action to the list of actions/logprobs and get the gradient of just that action and update the actor.
            # Now update this list using a new action from the updated actor and continue looping through agents using this updated list of actions.
            # Finally update each alpha with its own optimiser after updating the policy.

            # Important parts of the code HASAC:
            # Get "old" actions (done line 379 above): https://github.com/PKU-MARL/HARL/blob/main/harl/runners/off_policy_ha_runner.py#L81-L92
            # Turn on grads and get new action and logprob: https://github.com/PKU-MARL/HARL/blob/main/harl/runners/off_policy_ha_runner.py#L100-L109
            # Place action + LP the list of actions + LPs: https://github.com/PKU-MARL/HARL/blob/main/harl/runners/off_policy_ha_runner.py#L110-L119
            # Compute policy loss as normal: https://github.com/PKU-MARL/HARL/blob/main/harl/runners/off_policy_ha_runner.py#L142-L144
            # Autotune alpha using current agents logprob: https://github.com/PKU-MARL/HARL/blob/main/harl/runners/off_policy_ha_runner.py#L151-L155
            # Update *only* the actions in the list of actions using the updated policy: https://github.com/PKU-MARL/HARL/blob/main/harl/runners/off_policy_ha_runner.py#L162-L169
            for agent_id in agent_ids:
                actor_key, alpha_key = jax.random.split(key)

                agent_params = tree_slice(params.actor, agent_id)
                agent_opt_state = tree_slice(opt_states.actor, agent_id)
                agent_obs = tree_slice(data.obs, jnp.s_[:, agent_id])

                # Update actor.
                actor_grad_fn = jax.value_and_grad(actor_loss_fn)
                actor_loss, act_grads = actor_grad_fn(
                    agent_params,
                    agent_obs,
                    actions,
                    jnp.exp(params.log_alpha),
                    params.q.online,
                    actor_key,
                    agent_id,
                )
                # Mean over the device and batch dimensions.
                actor_loss, act_grads = lax.pmean((actor_loss, act_grads), axis_name="device")
                actor_loss, act_grads = lax.pmean((actor_loss, act_grads), axis_name="batch")
                actor_updates, new_actor_opt_state = actor_opt.update(act_grads, agent_opt_state)
                new_actor_params = optax.apply_updates(agent_params, actor_updates)

                # update actions list with new action from updated actor
                pi = actor_net.apply(new_actor_params, agent_obs)
                new_action = pi.sample(seed=key)

                # Add new action to list of actions
                actions = actions.at[:, agent_id].set(new_action)

                all_actor_params = tree_at_set(params.actor, agent_id, new_actor_params)
                all_opt_states = tree_at_set(opt_states.actor, agent_id, new_actor_opt_state)
                params = params._replace(actor=all_actor_params)
                opt_states = opt_states._replace(actor=all_opt_states)

                # Update alpha if autotuning
                alpha_loss = 0.0  # loss is 0 if autotune is off
                if cfg.system.autotune:
                    # Get log prob for alpha loss
                    # pi = actor_net.apply(params.actor, data.obs)
                    # action = pi.sample(seed=key)
                    # log_prob = pi.log_prob(action)
                    alpha_opt_state = tree_slice(opt_states.alpha, agent_id)

                    alpha_grad_fn = jax.value_and_grad(alpha_loss_fn)
                    alpha_loss, alpha_grads = alpha_grad_fn(
                        params.log_alpha[:, agent_id],
                        log_probs[:, agent_id],
                        target_entropy[:, agent_id],
                    )
                    alpha_loss, alpha_grads = lax.pmean(
                        (alpha_loss, alpha_grads), axis_name="device"
                    )
                    alpha_loss, alpha_grads = lax.pmean(
                        (alpha_loss, alpha_grads), axis_name="batch"
                    )
                    alpha_updates, new_alpha_opt_state = alpha_opt.update(
                        alpha_grads, alpha_opt_state
                    )
                    new_log_alpha = optax.apply_updates(
                        params.log_alpha[:, agent_id], alpha_updates
                    )

                    new_log_alphas = tree_at_set(params.log_alpha, agent_id, new_log_alpha)
                    new_alpha_opt_states = tree_at_set(
                        opt_states.alpha, agent_id, new_alpha_opt_state
                    )
                    params = params._replace(log_alpha=new_log_alphas)
                    opt_states = opt_states._replace(alpha=new_alpha_opt_states)

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

    def act(
        carry: Tuple[FrozenVariableDict, Array, State, BufferState, chex.PRNGKey], _: Any
    ) -> Tuple[Tuple[FrozenVariableDict, Array, State, BufferState, chex.PRNGKey], Dict]:
        """Acting loop: select action, step env, add to buffer."""
        actor_params, obs, env_state, buffer_state, key = carry
        key, act_key = jax.random.split(key)
        act_keys = jax.random.split(act_key, env.num_agents)

        actions, _ = get_action(actor_params, actor_net, act_keys, env, obs, cfg.arch.num_envs)

        next_obs, env_state, buffer_state, metrics = step(actions, obs, env_state, buffer_state)
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
    # evaluator, absolute_metric_evaluator = make_eval_fns(eval_env, actor.apply, cfg)

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
    logger.log({"step": t, "steps_per_second": sps}, t, 0, LogEvent.MISC)

    # Don't mind if episode isn't completed here, nice to have the graphs start near 0.
    # So we ignore the second return value.
    final_metrics, _ = episode_metrics.get_final_step_metrics(metrics)
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
        # key, eval_key = jax.random.split(key)
        # eval_keys = jax.random.split(eval_key, cfg.arch.n_devices)
        # eval_output = evaluator(unreplicate_batch_dim(learner_state.params.actor), eval_keys)
        # jax.block_until_ready(eval_output)

        # Log:
        # episode_return = jnp.mean(eval_output.episode_metrics["episode_return"])
        # logger.log(eval_output.episode_metrics, t, eval_idx, LogEvent.EVAL)

        # Save best actor params.
        # if cfg.arch.absolute_metric and max_episode_return <= episode_return:
        #     best_params = copy.deepcopy(unreplicate_batch_dim(learner_state.params.actor))
        #     max_episode_return = episode_return

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
    # eval_performance = float(jnp.mean(eval_output.episode_metrics[cfg.env.eval_metric]))

    # Measure absolute metric.
    # if cfg.arch.absolute_metric:
    #     eval_keys = jax.random.split(key, cfg.arch.n_devices)
    #
    #     eval_output = absolute_metric_evaluator(best_params, eval_keys)
    #     jax.block_until_ready(eval_output)
    #
    #     logger.log(eval_output.episode_metrics, t, eval_idx, LogEvent.ABSOLUTE)

    logger.stop()

    # return eval_performance
    return 0


@hydra.main(config_path="../../configs", config_name="default_ff_hasac.yaml", version_base="1.2")
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
