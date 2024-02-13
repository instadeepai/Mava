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
from typing import Any, Callable, Dict, NamedTuple, Tuple

import chex
import distrax
import flashbax as fbx
import flax.linen as nn
import hydra
import jax
import jax.lax as lax
import jax.numpy as jnp
import optax
from chex import PRNGKey
from colorama import Fore, Style
from flashbax.buffers.flat_buffer import TrajectoryBuffer
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
from flax.core.scope import FrozenVariableDict
from jax import Array
from jax.typing import ArrayLike
from jumanji.env import Environment, State
from omegaconf import DictConfig, OmegaConf
from typing_extensions import TypeAlias

from mava.evaluator import evaluator_setup
from mava.types import Observation
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.jax import unreplicate_batch_dim, unreplicate_learner_state
from mava.utils.logger import LogEvent, MavaLogger
from mava.wrappers import episode_metrics

# jax.config.update("jax_platform_name", "cpu")


# todo: types.py
class Qs(NamedTuple):
    # q1: FrozenVariableDict
    # q2: FrozenVariableDict
    q: FrozenVariableDict

class QsAndTarget(NamedTuple):
    online: Qs
    targets: Qs


# class SacParams(NamedTuple):
#     actor: FrozenVariableDict
#     q: QsAndTarget
#     log_alpha: Array
    
class QLearnParams(NamedTuple):
    dqns:QsAndTarget
    # NOTE (Louise) later add Qmix network


class OptStates(NamedTuple):
    # actor: optax.OptState
    q: optax.OptState
    # alpha: optax.OptState


class LearnerState(NamedTuple):
    obs: Array
    env_state: State
    buffer_state: TrajectoryBuffer
    params: QLearnParams
    opt_states: OptStates
    t: Array
    key: PRNGKey


class Transition(NamedTuple):
    obs: Array
    action: Array
    reward: Array
    done: Array
    next_obs: Array


Metrics = Dict[str, Array]
BufferState: TypeAlias = TrajectoryBufferState[Transition]
Networks: TypeAlias = Tuple[nn.Module, nn.Module]
Optimisers: TypeAlias = Tuple[
    optax.GradientTransformation#, optax.GradientTransformation, optax.GradientTransformation
]


# todo: jumanji PR


# todo: should this contain both networks?
num_actions = ... # TODO (Claude) we need to pass the number of actions at setup time NOTE (Louise) done! From your other code
class QNetwork(nn.Module):
    num_actions:int
    def setup(self) -> None:
        # self.net = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(256), nn.relu, nn.Dense(1)])
        self.net = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(256), nn.relu, nn.Dense(self.num_actions)])


    @nn.compact
    def __call__(self, obs: Observation) -> Array:
        x = obs.agents_view
        return self.net(x)


# rlax does: 0, -4 NOTE (Claude) we dont need this stuff. Its for stocastic policies
# LOG_STD_MAX = 2
# LOG_STD_MIN = -5


# class Actor(nn.Module):
#     n_actions: int
#     maximum: Array
#     minimum: Array

#     def setup(self) -> None:
#         self.torso = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(256), nn.relu])
#         self.mean_nn = nn.Dense(self.n_actions)
#         self.logstd_nn = nn.Dense(self.n_actions)

#         scale = 0.5 * (self.maximum - self.minimum)
#         self.bijector = distrax.Chain(
#             [
#                 distrax.ScalarAffine(shift=self.minimum, scale=scale),
#                 distrax.ScalarAffine(shift=1.0),
#                 distrax.Tanh(),
#             ]
#         )

#     @nn.compact
#     def __call__(self, obs: Observation) -> distrax.Distribution:
#         x = self.torso(obs.agents_view)

#         mean = self.mean_nn(x)

#         log_std = self.logstd_nn(x)
#         log_std = jnp.tanh(log_std)
#         # From SpinUp / Denis Yarats
#         log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
#         std = jnp.exp(log_std)

#         pi = distrax.Transformed(
#             distribution=distrax.MultivariateNormalDiag(loc=mean, scale_diag=std),
#             bijector=distrax.Block(self.bijector, ndims=1),
#         )

#         return pi


# # todo: inside actor? NOTE (Claude) we need to make a method like this that uses Q-values to select actions
# def sample_action(
#     mean: ArrayLike,
#     log_std: ArrayLike,
#     key: PRNGKey,
#     action_scale: Array,
#     action_bias: Array,
#     eval: bool = False,
# ) -> Tuple[Array, Array]:
#     std = jnp.exp(log_std)
#     # normal = distrax.Normal(mean, std)
#     normal = distrax.MultivariateNormalDiag(mean, std)

#     sampled_action = lax.cond(
#         eval,
#         lambda: mean,
#         lambda: normal.sample(seed=key),
#     )
#     bound_action = jnp.tanh(sampled_action)
#     scaled_action = bound_action * action_scale + action_bias

#     # log_prob = normal.log_prob(unbound_action)
#     # rescale_term = jnp.log(action_scale * (1 - bound_action**2) + 1e-6).sum(axis=-1)
#     # log_prob = log_prob - rescale_term

#     # todo: scale
#     log_prob = normal.log_prob(sampled_action)
#     rescal_term = 2 * (jnp.log(2) - sampled_action - jax.nn.softplus(-2 * sampled_action))
#     log_prob = log_prob - jnp.sum(rescal_term, axis=-1)

#     return scaled_action, log_prob[..., jnp.newaxis]


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
    n_devices = len(devices)

    def replicate(x: Any) -> Any:
        """First replicate the update batch dim then put on devices."""
        x = jax.tree_map(lambda y: jnp.broadcast_to(y, (cfg.system.update_batch_size, *y.shape)), x)
        return jax.device_put_replicated(x, devices)

    env, eval_env = environments.make(cfg)

    # n_agents = env.action_spec().shape[0]
    # action_dim = env.action_spec().shape[1] # TODO (CLaude) get the number of actions here
    num_actions = int(env.action_spec().num_values[0]) # NOTE (Claude) got this from the PPO systems

    # act_high = jnp.zeros(env.action_spec().shape) + env.action_spec().maximum
    # act_low = jnp.zeros(env.action_spec().shape) + env.action_spec().minimum

    # key, q1_key, q2_key, q1_target_key, q2_target_key = jax.random.split(key, 5) # TODO (Claude) remove the second q function
    key, q_key, q_target_key = jax.random.split(key, 3)

    init_obs = env.observation_spec().generate_value()
    init_acts = env.action_spec().generate_value()
    init_obs_batched = jax.tree_map(lambda x: x[0][jnp.newaxis, ...], init_obs)
    # init_act_batched = jax.tree_map(lambda x: x[jnp.newaxis, ...], init_acts)

    # Making actor network
    # actor = Actor(action_dim, act_high, act_low)
    # actor_params = actor.init(actor_key, init_obs_batched)

    # Making Q networks
    q = QNetwork(num_actions)
    q_params = q.init(q_key, init_obs_batched)
    # q2_params = q.init(q2_key, init_obs_batched, init_acts_batched)
    q_target_params = q.init(q_target_key, init_obs_batched)
    # q2_target_params = q.init(q2_target_key, init_obs_batched, init_acts_batched)

    # Automatic entropy tuning
    # target_entropy = -cfg.system.target_entropy_scale * action_dim
    # target_entropy = jnp.repeat(target_entropy, n_agents).astype(float)
    # making sure we have dim=3 so broacasting works fine
    # target_entropy = target_entropy[jnp.newaxis, :, jnp.newaxis]
    # if cfg.system.autotune:
    #     log_alpha = jnp.zeros_like(target_entropy)
    # else:
    #     log_alpha = jnp.log(cfg.system.init_alpha)
    #     log_alpha = jnp.broadcast_to(log_alpha, target_entropy.shape)

    # Pack params
    params = QLearnParams(QsAndTarget(Qs(q_params), Qs(q_target_params)))
    # target_q_params = Qs(q1_target_params, q2_target_params)
    # params = SacParams(actor_params, QsAndTarget(online_q_params, target_q_params), log_alpha)

    # # Make opt states.
    # actor_opt = optax.adam(cfg.system.policy_lr)
    # actor_opt_state = actor_opt.init(params.actor)

    q_opt = optax.adam(cfg.system.q_lr)
    q_opt_state = q_opt.init(params.dqns.online)

    # alpha_opt = optax.adam(cfg.system.q_lr)  # todo: alpha lr?
    # alpha_opt_state = alpha_opt.init(params.log_alpha)

    # # Pack opt states
    opt_states = OptStates(q_opt_state)

    # Distribute params and opt states across all devices
    params = replicate(params)
    opt_states = replicate(opt_states)

    # Create replay buffer
    init_transition = Transition(
        obs=init_obs,
        action=init_acts,
        # todo: n agents rewards/discounts
        reward=jnp.zeros((), dtype=float),
        done=jnp.zeros((), dtype=bool),
        next_obs=init_obs,
    )

    rb = fbx.make_item_buffer(
        max_length=cfg.system.buffer_size,
        min_length=cfg.system.explore_steps,
        sample_batch_size=cfg.system.batch_size,
        add_batches=True,
    )
    buffer_state = replicate(rb.init(init_transition))

    nns = (q, q) # NOTE (Louise) replace a q with mixer params later
    # opts = (actor_opt, q_opt, alpha_opt)

    # Reset env.
    n_keys = cfg.system.n_envs * n_devices * cfg.system.update_batch_size
    key_shape = (n_devices, cfg.system.update_batch_size, cfg.system.n_envs, -1)
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, n_keys)
    reset_keys = jnp.reshape(reset_keys, key_shape)

    # Keys passed to learner
    first_keys = jax.random.split(key, (n_devices * cfg.system.update_batch_size))
    first_keys = first_keys.reshape((n_devices, cfg.system.update_batch_size, -1))

    env_state, first_timestep = jax.pmap(  # devices
        jax.vmap(  # update_batch_size
            jax.vmap(env.reset),  # n_envs
            axis_name="batch",
        ),
        axis_name="device",
    )(reset_keys)
    first_obs = first_timestep.observation

    t = jnp.zeros((n_devices, cfg.system.update_batch_size), dtype=int)

    # Initial learner state.
    learner_state = LearnerState(
        first_obs, env_state, buffer_state, params, opt_states, t, first_keys
    )
    return (env, eval_env), nns, opt_states, rb, learner_state, logger, key # NOTE (Louise) deleted some things here, so need to refactor expected outputs


def make_update_fns(
    cfg: DictConfig,
    env: Environment,
    nns: Networks,
    opts: Optimisers,
    rb: TrajectoryBuffer,
    # target_entropy: chex.Array,
) -> Tuple[
    Callable[[LearnerState], Tuple[LearnerState, Metrics]],
    Callable[[LearnerState], Tuple[LearnerState, Tuple[Metrics, Metrics]]],
]:
    _, q = nns # NOTE (Louise) put mixer later
    q_opt = opts

    # full_action_shape = (cfg.system.n_envs, *env.action_spec().shape) # NOTE (Claude) I dont think this is right since we have discrete actions

    def step(
        action: Array, obs: Observation, env_state: State, buffer_state: BufferState
    ) -> Tuple[Array, State, BufferState, Dict]:
        """Given an action, step the environment and add to the buffer."""
        env_state, timestep = jax.vmap(env.step)(env_state, action)
        next_obs = timestep.observation
        rewards = timestep.reward[:, 0]
        terms = ~(timestep.discount).astype(bool)[:, 0]
        infos = timestep.extras

        real_next_obs = infos["real_next_obs"]

        transition = Transition(obs, action, rewards, terms, real_next_obs)
        buffer_state = rb.add(buffer_state, transition)

        return next_obs, env_state, buffer_state, infos["episode_metrics"]

    # losses:
    def q_loss_fn(q_params: QsAndTarget, obs: Array, action: Array, target: Array) -> Tuple[Array, Metrics]:
        q_online = q.apply(q_params.online, obs).reshape(-1)
        q_loss = jnp.mean((q_online - target) ** 2)

        loss_info = {
            "q_loss": q_loss,
            "mean_q": jnp.mean(q_online),
        }

        return q_loss, loss_info

    # def actor_loss_fn(
    #     actor_params: FrozenVariableDict, obs: Array, alpha: Array, q_params: Qs, key: chex.PRNGKey
    # ) -> Array:
    #     pi = actor.apply(actor_params, obs)
    #     action, log_prob = pi.sample_and_log_prob(seed=key)
    #     log_prob = log_prob[..., jnp.newaxis]

    #     qf1_pi = q.apply(q_params.q1, obs, action)
    #     qf2_pi = q.apply(q_params.q2, obs, action)
    #     min_qf_pi = jnp.minimum(qf1_pi, qf2_pi)

    #     return ((alpha * log_prob) - min_qf_pi).mean()

    # def alpha_loss_fn(log_alpha: Array, log_pi: Array, target_entropy: Array) -> Array:
    #     return jnp.mean(-jnp.exp(log_alpha) * (log_pi + target_entropy))

    # Update functions:
    def update_q(
        params: QsAndTarget, opt_states: OptStates, data: Transition, key: chex.PRNGKey
    ) -> Tuple[QsAndTarget, OptStates, Metrics]:
        """Update the Q parameters."""
        # Calculate Q target values.
        rewards = data.reward[..., jnp.newaxis, jnp.newaxis]
        dones = data.done[..., jnp.newaxis, jnp.newaxis]

        # pi = actor.apply(params.actor, data.next_obs)
        # next_action, next_log_prob = pi.sample_and_log_prob(seed=key)
        # next_log_prob = next_log_prob[..., jnp.newaxis]

        next_q_vals_online = q.apply(params.online, data.next_obs) #NOTE (Louise) not superduper efficient
        next_q_vals_target = q.apply(params.target, data.next_obs)

        # TODO (Claude) do double q-value selection here...
        next_action = jnp.argmax(next_q_vals_online)
        next_q_val = next_q_vals_target[next_action]

        target_q_val = (rewards + (1.0 - dones) * cfg.system.gamma * next_q_val).reshape(-1)

        # Update Q function.
        q_grad_fn = jax.grad(q_loss_fn, has_aux=True)
        q_grads, q_loss_info = q_grad_fn(params.online, data.obs, data.action, target_q_val)
        # Mean over the device and batch dimension.
        q_grads, q_loss_info = lax.pmean((q_grads, q_loss_info), axis_name="device")
        q_grads, q_loss_info = lax.pmean((q_grads, q_loss_info), axis_name="batch")
        q_updates, new_q_opt_state = q_opt.update(q_grads, opt_states.q)
        new_online_q_params = optax.apply_updates(params.q.online, q_updates)

        # Target network polyak update.
        new_target_q_params = optax.incremental_update(
            new_online_q_params, params.target, cfg.system.tau
        )

        # Repack params and opt_states.
        q_and_target = QsAndTarget(Qs(new_online_q_params), Qs(new_target_q_params))
        params = params._replace(q=q_and_target)
        opt_states = opt_states._replace(q=new_q_opt_state)

        return params, opt_states, q_loss_info

    # def update_actor_and_alpha(
    #     params: SacParams, opt_states: OptStates, data: Transition, key: chex.PRNGKey
    # ) -> Tuple[SacParams, OptStates, Metrics]:
    #     """Update the actor and alpha parameters. Compensated for the delay in policy updates."""
    #     # compensate for the delay by doing `policy_frequency` updates instead of 1.
    #     assert cfg.system.policy_frequency > 0, "Need to have a policy frequency > 0."
    #     for _ in range(cfg.system.policy_frequency):
    #         actor_key, alpha_key = jax.random.split(key)

    #         # Update actor.
    #         actor_grad_fn = jax.value_and_grad(actor_loss_fn)
    #         actor_loss, act_grads = actor_grad_fn(
    #             params.actor, data.obs, jnp.exp(params.log_alpha), params.q.online, actor_key
    #         )
    #         # Mean over the device and batch dimensions.
    #         actor_loss, act_grads = lax.pmean((actor_loss, act_grads), axis_name="device")
    #         actor_loss, act_grads = lax.pmean((actor_loss, act_grads), axis_name="batch")
    #         actor_updates, new_actor_opt_state = actor_opt.update(act_grads, opt_states.actor)
    #         new_actor_params = optax.apply_updates(params.actor, actor_updates)

    #         params = params._replace(actor=new_actor_params)
    #         opt_states = opt_states._replace(actor=new_actor_opt_state)

    #         # Update alpha if autotuning
    #         alpha_loss = 0.0  # loss is 0 if autotune is off
    #         if cfg.system.autotune:
    #             # Get log prob for alpha loss
    #             pi = actor.apply(params.actor, data.obs)
    #             _, log_prob = pi.sample_and_log_prob(seed=alpha_key)
    #             log_prob = log_prob[..., jnp.newaxis]

    #             alpha_grad_fn = jax.value_and_grad(alpha_loss_fn)
    #             alpha_loss, alpha_grads = alpha_grad_fn(params.log_alpha, log_prob, target_entropy)
    #             alpha_loss, alpha_grads = lax.pmean((alpha_loss, alpha_grads), axis_name="device")
    #             alpha_loss, alpha_grads = lax.pmean((alpha_loss, alpha_grads), axis_name="batch")
    #             alpha_updates, new_alpha_opt_state = alpha_opt.update(alpha_grads, opt_states.alpha)
    #             new_log_alpha = optax.apply_updates(params.log_alpha, alpha_updates)

    #             params = params._replace(log_alpha=new_log_alpha)
    #             opt_states = opt_states._replace(alpha=new_alpha_opt_state)

    #     loss_info = {"actor_loss": actor_loss, "alpha_loss": alpha_loss}
    #     return params, opt_states, loss_info

    # Act/learn loops:
    def update_epoch(
        carry: Tuple[BufferState, QsAndTarget, OptStates, int, chex.PRNGKey], _: Any
    ) -> Tuple[Tuple[BufferState, QsAndTarget, OptStates, int, chex.PRNGKey], Metrics]:
        """Update the Q function and optionally policy/alpha with TD3 delayed update."""
        buffer_state, params, opt_states, t, key = carry
        key, buff_key, q_key, actor_key = jax.random.split(key, 4)

        # sample
        data = rb.sample(buffer_state, buff_key).experience

        # learn
        params, opt_states, q_loss_info = update_q(params, opt_states, data, q_key)

        # params, opt_states, act_loss_info = lax.cond(
        #     t % cfg.system.policy_frequency == 0,  # TD 3 Delayed update support
        #     update_actor_and_alpha,
        #     # just return same params and opt_states and 0 for losses
        #     lambda params, opt_states, *_: (
        #         params,
        #         opt_states,
        #         {"actor_loss": 0.0, "alpha_loss": 0.0},
        #     ),
        #     params,
        #     opt_states,
        #     data,
        #     actor_key,
        # )

        return (buffer_state, params, opt_states, t, key), q_loss_info

    def act( # TODO we will need to implement epsilon greedy exploration. We can start with a fixed epsilon (e.g. 0.1) and implement epsilon decay later.
        carry: Tuple[FrozenVariableDict, Array, State, BufferState, chex.PRNGKey], _: Any
    ) -> Tuple[Tuple[FrozenVariableDict, Array, State, BufferState, chex.PRNGKey], Dict]:
        """Acting loop: select action, step env, add to buffer."""
        actor_params, obs, env_state, buffer_state, key = carry

        action = jax.numpy.zeros((128,5)) #  NOTE (Louise) make this better. This needs to be better
        # pi = actor.apply(actor_params, obs)
        # action = pi.sample(seed=key)

        next_obs, env_state, buffer_state, metrics = step(action, obs, env_state, buffer_state)
        return (actor_params, next_obs, env_state, buffer_state, key), metrics

    # def explore(carry: LearnerState, _: Any) -> Tuple[LearnerState, Metrics]:
    #     """Take random actions to fill up buffer at the start of training."""
    #     obs, env_state, buffer_state, _, _, t, key = carry
    #     key, explore_key = jax.random.split(key)
    #     action = jax.random.uniform(explore_key, full_action_shape)
    #     next_obs, env_state, buffer_state, metrics = step(action, obs, env_state, buffer_state)

    #     t += cfg.system.n_envs
    #     learner_state = carry._replace(
    #         obs=next_obs, env_state=env_state, buffer_state=buffer_state, t=t, key=key
    #     )
    #     return learner_state, metrics

    scanned_update = lambda state: lax.scan(update_epoch, state, None, length=cfg.system.epochs)
    scanned_act = lambda state: lax.scan(act, state, None, length=cfg.system.rollout_length)

    # Act loop -> sample -> update loop
    def update_step(carry: LearnerState, _: Any) -> Tuple[LearnerState, Tuple[Metrics, Metrics]]:
        """Act, sample, learn."""

        obs, env_state, buffer_state, params, opt_states, t, key = carry
        key, act_key, learn_key = jax.random.split(key, 3)
        # Act
        act_state = (params.dqns.online, obs, env_state, buffer_state, act_key)
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
    # explore_steps = cfg.system.explore_steps // cfg.system.n_envs
    # pmaped_explore = jax.pmap(
    #     jax.vmap(
    #         lambda state: lax.scan(explore, state, None, length=explore_steps),
    #         axis_name="batch",
    #     ),
    #     axis_name="device",
    #     donate_argnums=0,
    # )
    
    pmaped_steps = 1024  # todo: config option # NOTE (Claude) this is related to steps between logging.


    pmaped_updated_step = jax.pmap(
        jax.vmap(
            lambda state: lax.scan(update_step, state, None, length=pmaped_steps),
            axis_name="batch",
        ),
        axis_name="device",
        donate_argnums=0,
    )

    return pmaped_updated_step


def run_experiment(cfg: DictConfig) -> float:
    n_devices = len(jax.devices())

    pmaped_steps = 1024  # todo: config option
    steps_per_rollout = n_devices * cfg.system.n_envs * cfg.system.rollout_length * pmaped_steps
    max_episode_return = -jnp.inf

    (env, eval_env), nns, opts, rb, learner_state, logger, key = init(cfg)
    update = make_update_fns(cfg, env, nns, opts, rb)

    _,q = nns
    key, eval_key = jax.random.split(key)
    # todo: don't need to return trained_params or eval keys
    evaluator, absolute_metric_evaluator, (trained_params, eval_keys) = evaluator_setup(
        eval_env=eval_env,
        key=eval_key,
        network=q, #NOTE (Louise) this part needs redoing with that replacement actor function
        params=learner_state.params.dqns.online,
        config=cfg,
    )

    if cfg.logger.checkpointing.save_model:
        checkpointer = Checkpointer(
            metadata=cfg,  # Save all config as metadata in the checkpoint
            model_name=cfg.logger.system_name,
            **cfg.logger.checkpointing.save_args,  # Checkpoint args
        )

    start_time = time.time()

    # Fill up buffer/explore.
    # learner_state, metrics = explore(learner_state)

    # # Log explore metrics.
    # final_metrics = episode_metrics.get_final_step_metrics(metrics)
    # t = int(jnp.sum(learner_state.t))
    # sps = t / (time.time() - start_time)
    # logger.log({"step": t, "steps per second": sps}, t, 0, LogEvent.MISC)
    # logger.log(final_metrics, cfg.system.explore_steps, 0, LogEvent.ACT)

    # Main loop:
    # We want start to align with the final step of the first pmaped_learn,
    # where we've done explore_steps and 1 full learn step.
    start = cfg.system.explore_steps + steps_per_rollout
    for eval_idx, t in enumerate(range(start, int(cfg.system.total_timesteps), steps_per_rollout)):
        # Learn loop:
        learner_state, (metrics, losses) = update(learner_state)
        jax.block_until_ready(learner_state)

        # Log:
        # Multiply by learn steps here because anakin steps per second is learn + act steps
        # But we want to make sure we're counting env steps correctly so it's not included
        # in the loop counter.
        sps = t * cfg.system.epochs / (time.time() - start_time)
        final_metrics = episode_metrics.get_final_step_metrics(metrics)
        loss_metrics = losses | {"log_alpha": learner_state.params.log_alpha}
        logger.log({"step": t, "steps_per_second": sps}, t, eval_idx, LogEvent.MISC)
        logger.log(final_metrics, t, eval_idx, LogEvent.ACT)
        logger.log(loss_metrics, t, eval_idx, LogEvent.TRAIN)

        # Evaluate:
        key, eval_key = jax.random.split(key)
        eval_keys = jax.random.split(eval_key, n_devices)
        # todo: bug likely here -> don't have batch vmap yet so shouldn't unreplicate_batch_dim
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
            unreplicated_learner_state = unreplicate_learner_state(learner_state)  # type: ignore
            checkpointer.save(
                timestep=t,
                unreplicated_learner_state=unreplicated_learner_state,
                episode_return=episode_return,
            )

    # Measure absolute metric.
    if cfg.arch.absolute_metric:
        eval_keys = jax.random.split(key, n_devices)

        eval_output = absolute_metric_evaluator(best_params, eval_keys)
        jax.block_until_ready(eval_output)

        logger.log(eval_output.episode_metrics, t, eval_idx, LogEvent.ABSOLUTE)

    logger.stop()

    return float(max_episode_return)


@hydra.main(config_path="../configs", config_name="default_ff_isac.yaml", version_base="1.2")
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
