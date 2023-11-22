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

from typing import Callable, Dict, NamedTuple, Sequence, Tuple

import distrax
import flashbax as fbx
import flax.linen as nn
import jax
import jax.numpy as jnp
import jumanji
import optax
from chex import Array, Numeric, PRNGKey
from flashbax.buffers.trajectory_buffer import TrajectoryBuffer
from flashbax.buffers.trajectory_buffer import TrajectoryBufferSample as BufferSample
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState as BufferState
from jumanji.wrappers import AutoResetWrapper

from mava.types import ActorApply, CriticApply, LearnerState, Observation
from mava.wrappers.jumanji import RwareMultiAgentWrapper


def sample_action(mean: Array, log_std: Array, key: PRNGKey) -> Tuple[Array, Array]:
    std = jnp.exp(log_std)
    normal = distrax.Normal(mean, std)
    x_t = normal.sample(seed=key)
    y_t = jnp.tanh(x_t)
    # action = 2 * y_t  # actions [-2,2]
    action = (y_t * 0.5) + 0.5  # enforce actions between [0, 1]
    log_prob = normal.log_prob(x_t)
    log_prob -= jnp.log(0.5 * (1 - y_t**2) + 1e-6)
    log_prob = jnp.sum(log_prob, axis=-1, keepdims=True)

    return action, log_prob


def select_actions_sac(
    apply_fn: Callable, params: nn.FrozenDict, obs: Array, key: PRNGKey
) -> Array:
    mean, log_std = apply_fn(params, obs)
    actions, _ = sample_action(mean, log_std, key)
    return actions


class Transition(NamedTuple):
    obs: Array
    action: Numeric
    reward: Array
    done: bool


class CriticParams(NamedTuple):
    """Parameters for a critic network since SAC uses two critics."""

    first: nn.FrozenDict
    second: nn.FrozenDict


class CriticAndTarget(NamedTuple):
    critics: CriticParams
    targets: CriticParams


class Params(NamedTuple):
    actor: nn.FrozenDict
    critic: CriticParams
    log_alpha: Numeric


class OptStates(NamedTuple):
    actor: optax.OptState
    critic: optax.OptState
    alpha: optax.OptState


State = Tuple[LearnerState, BufferState[Transition]]


class Actor(nn.Module):
    """Actor Network."""

    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, observation: Observation) -> distrax.Categorical:
        """Forward pass."""
        x = observation.agents_view

        actor_output = nn.Dense(128)(x)
        actor_output = nn.relu(actor_output)
        actor_output = nn.Dense(128)(actor_output)
        actor_output = nn.relu(actor_output)
        actor_output = nn.Dense(self.action_dim)(actor_output)

        masked_logits = jnp.where(observation.action_mask, actor_output, jnp.finfo(jnp.float32).min)
        actor_policy = distrax.Categorical(logits=masked_logits)

        return actor_policy


#
#
# @partial(jax.pmap, axis_name="learner_devices", devices=learner_devices)
# def update(
#     sample: fbx.flat_buffer.TransitionSample[Transition],
#     actor_params: nn.FrozenDict,
#     critic_params: CriticAndTarget,
#     log_alpha: Numeric,
#     opt_states: Tuple[optax.OptState, optax.OptState, optax.OptState],
#     cfg: Dict,  # nested Dict[str: array]
#     key: PRNGKey,
# ):
#     # Reshape experience for the minibatch size
#     # (B, N, ...) -> (B // minibatch_size, minibatch_size, N, ...)
#     exp = jax.tree_map(
#         lambda x: jnp.reshape(
#             x,
#             (n_minibatches, cfg.system.minibatch_size, *x.shape[1:]),
#         ),
#         sample.experience,
#     )
#
#     def minibatch(i, carry):
#         exp, actor_params, critic_params, log_alpha, opt_states, _ = carry
#         actor_opt_state, critic_opt_state, alpha_opt_state = opt_states
#
#         obs = exp.first.obs[i]
#         next_obs = exp.second.obs[i]
#         act = exp.first.action[i]
#         rew = exp.first.reward[i]
#         done = exp.first.done
#
#         next_act_key, policy_loss_key, alpha_loss_key = jax.random.split(key, 3)
#
#         mean, log_std = vmapped_actor(actor_params, next_obs)
#         next_act, next_act_log_prob = sample_action(mean, log_std, next_act_key)
#
#         critic_input = jnp.concatenate([next_obs, next_act], axis=-1)
#
#         next_q1 = vmapped_critic(critic_params.targets.first, critic_input)
#         next_q2 = vmapped_critic(critic_params.targets.second, critic_input)
#         next_q = jnp.minimum(next_q1, next_q2).squeeze(axis=-1)
#
#         # (B, N)
#         # rew = jnp.expand_dims(rew, -1)
#         # done = jnp.expand_dims(done, -1)
#         target = rew + cfg.system.gamma * done * (
#             next_q - jnp.exp(log_alpha) * next_act_log_prob.squeeze(axis=-1)
#         )
#
#         c_loss, critic_grads = jax.value_and_grad(critic_loss)(
#             critic_params.critics, jnp.concatenate([obs, act], axis=-1), target
#         )
#         a_loss, actor_grads = jax.value_and_grad(policy_loss)(
#             actor_params, critic_params.critics, log_alpha, obs, policy_loss_key
#         )
#         alp_loss, alpha_grads = jax.value_and_grad(alpha_loss)(
#             log_alpha, actor_params, obs, alpha_loss_key
#         )
#
#         # todo: do a single pmean over a tuple of these?
#         # is that more performant?
#         actor_grads = jax.lax.pmean(actor_grads, "learner_devices")
#         critic_grads = jax.lax.pmean(critic_grads, "learner_devices")
#         alpha_grads = jax.lax.pmean(alpha_grads, "learner_devices")
#         a_loss, c_loss, alp_loss = jax.lax.pmean((a_loss, c_loss, alp_loss), "learner_devices")
#
#         # todo: join these updates into a single update
#         actor_updates, actor_opt_state = optim.update(actor_grads, actor_opt_state, actor_params)
#         critic_updates, critic_opt_state = optim.update(
#             critic_grads, critic_opt_state, critic_params.critics
#         )
#         alpha_updates, alpha_opt_state = optim.update(alpha_grads, alpha_opt_state, log_alpha)
#
#         actor_params = optax.apply_updates(actor_params, actor_updates)
#         new_critic_params = optax.apply_updates(critic_params.critics, critic_updates)
#         log_alpha = optax.apply_updates(log_alpha, alpha_updates)
#
#         new_target_params = optax.incremental_update(
#             critic_params.critics, critic_params.targets, cfg.system.tau
#         )
#
#         critic_params = CriticAndTarget(new_critic_params, new_target_params)
#
#         new_opt_states = (actor_opt_state, critic_opt_state, alpha_opt_state)
#         losses = (a_loss, c_loss, alp_loss)
#         return exp, actor_params, critic_params, log_alpha, new_opt_states, losses
#
#     init_val = (exp, actor_params, critic_params, log_alpha, opt_states, (0, 0, 0))
#     (
#         _,
#         actor_params,
#         critic_params,
#         log_alpha,
#         opt_states,
#         (a_loss, c_loss, alp_loss),
#     ) = jax.lax.fori_loop(0, n_minibatches, minibatch, init_val)
#
#     return (
#         actor_params,
#         critic_params,
#         log_alpha,
#         opt_states,
#         a_loss,
#         c_loss,
#         alp_loss,
#     )


def get_learner_fn(
    env: jumanji.Environment,
    apply_fns: Tuple[ActorApply, CriticApply],
    update_fn: optax.TransformUpdateFn,  # todo: update fn per param?
    buffer: TrajectoryBuffer,
    config: Dict,
) -> Callable[[LearnerState, BufferState[Transition]], State]:
    # Get apply and update functions for actor and critic networks.
    actor_apply_fn, critic_apply_fn = apply_fns

    n_rollouts = config["rollout_length"]
    target_entropy = env.action_spec().num_values[0]

    def critic_loss(critic_params: CriticParams, critic_input: Array, target: Array) -> Numeric:
        q1 = critic_apply_fn(critic_params.first, critic_input).squeeze(axis=-1)
        q2 = critic_apply_fn(critic_params.second, critic_input).squeeze(axis=-1)
        return jnp.mean((target - q1) ** 2) + jnp.mean((target - q2) ** 2)

    def policy_loss(
        policy_params: nn.FrozenDict,
        critic_params: CriticParams,
        log_alpha: Numeric,
        obs: Array,
        key: PRNGKey,
    ) -> Numeric:
        mean, log_std = actor_apply_fn(policy_params, obs)
        act, log_prob = sample_action(mean, log_std, key)

        critic_input = jnp.concatenate([obs, act], axis=-1)

        q1 = critic_apply_fn(critic_params.first, critic_input).squeeze(axis=-1)
        q2 = critic_apply_fn(critic_params.second, critic_input).squeeze(axis=-1)

        q = jnp.minimum(q1, q2)

        return jnp.mean(jnp.exp(log_alpha) * log_prob.squeeze(axis=-1) - q)

    def alpha_loss(
        log_alpha: Numeric, actor_params: nn.FrozenDict, obs: Array, key: PRNGKey
    ) -> Numeric:
        # todo: do this once! (double work here and in policy_loss)
        mean, log_std = actor_apply_fn(actor_params, obs)
        _, log_prob = sample_action(mean, log_std, key)
        return -jnp.exp(log_alpha) * jnp.mean((log_prob + target_entropy))

    def update(
        learner_state: LearnerState, batch: BufferSample[Transition]
    ) -> Tuple[LearnerState, Dict[str, Array]]:
        return learner_state, {}

    def act(_: int, carry: State) -> State:
        learner_state, buffer_state = carry
        actor_params = learner_state.params.actor

        # SELECT ACTION
        key, policy_key = jax.random.split(learner_state.key)
        actor_policy = actor_apply_fn(actor_params, learner_state.timestep.observation)
        action = actor_policy.sample(seed=policy_key)

        # STEP ENVIRONMENT
        env_state, timestep = jax.vmap(env.step)(learner_state.env_state, action)

        # LOG EPISODE METRICS
        done, reward = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x, config["num_agents"]).reshape(config["num_envs"], -1),
            (timestep.last(), timestep.reward),
        )

        # learner_state.timestep is the obs and timestep is next obs
        obs = learner_state.timestep.observation  # todo - save whole obs?
        transition = Transition(obs=obs, action=action, reward=reward, done=done)
        learner_state = learner_state._replace(env_state=env_state, timestep=timestep, key=key)
        # todo: check if the donate_argnums is preserved here
        buffer_state = buffer.add(buffer_state, transition)

        return learner_state, buffer_state

    def act_and_learn(learner_state: LearnerState, buffer_state: BufferState[Transition]) -> State:
        def _act_and_log(_: int, carry: State) -> State:
            learner_state, buffer_state = carry

            key, sample_key = jax.random.split(learner_state.key)
            learner_state = learner_state._replace(key=key)

            learner_state, buffer_state = jax.lax.fori_loop(0, n_rollouts, act, carry)
            batches = buffer.sample(buffer_state, sample_key)
            # todo treemap -> reshape(num_minibatches, ...)
            learner_state, metrics = jax.lax.scan(update, learner_state, batches)

            return learner_state, buffer_state

        return jax.lax.fori_loop(
            0, config["num_updates"], _act_and_log, (learner_state, buffer_state)
        )

    return act_and_learn


def main() -> None:
    key = jax.random.PRNGKey(0)

    env = jumanji.make("RobotWarehouse-v0")
    env = RwareMultiAgentWrapper(env)
    env = AutoResetWrapper(env)

    config = {
        "num_updates": 2,
        "num_minibatches": 3,
        "rollout_length": 10,
        "num_agents": env.num_agents,
        "num_envs": 2,
    }

    actor = Actor(env.action_spec().num_values[0])
    critic = nn.Dense(1)
    opt = optax.adam(1e-3)
    buffer = fbx.make_flat_buffer(1_000_000, 0, 64, add_batch_size=config["num_envs"])

    dummy_act = env.action_spec().generate_value()
    dummy_obs = env.observation_spec().generate_value()
    dummy_obs_array = dummy_obs.agents_view[0]
    dummy_transition = Transition(
        obs=dummy_obs,
        action=dummy_act,
        reward=jnp.zeros(env.num_agents),
        done=jnp.zeros(env.num_agents, dtype=bool),
    )

    params = Params(
        actor=actor.init(key, dummy_obs),
        critic=CriticParams(
            first=critic.init(key, jnp.concatenate([dummy_obs_array, dummy_act], axis=-1)),
            second=critic.init(key, jnp.concatenate([dummy_obs_array, dummy_act], axis=-1)),
        ),
        log_alpha=jnp.zeros(1),
    )
    opt_states = OptStates(
        # todo: allow for different optimizers and different learning rates.
        actor=opt.init(params.actor),
        critic=opt.init(params.critic.first),
        alpha=opt.init(params.log_alpha),
    )
    buffer_state = buffer.init(dummy_transition)

    reset_keys = jax.random.split(key)  # todo: num_envs
    state, timestep = jax.vmap(env.reset)(reset_keys)
    learner_state = LearnerState(
        params=params,
        opt_states=opt_states,
        key=key,
        env_state=state,
        timestep=timestep,
    )

    learner_fn = get_learner_fn(env, (actor.apply, critic.apply), opt.update, buffer, config)
    learner_fn(learner_state, buffer_state)


if __name__ == "__main__":
    main()
