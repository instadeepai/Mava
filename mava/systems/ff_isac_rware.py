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
from flashbax.buffers.flat_buffer import TransitionSample as BufferSample
from flashbax.buffers.trajectory_buffer import TrajectoryBuffer
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState as BufferState
from jumanji.wrappers import AutoResetWrapper

from mava.types import ActorApply, CriticApply, LearnerState, Observation
from mava.wrappers.jumanji import RwareMultiAgentWrapper


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
    critic: CriticAndTarget
    log_alpha: Numeric


class OptStates(NamedTuple):
    actor: optax.OptState
    critic: optax.OptState
    alpha: optax.OptState


State = Tuple[LearnerState, BufferState[Transition]]


class Actor(nn.Module):
    """Actor Network."""

    action_dim: int

    @nn.compact
    def __call__(self, observation: Observation) -> distrax.Categorical:
        """Forward pass."""
        x = observation.agents_view

        x = nn.relu(nn.Dense(128)(x))
        x = nn.relu(nn.Dense(128)(x))
        logits = nn.Dense(self.action_dim)(x)

        masked_logits = jnp.where(observation.action_mask, logits, jnp.finfo(jnp.float32).min)
        return distrax.Categorical(logits=masked_logits)


class Critic(nn.Module):
    """Actor Network."""

    action_dim: int

    @nn.compact
    def __call__(self, observation: Observation) -> distrax.Categorical:
        """Forward pass."""
        x = observation.agents_view

        x = nn.relu(nn.Dense(128)(x))
        x = nn.relu(nn.Dense(128)(x))
        qs = nn.Dense(self.action_dim)(x)

        masked_qs = jnp.where(observation.action_mask, qs, -jnp.inf)
        return masked_qs


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
    # todo: just pass in actor and opt
    actor: Actor,
    critic: nn.Module,  # todo
    opt: optax.GradientTransformation,
    buffer: TrajectoryBuffer,
    config: Dict,
) -> Callable[[LearnerState, BufferState[Transition]], State]:
    n_rollouts = config["rollout_length"]
    target_entropy = env.action_spec().num_values[0]
    act_dim = int(env.action_spec().num_values[0])

    def critic_loss(critic_params: CriticParams, obs: Array, target: Array, act: Array) -> Numeric:
        act_one_hot = jax.nn.one_hot(act, act_dim)

        q1 = critic.apply(critic_params.first, obs)
        q1 = (q1 * act_one_hot).sum(axis=-1)
        q2 = critic.apply(critic_params.second, obs)
        q2 = (q2 * act_one_hot).sum(axis=-1)

        return jnp.mean((target - q1) ** 2) + jnp.mean((target - q2) ** 2)

    def policy_loss(
        policy_params: nn.FrozenDict,
        critic_params: CriticParams,
        log_alpha: Numeric,
        obs: Observation,
        key: PRNGKey,
    ) -> Numeric:
        policy = actor.apply(policy_params, obs)
        act, log_prob = policy.sample_and_log_prob(seed=key)

        act_one_hot = jax.nn.one_hot(act, act_dim)

        q1 = critic.apply(critic_params.first, obs)
        q1 = (q1 * act_one_hot).sum(axis=-1)  # todo: method
        q2 = critic.apply(critic_params.second, obs)
        q2 = (q2 * act_one_hot).sum(axis=-1)

        q = jnp.minimum(q1, q2)

        # todo sac discrete does sum and then mean, we should test this
        # https://github.com/BY571/SAC_discrete/blob/main/agent.py#L80C79-L80C79
        return jnp.mean(jnp.exp(log_alpha) * log_prob)

    def alpha_loss(
        log_alpha: Numeric, actor_params: nn.FrozenDict, obs: Array, key: PRNGKey
    ) -> Numeric:
        # todo: do this once! (double work here and in policy_loss)
        policy: distrax.Categorical = actor.apply(actor_params, obs)
        action = policy.sample(seed=key)
        log_prob = policy.log_prob(action)
        return -jnp.exp(log_alpha) * jnp.mean((log_prob + target_entropy))

    def update(
        learner_state: LearnerState[Params, OptStates], batch: BufferSample[Transition]
    ) -> Tuple[LearnerState, Dict[str, Array]]:
        obs = batch.experience.first.obs
        next_obs = batch.experience.second.obs
        act = batch.experience.first.action
        rew = batch.experience.first.reward
        done = batch.experience.first.done

        key, next_act_key, policy_loss_key, alpha_loss_key = jax.random.split(learner_state.key, 4)
        learner_state = learner_state._replace(key=key)

        next_policy = actor.apply(learner_state.params.actor, next_obs)
        next_act, next_act_log_prob = next_policy.sample_and_log_prob(seed=next_act_key)
        next_act_one_hot = jax.nn.one_hot(next_act, act_dim)

        next_q1 = critic.apply(learner_state.params.critic.targets.first, next_obs)
        next_q1 = (next_q1 * next_act_one_hot).sum(axis=-1)
        next_q2 = critic.apply(learner_state.params.critic.targets.second, next_obs)
        next_q2 = (next_q2 * next_act_one_hot).sum(axis=-1)
        next_q = jnp.minimum(next_q1, next_q2)

        # (B, N)
        # rew = jnp.expand_dims(rew, -1)
        # done = jnp.expand_dims(done, -1)
        target = rew + config["gamma"] * (1 - done) * (
            next_q - jnp.exp(learner_state.params.log_alpha) * next_act_log_prob
        )

        c_loss, critic_grads = jax.value_and_grad(critic_loss)(
            learner_state.params.critic.critics, obs, target, act
        )
        a_loss, actor_grads = jax.value_and_grad(policy_loss)(
            learner_state.params.actor,
            learner_state.params.critic.critics,
            learner_state.params.log_alpha,
            obs,
            policy_loss_key,
        )
        alp_loss, alpha_grads = jax.value_and_grad(alpha_loss)(
            learner_state.params.log_alpha, learner_state.params.actor, obs, alpha_loss_key
        )

        # todo: do a single pmean over a tuple of these?
        # is that more performant?
        # todo: add this back in when pmapping
        # actor_grads = jax.lax.pmean(actor_grads, "learner_devices")
        # critic_grads = jax.lax.pmean(critic_grads, "learner_devices")
        # alpha_grads = jax.lax.pmean(alpha_grads, "learner_devices")
        # a_loss, c_loss, alp_loss = jax.lax.pmean((a_loss, c_loss, alp_loss), "learner_devices")

        # todo: join these updates into a single update
        actor_updates, actor_opt_state = opt.update(
            actor_grads, learner_state.opt_states.actor, learner_state.params.actor
        )
        critic_updates, critic_opt_state = opt.update(
            critic_grads, learner_state.opt_states.critic, learner_state.params.critic.critics
        )
        alpha_updates, alpha_opt_state = opt.update(
            alpha_grads, learner_state.opt_states.alpha, learner_state.params.log_alpha
        )

        actor_params = optax.apply_updates(learner_state.params.actor, actor_updates)
        new_critic_params = optax.apply_updates(learner_state.params.critic.critics, critic_updates)
        log_alpha = optax.apply_updates(learner_state.params.log_alpha, alpha_updates)

        new_target_params = optax.incremental_update(
            learner_state.params.critic.critics, learner_state.params.critic.targets, config["tau"]
        )

        critic_params = CriticAndTarget(new_critic_params, new_target_params)

        new_opt_states = OptStates(
            actor=actor_opt_state, critic=critic_opt_state, alpha=alpha_opt_state
        )

        # todo: return these!
        losses = (a_loss, c_loss, alp_loss)
        learner_state = LearnerState(
            params=Params(actor=actor_params, critic=critic_params, log_alpha=log_alpha),
            opt_states=new_opt_states,
            key=key,
            env_state=learner_state.env_state,
            timestep=learner_state.timestep,
        )
        return learner_state, losses

    def act(_: int, carry: State) -> State:
        learner_state, buffer_state = carry
        actor_params = learner_state.params.actor

        # SELECT ACTION
        key, policy_key = jax.random.split(learner_state.key)
        actor_policy = actor.apply(actor_params, learner_state.timestep.observation)
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
            # todo: get metrics/losses from here
            learner_state, metrics = jax.lax.scan(update, learner_state, batches)

            return learner_state, buffer_state

        return jax.lax.fori_loop(
            0,
            int(config["num_updates"] / config["num_evals"]),
            _act_and_log,
            (learner_state, buffer_state),
        )

    return act_and_learn


def main() -> None:
    key = jax.random.PRNGKey(0)

    env = jumanji.make("RobotWarehouse-v0")
    env = RwareMultiAgentWrapper(env)
    env = AutoResetWrapper(env)

    config = {
        "tau": 0.005,
        "gamma": 0.95,
        "num_updates": 20,
        "num_minibatches": 3,
        "rollout_length": 10,
        "num_agents": env.num_agents,
        "num_envs": 2,
        "num_evals": 10,
    }

    actor = Actor(env.action_spec().num_values[0])
    critic = Critic(env.action_spec().num_values[0])  # todo: better critic
    opt = optax.adam(1e-3)
    buffer = fbx.make_flat_buffer(1_000_000, 0, 64, add_batch_size=config["num_envs"])

    dummy_act = env.action_spec().generate_value()
    dummy_obs = env.observation_spec().generate_value()
    dummy_transition = Transition(
        obs=dummy_obs,
        action=dummy_act,
        reward=jnp.zeros(env.num_agents),
        done=jnp.zeros(env.num_agents, dtype=bool),
    )

    params = Params(
        actor=actor.init(key, dummy_obs),
        # todo better names, this is confusing
        # critics -> online
        critic=CriticAndTarget(
            critics=CriticParams(
                # todo: keys!
                first=critic.init(key, dummy_obs),
                second=critic.init(key, dummy_obs),
            ),
            targets=CriticParams(
                first=critic.init(key, dummy_obs),
                second=critic.init(key, dummy_obs),
            ),
        ),
        # todo: separate log alphas
        log_alpha=jnp.asarray(0.0),
    )
    opt_states = OptStates(
        # todo: allow for different optimizers and different learning rates.
        actor=opt.init(params.actor),
        critic=opt.init(params.critic.critics),
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

    learner_fn = get_learner_fn(env, actor, critic, opt, buffer, config)
    learner_fn(learner_state, buffer_state)


if __name__ == "__main__":
    main()
