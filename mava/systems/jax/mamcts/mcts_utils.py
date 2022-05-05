import jax
import jax.numpy as jnp
import mctx
from acme.jax import utils

from mava.utils.tree_utils import add_batch_dim_tree, remove_batch_dim_tree, stack_trees
from mava.wrappers.env_wrappers import EnvironmentModelWrapper


def generic_root_fn(forward_fn, params, key, env_state, observation):

    prior_logits, values = forward_fn(observations=observation, params=params)

    return mctx.RootFnOutput(
        prior_logits=prior_logits.logits,
        value=values,
        embedding=add_batch_dim_tree(env_state),
    )


def default_action_recurrent_fn(
    environment_model: EnvironmentModelWrapper,
    forward_fn,
    params,
    rng_key,
    action,
    env_state,
    agent_info,
    default_action,
) -> mctx.RecurrentFnOutput:

    agent_list = environment_model.get_possible_agents()

    actions = {agent_id: default_action for agent_id in agent_list}

    actions[agent_info] = jnp.squeeze(action)

    env_state = remove_batch_dim_tree(env_state)

    next_state, timestep, _ = environment_model.step(env_state, actions)

    observation = environment_model.get_observation(next_state, agent_info)

    prior_logits, values = forward_fn(
        observations=utils.add_batch_dim(observation), params=params
    )

    reward = timestep.reward[agent_info].reshape(1,)

    discount = timestep.discount[agent_info].reshape(1,)

    return (
        mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=prior_logits.logits,
            value=values,
        ),
        add_batch_dim_tree(next_state),
    )


def random_action_recurrent_fn(
    environment_model: EnvironmentModelWrapper,
    forward_fn,
    params,
    rng_key,
    action,
    env_state,
    agent_info,
) -> mctx.RecurrentFnOutput:

    agent_list = environment_model.get_possible_agents()

    rng_key, *agent_action_keys = jax.random.split(rng_key, len(agent_list))

    actions = {
        agent_id: jax.random.randint(
            agent_rng_key,
            (),
            minval=0,
            maxval=environment_model.action_spec().num_values,
        )
        for agent_rng_key, agent_id in zip(agent_action_keys, agent_list)
    }

    actions[agent_info] = jnp.squeeze(action)

    env_state = remove_batch_dim_tree(env_state)

    next_state, timestep, _ = environment_model.step(env_state, actions)

    observation = environment_model.get_observation(next_state, agent_info)

    prior_logits, values = forward_fn(
        observations=utils.add_batch_dim(observation), params=params
    )

    reward = timestep.reward[agent_info].reshape(1,)

    discount = timestep.discount[agent_info].reshape(1,)

    return (
        mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=prior_logits.logits,
            value=values,
        ),
        add_batch_dim_tree(next_state),
    )


def greedy_policy_recurrent_fn(
    environment_model: EnvironmentModelWrapper,
    forward_fn,
    params,
    rng_key,
    action,
    env_state,
    agent_info,
) -> mctx.RecurrentFnOutput:

    agent_list = environment_model.get_possible_agents()

    stacked_agents = stack_trees(agent_list)

    env_state = remove_batch_dim_tree(env_state)

    prev_observations = jax.vmap(environment_model.get_observation, in_axes=(None, 0))(
        env_state, stacked_agents
    )

    prev_prior_logits, _ = forward_fn(observations=prev_observations, params=params)

    agent_actions = jnp.argmax(prev_prior_logits.logits, -1)

    actions = {agent_id: agent_actions[agent_id.id] for agent_id in agent_list}

    actions[agent_info] = jnp.squeeze(action)

    next_state, timestep, _ = environment_model.step(env_state, actions)

    observation = environment_model.get_observation(next_state, agent_info)

    prior_logits, values = forward_fn(
        observations=utils.add_batch_dim(observation), params=params
    )

    reward = timestep.reward[agent_info].reshape(1,)

    discount = timestep.discount[agent_info].reshape(1,)

    return (
        mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=prior_logits.logits,
            value=values,
        ),
        add_batch_dim_tree(next_state),
    )

