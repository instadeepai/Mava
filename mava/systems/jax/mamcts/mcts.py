import functools
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

import chex
import jax.numpy as jnp
import mctx
import numpy as np
from haiku import Params
from jax import jit

from mava.utils.id_utils import EntityId
from mava.utils.tree_utils import add_batch_dim_tree, remove_batch_dim_tree
from mava.wrappers.env_wrappers import EnvironmentModelWrapper

RecurrentState = Any
RootFn = Callable[[Params, chex.PRNGKey, Any], mctx.RootFnOutput]
RecurrentState = Any
RecurrentFn = Callable[
    [Params, chex.PRNGKey, chex.Array, RecurrentState],
    Tuple[mctx.RecurrentFnOutput, RecurrentState],
]
MaxDepth = Optional[int]
SearchOutput = mctx.PolicyOutput[Union[mctx.GumbelMuZeroExtraData, None]]
TreeSearch = Callable[
    [Params, chex.PRNGKey, mctx.RootFnOutput, RecurrentFn, int, MaxDepth], SearchOutput
]

def generic_root_fn(forward_fn, params, key, env_state, observation):

        prior_logits, values = forward_fn(observations=observation, params=params)

        return mctx.RootFnOutput(
            prior_logits=prior_logits.logits,
            value=values,
            embedding=add_batch_dim_tree(env_state),
        )

def generic_recurrent_fn(
        environment_model : EnvironmentModelWrapper,
        forward_fn,
        params,
        rng_key,
        action,
        env_state,
        agent_info,
    ) -> mctx.RecurrentFnOutput:

        actions = {agent_id : 0 for agent_id in environment_model.get_possible_agents()}

        actions[agent_info] = jnp.squeeze(action)

        env_state = remove_batch_dim_tree(env_state)

        next_state, timestep, _ = environment_model.step(env_state, actions)

        observation = environment_model.get_observation(next_state, agent_info)

        prior_logits, values = forward_fn(
            observations=observation, params=params
        )

        reward = timestep.reward[agent_info].reshape(
            1,
        )

        discount = timestep.discount[agent_info].reshape(
            1,
        )

        return (
            mctx.RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=prior_logits.logits,
                value=values,
            ),
            add_batch_dim_tree(next_state),
        )

class MCTS:
    """TODO: Add description here."""

    def __init__(self, config) -> None:
        """TODO: Add description here."""
        self.config = config

    def get_action(
        self,
        forward_fn,
        params,
        rng_key,
        env_state,
        observation,
        action_mask,
        agent_info,
    ):
        """TODO: Add description here."""
        # agent_info = EntityId.from_string(agent_info)
        search_out = self.search(
            forward_fn,
            params,
            rng_key,
            env_state,
            observation,
            action_mask,
            str(agent_info),
        )

        return (
            jnp.squeeze(search_out.action.astype(jnp.int64)),
            {"search_policies": jnp.squeeze(search_out.action_weights)},
        )

    @functools.partial(jit, static_argnums=(0, 1, 7))
    def search(
        self,
        forward_fn,
        params,
        rng_key,
        env_state,
        observation,
        action_mask,
        agent_info,
    ):
        """TODO: Add description here."""

        root = self.config.root_fn(forward_fn, params, rng_key, env_state, observation)

        def recurrent_fn(params, rng_key, action, embedding):

            return self.config.recurrent_fn(
                self.config.environment_model,
                forward_fn,
                params,
                rng_key,
                action,
                embedding,
                agent_info,
            )

        search_output = self.config.search(
            params=params,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=self.config.num_simulations,
            invalid_actions=1 - action_mask.reshape(1, -1),
            max_depth=self.config.max_depth,
        )

        return search_output
