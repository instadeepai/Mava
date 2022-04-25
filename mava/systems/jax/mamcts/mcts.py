import functools
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

import chex
import jax.numpy as jnp
import mctx
import numpy as np
from haiku import Params
from jax import jit

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


class MCTS:
    """TODO: Add description here."""

    def __init__(self, config) -> None:
        """TODO: Add description here."""
        self.config = config

    def get_action(self, forward_fn, params, rng_key, observation, action_mask):
        """TODO: Add description here."""

        search_out = self.search(forward_fn, params, rng_key, observation, action_mask)

        return (
            jnp.squeeze(search_out.action),
            {"search_policies": jnp.squeeze(search_out.action_weights)},
        )

    @functools.partial(jit, static_argnums=(0, 1))
    @chex.assert_max_traces(n=1)
    def search(self, forward_fn, params, rng_key, observation, action_mask):
        """TODO: Add description here."""

        root = self.config.root_fn(forward_fn, params, rng_key, observation)

        def recurrent_fn(params, rng_key, action, embedding):

            return self.config.recurrent_fn(
                self.config.environment_model,
                forward_fn,
                params,
                rng_key,
                action,
                embedding,
            )

        print(action_mask)
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
