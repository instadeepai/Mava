import functools
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

import chex
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


@dataclass
class MCTS:
    """TODO: Add description here."""

    def __init__(self, config) -> None:
        """TODO: Add description here."""
        self.config = config

    def get_action(self, forward_fn, params, rng_key, observation, **kwargs):
        """TODO: Add description here."""
        search_out = self.search(forward_fn, params, rng_key, observation, **kwargs)

        return (
            np.squeeze(np.array(search_out.action, np.int64)),
            {"search_policies": np.squeeze(np.array(search_out.action_weights))},
        )

    def search(self, forward_fn, params, rng_key, observation, **kwargs):
        """TODO: Add description here."""

        def perform_search(params, rng_key, observation):
            forward_w_params = functools.partial(forward_fn, params=params)

            def search_recurrent_fn(params, rng_key, action, observation):
                return self.config.recurrent_fn(
                    self.config.environment_model,
                    forward_w_params,
                    rng_key,
                    action,
                    observation,
                    **kwargs,
                )

            root_output = self.config.root_fn(forward_w_params, rng_key, observation)

            search_output = self.config.search(
                params,
                rng_key,
                root_output,
                search_recurrent_fn,
                self.config.num_simulations,
                self.config.max_depth,
            )
            return search_output

        jitted_perform_search = jit(perform_search)

        search_output = jitted_perform_search(params, rng_key, observation)

        return search_output
