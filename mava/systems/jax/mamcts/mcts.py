from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

import chex
import mctx
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
class MCTSConfig:
    root_fn: RootFn
    recurrent_fn: RecurrentFn
    search: TreeSearch
    num_simulations: int = 10
    max_depth: MaxDepth = None


@dataclass
class MCTS:
    """TODO: Add description here."""

    def __init__(self, config: MCTSConfig) -> None:
        """TODO: Add description here."""
        self.config = config

    def search(self, params, rng_key, observation):
        @jit
        def perform_search():
            root_output = self.config.root_fn(params, rng_key, observation)

            search_output = self.config.search(
                params,
                rng_key,
                root_output,
                self.config.recurrent_fn,
                self.config.num_simulations,
                self.config.max_depth,
            )
            return search_output

        return perform_search()
