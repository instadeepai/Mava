import dataclasses
from typing import Dict, Tuple

import chex
import jax.numpy as jnp
import numpy as np
from acme.jax import networks as networks_lib
from jax import jit

from mava.components.jax.executing.epsilon_greedy import EpsilonGreedyWithMask


@dataclasses.dataclass
class DQNNetworks:
    """A Class for DQN network.

    Args:
        network: pure function defining the feedforward neural network
        params: the parameters of the network.
    """

    def __init__(
        self,
        network: networks_lib.FeedForwardNetwork,
        params: networks_lib.Params,
    ) -> None:
        """Instantiate the class."""
        self.network = network
        self.params = params

        @jit
        def forward_fn(
            params: Dict[str, jnp.ndarray], observations: networks_lib.Observation
        ) -> jnp.ndarray:
            """Forward evaluation of the network.

            Args:
                params: the network parameters
                observations: the observation

            Returns:
                the results of the feedforward evaluation, i.e. the q-values in DQN
            """
            # The parameters of the network might change. So it has to
            # be fed into the jitted function.
            q_values = self.network.apply(params, observations)

            return q_values

        self.forward_fn = forward_fn

    def get_action(
        self,
        observations: networks_lib.Observation,
        key: networks_lib.PRNGKey,
        epsilon: float,
        mask: chex.Array,
    ) -> Tuple[np.ndarray, Dict]:
        """Taking actions using epsilon greedy approach.

        Args:
            observations: the observations
            key: the random number generator key
            epsilon: the epsilon value for the epsilon-greedy approach. If 0.0, then the
             action is taken deterministically.

        Returns:
            the actions and a dictionary with q-values
        """
        action_values = self.forward_fn(self.params, observations)
        actions = EpsilonGreedyWithMask(
            preferences=action_values, epsilon=epsilon, mask=mask  # type: ignore
        ).sample(seed=key)
        assert len(actions) == 1, "Only one action is allowed."
        actions = np.array(actions, dtype=np.int64)
        actions = np.squeeze(actions)

        return actions, {"action_values": np.squeeze(action_values)}

    def get_value(self, observations: networks_lib.Observation) -> jnp.ndarray:
        """Get the value of the network.

        Args:
            observations: the observations
        Returns:
            the feedforward values of the network, i.e. the Q-values in DQN.
        """
        q_value = self.network.apply(self.params, observations)
        return q_value
