import copy
import dataclasses
from typing import Dict

import jax
import jax.numpy as jnp
from acme.jax import networks as networks_lib
from distrax import Categorical  # type: ignore


@dataclasses.dataclass
class IDQNNetwork:
    def __init__(
        self,
        policy_params: networks_lib.Params,
        network: networks_lib.FeedForwardNetwork,
    ) -> None:
        """A container for IDQN networks.

        Holds target and main network

        Args:
            policy_params: parameters of the policy network
            network: structure of the policy network

        Return:
            IDQNNetwork
        """
        self.policy_params: networks_lib.Params = policy_params
        self.target_policy_params: networks_lib.Params = copy.deepcopy(policy_params)
        self.policy_network: networks_lib.FeedForwardNetwork = network

        def forward_fn(
            policy_params: Dict[str, jnp.ndarray],
            observations: networks_lib.Observation,
        ) -> jnp.ndarray:
            """Get Q values from the network given observations

            Args:
                policy_params: parameters of the policy network
                observations: agent observations

            Returns: Q-values of all actions in the current state
            """
            return self.policy_network.apply(policy_params, observations)

        self.forward = forward_fn

    def get_action(
        self,
        params: networks_lib.Params,
        observations: networks_lib.Observation,
        epsilon: float,
        base_key: jax.random.KeyArray,
        mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Get actions from policy network given observations.

        Args:
            policy_params: parameters of the policy network
            observations: agent observations
            epsilon: probability that the agent takes a random action
            base_key: jax random key
            mask: action mask of the legal actions

        Returns:
            an action to take in the current state
        """
        q_values = self.forward(params, observations)
        masked_q_values = jnp.where(mask == 1.0, q_values, jnp.finfo(jnp.float32).min)

        greedy_actions = masked_q_values == jnp.max(masked_q_values)
        greedy_actions_probs = greedy_actions / jnp.sum(greedy_actions)

        random_action_probs = mask / jnp.sum(mask)

        weighted_greedy_probs = (1 - epsilon) * greedy_actions_probs
        weighted_rand_probs = epsilon * random_action_probs
        combined_probs = weighted_greedy_probs + weighted_rand_probs

        action_dist = Categorical(probs=combined_probs)
        return action_dist.sample(seed=base_key)

    def get_params(
        self,
    ) -> Dict[str, networks_lib.Params]:
        """Return current params of the target and policy network.

        Returns:
            policy and target policy params.
        """
        return {
            "policy_network": self.policy_params,
            "target_policy_network": self.target_policy_params,
        }
