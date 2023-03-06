import copy
import dataclasses
from typing import Dict

import chex
import distrax
import jax.numpy as jnp
from acme.jax import networks as networks_lib


@dataclasses.dataclass
class IDRQNNetwork:
    def __init__(
        self,
        policy_params: networks_lib.Params,
        policy_init_state: chex.Array,
        network: networks_lib.FeedForwardNetwork,
    ) -> None:
        """A container for IDRQN networks.

        Holds target and main network

        Args:
            policy_params: parameters of the policy network
            policy_init_state: initial hidden state of the network
            network: structure of the policy network
        """
        self.policy_params: networks_lib.Params = policy_params
        self.target_policy_params: networks_lib.Params = copy.deepcopy(policy_params)
        self.policy_init_state = policy_init_state
        self.policy_network: networks_lib.FeedForwardNetwork = network

        def forward_fn(
            policy_params: networks_lib.Params,
            observations: networks_lib.Observation,
            policy_state: chex.Array,
        ) -> chex.Array:
            """Get Q values from the network given observations

            Args:
                policy_params: parameters of the policy network
                observations: agent observations
                policy_state: hidden states of all agents

            Returns: Q-values of all actions in the current state
            """
            return self.policy_network.apply(
                policy_params, [observations, policy_state]
            )

        self.forward = forward_fn

    def get_action(
        self,
        params: networks_lib.Params,
        policy_state: chex.Array,
        observations: networks_lib.Observation,
        epsilon: float,
        base_key: chex.PRNGKey,
        mask: chex.Array,
    ) -> chex.Array:
        """Get actions from policy network given observations.

        Args:
            params: parameters of the policy network
            policy_state: hidden state of the network
            observations: agent observations
            epsilon: probability that the agent takes a random action
            base_key: jax random key
            mask: action mask of the legal actions

        Returns: an action to take in the current state
        """
        q_values, new_policy_state = self.forward(params, observations, policy_state)
        masked_q_values = jnp.where(mask == 1.0, q_values, -99999)  # todo

        greedy_actions = masked_q_values == jnp.max(masked_q_values)
        greedy_actions_probs = greedy_actions / jnp.sum(greedy_actions)

        random_action_probs = mask / jnp.sum(mask)

        combined_probs = (
            1 - epsilon
        ) * greedy_actions_probs + epsilon * random_action_probs

        action_dist = distrax.Categorical(probs=combined_probs)
        return action_dist.sample(seed=base_key), new_policy_state

    def get_params(
        self,
    ) -> Dict[str, jnp.ndarray]:
        """Return current params.

        Returns:
            policy and target policy params.
        """
        return {
            "policy_network": self.policy_params,
            "target_policy_network": self.target_policy_params,
        }

    def get_init_state(self) -> chex.Array:
        """Get the initial hidden state of the network"""
        return self.policy_init_state
