import copy
import dataclasses
from typing import Dict

from acme.jax import networks as networks_lib
import jax.numpy as jnp
import jax
import distrax


@dataclasses.dataclass
class IDQNNetwork:
    def __init__(self, policy_params, network) -> None:
        self.policy_params: networks_lib.Params = policy_params
        self.target_policy_params: networks_lib.Params = copy.deepcopy(policy_params)
        self.policy_network: networks_lib.FeedForwardNetwork = network

        def forward_fn(policy_params: Dict[str, jnp.ndarray], observations: networks_lib.Observation):
            return self.policy_network.apply(policy_params, observations)

        self.forward = forward_fn

    def get_action(self, params, observations, epsilon, base_key, mask: jnp.array):
        q_values = self.forward(params, observations)
        masked_q_values = jnp.where(mask == 1.0, q_values, -99999)  # todo

        greedy_actions = masked_q_values == jnp.max(masked_q_values)
        greedy_actions_probs = greedy_actions / jnp.sum(greedy_actions)

        random_action_probs = mask / jnp.sum(mask)

        combined_probs = (
            1 - epsilon
        ) * greedy_actions_probs + epsilon * random_action_probs

        action_dist = distrax.Categorical(probs = combined_probs)
        return action_dist.sample(seed = base_key)

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