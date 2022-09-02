# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
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
from dataclasses import dataclass
from typing import Callable, Optional

from mava.components.jax.training.losses import Loss
from mava.core_jax import SystemTrainer


@dataclass
class MAPGWithTrustRegionClippingLossGdnPolicyConfig:
    clipping_epsilon: float = 0.2
    value_clip_parameter: float = 0.2
    clip_value: bool = True
    entropy_cost: float = 0.01
    value_cost: float = 0.5


class MAPGWithTrustRegionClippingLossGdnPolicy(Loss):
    def __init__(
        self,
        config: MAPGWithTrustRegionClippingLossGdnPolicyConfig = MAPGWithTrustRegionClippingLossGdnPolicyConfig(),  # noqa: E501
    ):
        """Component defines a MAPGWithTrustRegionClipping loss function for a GDN.

        It uses the policy loss to compute gradients for the GDN.
        Another alternative would be to use the critic loss.

        Args:
            config : MAPGTrustRegionClippingLossSeparateNetworksConfig
        """
        self.config = config

    def on_training_loss_fns(self, trainer: SystemTrainer) -> None:
        """Create and store MAPGWithTrustRegionClippingLossGdnPolicy loss function.

        Args:
            trainer: SystemTrainer.

        Returns:
            None.
        """

        # TODO(Matthew): implement this properly
        def gdn_loss_grad_fn(
            policy_params: Any,
            observations: Any,
            actions: Dict[str, jnp.ndarray],
            behaviour_log_probs: Dict[str, jnp.ndarray],
            advantages: Dict[str, jnp.ndarray],
        ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Dict[str, jnp.ndarray]]]:
            """Surrogate loss using clipped probability ratios.

            Args:
                policy_params: policy network parameters.
                observations: agent observations.
                actions: actions the agents took.
                behaviour_log_probs: Log probabilities of actions taken by
                    current policy in the environment.
                advantages: advantage estimation values per agent.

            Returns:
                Tuple[policy gradients, policy loss information]
            """

            policy_grads = {}
            loss_info_policy = {}
            for agent_key in trainer.store.trainer_agents:
                agent_net_key = trainer.store.trainer_agent_net_keys[agent_key]
                network = trainer.store.networks["networks"][agent_net_key]

                # Note (dries): This is placed here to set the networks correctly in
                # the case of non-shared weights.
                def policy_loss_fn(
                    policy_params: Any,
                    observations: Any,
                    actions: jnp.ndarray,
                    behaviour_log_probs: jnp.ndarray,
                    advantages: jnp.ndarray,
                ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
                    """Inner policy loss function: see outer function for parameters."""
                    # TODO(Matthew): GNN application could go here?
                    distribution_params = network.policy_network.apply(
                        policy_params, observations
                    )
                    log_probs = network.log_prob(distribution_params, actions)
                    entropy = network.entropy(distribution_params)
                    # Compute importance sampling weights:
                    # current policy / behavior policy.
                    rhos = jnp.exp(log_probs - behaviour_log_probs)
                    clipping_epsilon = self.config.clipping_epsilon

                    policy_loss = rlax.clipped_surrogate_pg_loss(
                        rhos, advantages, clipping_epsilon
                    )

                    # Entropy regulariser.
                    entropy_loss = -jnp.mean(entropy)

                    total_policy_loss = (
                        policy_loss + entropy_loss * self.config.entropy_cost
                    )

                    # TODO: (Ruan) Keeping the entropy penalty for now.
                    # can remove or add a flag for including it.
                    loss_info_policy = {
                        "policy_loss_total": total_policy_loss,
                        "loss_policy": policy_loss,
                        "loss_entropy": entropy_loss,
                    }

                    return total_policy_loss, loss_info_policy

                policy_grads[agent_key], loss_info_policy[agent_key] = jax.grad(
                    policy_loss_fn, has_aux=True
                )(
                    policy_params[agent_net_key],
                    observations[agent_key].observation,
                    actions[agent_key],
                    behaviour_log_probs[agent_key],
                    advantages[agent_key],
                )
            return policy_grads, loss_info_policy

        # Save the gradient function
        trainer.store.gdn_grad_fn = gdn_loss_grad_fn

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return MAPGWithTrustRegionClippingLossGdnPolicyConfig
