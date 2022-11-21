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

"""Trainer components for calculating losses."""
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import rlax

from mava.components.training.losses import Loss
from mava.core_jax import SystemTrainer


@dataclass
class IDQNLossConfig:
    gamma: float = 0.99


class IDQNLoss(Loss):
    def __init__(
        self,
        config: IDQNLossConfig = IDQNLossConfig(),
    ):
        """Component defines a MAPGWithTrustRegionClipping loss function.

        Args:
            config: MAPGTrustRegionClippingLossConfig.
        """
        self.config = config

    def on_training_loss_fns(self, trainer: SystemTrainer) -> None:
        """Create and store IDQN loss function.

        Args:
            trainer: SystemTrainer.

        Returns:
            None.
        """

        def policy_loss_grad_fn(
            policy_params: Any,
            target_policy_params: Any,
            observations: Any,
            actions: Dict[str, jnp.ndarray],
            rewards: Dict[str, jnp.ndarray],
            next_observations: Any,
            discounts: Dict[str, jnp.ndarray],
        ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Dict[str, jnp.ndarray]]]:
            """Surrogate loss using clipped probability ratios.

            Args:
                policy_params: policy network parameters.
                target_policy_params: target network parameters
                observations: agent observations at timestep t.
                actions: actions the agents took.
                rewards: rewards given to the agent
                next_observations: agent observations at timestep t+1
                discounts: terminal agent mask (dm_env discounts)

            Returns:
                Tuple[policy gradients, policy loss information]
            """

            policy_grads = {}
            loss_info_policy = {}
            for agent_key in trainer.store.trainer_agents:
                agent_net_key = trainer.store.trainer_agent_net_keys[agent_key]
                network = trainer.store.networks[agent_net_key]

                def policy_loss_fn(
                    policy_params: Any,
                    target_policy_params: Any,
                    observations: Any,
                    actions: jnp.ndarray,
                    rewards: jnp.ndarray,
                    next_observations: Any,
                    discounts: jnp.ndarray,
                    masks: jnp.ndarray,
                ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
                    """Inner policy loss function: see outer function for parameters."""

                    # Feedforward actor.
                    q_tm1 = network.forward(policy_params, observations)
                    q_t_value = network.forward(target_policy_params, next_observations)
                    q_t_selector = network.forward(policy_params, next_observations)

                    q_t_selector = jnp.where(masks == 1.0, q_t_selector, -99999)  # TODO

                    batch_double_q_learning_loss_fn = jax.vmap(
                        rlax.double_q_learning, (0, 0, 0, 0, 0, 0, None)
                    )

                    error = batch_double_q_learning_loss_fn(
                        q_tm1,
                        actions,
                        rewards,
                        discounts * self.config.gamma,
                        q_t_value,
                        q_t_selector,
                        True,
                    )

                    loss = jax.numpy.mean(rlax.l2_loss(error))

                    loss_info_policy = {
                        "policy_loss_total": loss.item(),
                    }

                    return loss, loss_info_policy

                policy_grads[agent_key], loss_info_policy[agent_key] = jax.grad(
                    policy_loss_fn, has_aux=True
                )(
                    policy_params[agent_net_key],
                    target_policy_params[agent_net_key],
                    observations[agent_key].observation,
                    actions[agent_key],
                    rewards[agent_key],
                    next_observations[agent_key].observation,
                    discounts[agent_key],
                    next_observations[agent_key].legal_actions,
                )
            return policy_grads, loss_info_policy

        # Save the gradient funcitons.
        trainer.store.policy_grad_fn = policy_loss_grad_fn
