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

import jax.numpy as jnp
import rlax

from mava.components.jax.training import Loss
from mava.core_jax import SystemTrainer


@dataclass
class MAPGTrustRegionClippingLossConfig:
    clipping_epsilon: float = 0.2
    clip_value: bool = True
    entropy_cost: float = 0.0
    value_cost: float = 1.0


class MAPGWithTrustRegionClippingLoss(Loss):
    def __init__(
        self,
        config: MAPGTrustRegionClippingLossConfig = MAPGTrustRegionClippingLossConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_loss_fns(self, trainer: SystemTrainer) -> None:
        """_summary_"""

        def loss(
            params: Any,
            observations: Any,
            actions: jnp.array,
            behaviour_log_probs: jnp.array,
            target_values: jnp.array,
            advantages: jnp.array,
            behavior_values: jnp.array,
        ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
            """Surrogate loss using clipped probability ratios."""

            distribution_params, values = trainer.attr.networks.apply(
                params, observations
            )
            log_probs = trainer.attr.networks.log_prob(distribution_params, actions)
            entropy = trainer.attr.networks.entropy(distribution_params)

            # Compute importance sampling weights: current policy / behavior policy.
            rhos = jnp.exp(log_probs - behaviour_log_probs)
            clipping_epsilon = self.config.clipping_epsilon

            policy_loss = rlax.clipped_surrogate_pg_loss(
                rhos, advantages, clipping_epsilon
            )

            # Value function loss. Exclude the bootstrap value
            unclipped_value_error = target_values - values
            unclipped_value_loss = unclipped_value_error ** 2

            if self.config.clip_value:
                # Clip values to reduce variablility during critic training.
                clipped_values = behavior_values + jnp.clip(
                    values - behavior_values,
                    -clipping_epsilon,
                    clipping_epsilon,
                )
                clipped_value_error = target_values - clipped_values
                clipped_value_loss = clipped_value_error ** 2
                value_loss = jnp.mean(
                    jnp.fmax(unclipped_value_loss, clipped_value_loss)
                )
            else:
                value_loss = jnp.mean(unclipped_value_loss)

            # Entropy regulariser.
            entropy_loss = -jnp.mean(entropy)

            total_loss = (
                policy_loss
                + value_loss * self.config.value_cost
                + entropy_loss * self.config.entropy_cost
            )
            return total_loss, {
                "loss_total": total_loss,
                "loss_policy": policy_loss,
                "loss_value": value_loss,
                "loss_entropy": entropy_loss,
            }

        trainer.attr.loss_fn = loss
