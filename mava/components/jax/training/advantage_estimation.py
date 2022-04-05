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

"""Trainer components for advantage calculations."""
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import rlax

from mava.components.jax.training.base import Utility
from mava.core_jax import SystemTrainer


@dataclass
class GAEConfig:
    gae_lambda: float = 0.95
    max_abs_reward: float = np.inf


class GAE(Utility):
    def __init__(
        self,
        config: GAEConfig = GAEConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_utility_fns(self, trainer: SystemTrainer) -> None:
        """_summary_"""

        def gae_advantages(
            rewards: jnp.array, discounts: jnp.array, values: jnp.array
        ) -> Tuple[jnp.ndarray, jnp.array]:
            """Uses truncated GAE to compute advantages."""

            # Apply reward clipping.
            max_abs_reward = self.config.max_abs_reward
            rewards = jnp.clip(rewards, -max_abs_reward, max_abs_reward)

            advantages = rlax.truncated_generalized_advantage_estimation(
                rewards[:-1], discounts[:-1], self.config.gae_lambda, values
            )
            advantages = jax.lax.stop_gradient(advantages)

            # Exclude the bootstrap value
            target_values = values[:-1] + advantages
            target_values = jax.lax.stop_gradient(target_values)

            return advantages, target_values

        trainer.store.gae_fn = gae_advantages

    @property
    def name(self) -> str:
        """_summary_

        Returns:
            _description_
        """
        return "advantage_estimator"
