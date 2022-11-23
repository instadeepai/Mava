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
from typing import List, Tuple, Type

import jax
import jax.numpy as jnp
import numpy as np
import rlax

from mava.callbacks import Callback
from mava.components.training.base import Utility
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
        """Component defines advantage estimation function.

        Args:
            config: GAEConfig.
        """
        self.config = config

    def on_training_utility_fns(self, trainer: SystemTrainer) -> None:
        """Create and store a GAE advantage function.

        Args:
            trainer: SystemTrainer.

        Returns:
            None.
        """

        def gae_advantages(
            rewards: jnp.ndarray,
            valid_steps: jnp.ndarray,
            values: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Use truncated GAE to compute advantages.

            Args:
                rewards: Agent rewards.
                valid_steps: Agent took a valid step.
                values: Agent value estimations.

            Returns:
                Tuple of advantage values, target values.
            """

            # Apply reward clipping.
            max_abs_reward = self.config.max_abs_reward
            rewards = jnp.clip(rewards, -max_abs_reward, max_abs_reward)

            advantages = rlax.truncated_generalized_advantage_estimation(
                rewards[:-1],
                valid_steps[:-1],
                self.config.gae_lambda,
                values,
            )
            advantages = jax.lax.stop_gradient(advantages)

            # Exclude the bootstrap value
            target_values = values[:-1] + advantages
            target_values = jax.lax.stop_gradient(target_values)

            return advantages, target_values

        trainer.store.gae_fn = gae_advantages

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "gae_fn"

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        None required.

        Returns:
            List of required component classes.
        """
        return []
