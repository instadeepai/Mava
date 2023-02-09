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

"""Execution components for system builders"""

import abc
from dataclasses import dataclass
from typing import Any, List, Optional, Type

import optax
from optax._src import base as optax_base

from mava.callbacks import Callback
from mava.components import Component
from mava.core_jax import SystemBuilder


class Optimisers(Component):
    @abc.abstractmethod
    def __init__(
        self,
        config: Any,
    ):
        """Abstract component defining the skeleton for initialising optimisers.

        Args:
            config: Any.
        """
        self.config = config

    @abc.abstractmethod
    def on_building_init_start(self, builder: SystemBuilder) -> None:
        """Create and store the optimisers.

        Args:
            builder: SystemBuilder.

        Returns:
            None.
        """
        pass

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "optimisers"

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        Returns:
            List of required component classes.
        """
        return []


@dataclass
class ActorCriticOptimisersConfig:
    policy_learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3
    adam_epsilon: float = 1e-5
    max_gradient_norm: float = 0.5
    policy_optimiser: Optional[optax_base.GradientTransformation] = None
    critic_optimiser: Optional[optax_base.GradientTransformation] = None


class ActorCriticOptimisers(Optimisers):
    def __init__(
        self,
        config: ActorCriticOptimisersConfig = ActorCriticOptimisersConfig(),
    ):
        """Component defines the default way to initialise optimisers.

        Args:
            config: DefaultOptimisers.
        """
        self.config = config

    def on_building_init_start(self, builder: SystemBuilder) -> None:
        """Create and store the optimisers.

        Args:
            builder: SystemBuilder.

        Returns:
            None.
        """
        # Build the optimiser function here
        if not self.config.policy_optimiser:
            builder.store.policy_optimiser = optax.chain(
                optax.clip_by_global_norm(self.config.max_gradient_norm),
                optax.scale_by_adam(eps=self.config.adam_epsilon),
                optax.scale(-self.config.policy_learning_rate),
            )
        else:
            builder.store.policy_optimiser = self.config.policy_optimiser

        if not self.config.critic_optimiser:
            builder.store.critic_optimiser = optax.chain(
                optax.clip_by_global_norm(self.config.max_gradient_norm),
                optax.scale_by_adam(eps=self.config.adam_epsilon),
                optax.scale(-self.config.policy_learning_rate),
            )
        else:
            builder.store.critic_optimiser = self.config.critic_optimiser
