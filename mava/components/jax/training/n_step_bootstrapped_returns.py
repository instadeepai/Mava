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
from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import rlax
from chex import Array, Scalar

from mava.components.jax.training.base import Utility
from mava.core_jax import SystemTrainer


@dataclass
class NStepBootStrappedReturnsConfig:
    n_step: int = 10
    lambda_t: Union[Array, Scalar] = 1.0


class NStepBootStrappedReturns(Utility):
    def __init__(
        self,
        config: NStepBootStrappedReturnsConfig = NStepBootStrappedReturnsConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_utility_fns(self, trainer: SystemTrainer) -> None:
        """_summary_"""

        def n_step_bootstrapped_returns(
            rewards: jnp.ndarray,
            discounts: jnp.ndarray,
            values: jnp.ndarray,
            n_step: int,
            lambda_t: Union[Array, Scalar],
        ) -> jnp.ndarray:
            """Uses n-step boostrapped returns to compute target values"""

            target_values = rlax.n_step_bootstrapped_returns(
                rewards, discounts, values, n_step, lambda_t, True
            )

            return target_values

        trainer.store.n_step_fn = n_step_bootstrapped_returns

    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "advantage_estimator"

    @staticmethod
    def config_class() -> Callable:
        return NStepBootStrappedReturnsConfig
