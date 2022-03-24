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

"""Commonly used adder components for system builders"""
import abc
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from mava.components.jax.training import TrainingState, Utility
from mava.core_jax import SystemTrainer


@dataclass
class InitialStateConfig:
    pass


class InitialState(Utility):
    def __init__(
        self,
        config: InitialStateConfig = InitialStateConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    @abc.abstractmethod
    def on_training_utility_fns(self, trainer: SystemTrainer) -> None:
        """_summary_"""
        networks = trainer.attr.networks
        optimizer = trainer.attr.optimizer

        def make_initial_state(key: jnp.ndarray) -> TrainingState:
            """Initialises the training state (parameters and optimiser state)."""
            key_init, key_state = jax.random.split(key)
            initial_params = networks.network.init(key_init)
            initial_opt_state = optimizer.init(initial_params)
            return TrainingState(
                params=initial_params, opt_state=initial_opt_state, random_key=key_state
            )

        trainer.attr.initial_state_fn = make_initial_state

    @property
    def name(self) -> str:
        """_summary_

        Returns:
            _description_
        """
        return "initial_state_fn"
