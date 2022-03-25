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

"""Custom Jax MAPPO system components."""

from dataclasses import dataclass

import jax.numpy as jnp

from mava.components.jax import Component
from mava.core_jax import SystemTrainer


@dataclass
class TrainerConfig:
    random_key: jnp.ndarray = 42


class Trainer(Component):
    def __init__(
        self,
        config: TrainerConfig = TrainerConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_init(self, trainer: SystemTrainer) -> None:
        """_summary_"""
        # Initialise training state (parameters and optimiser state).
        self._state = trainer.attr.make_initial_state(self.config.random_key)

        # Internalise iterator.
        self._iterator = trainer.attr.iterator
        self._sgd_step = trainer.attr.sgd_step

        # Set up logging/counting.
        # self._counter = counter or counting.Counter() # TODO: check if we need this
        self._logger = trainer.attr.logger

    def on_training_step(self, trainer: SystemTrainer) -> None:
        """Does a step of SGD and logs the results."""

        # Do a batch of SGD.
        sample = next(self._iterator)
        self._state, results = self._sgd_step(self._state, sample)

        # Update our counts and record it.
        # counts = self._counter.increment(steps=1) # TODO: add back in later

        # Snapshot and attempt to write logs.
        # self._logger.write({**results, **counts})
        self._logger.write({**results})

    @property
    def name(self) -> str:
        """_summary_

        Returns:
            _description_
        """
        return "trainer"
