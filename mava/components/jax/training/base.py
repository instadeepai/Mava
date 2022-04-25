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

"""Base Trainer components."""

import abc
from typing import Any, Dict, NamedTuple

import optax

from mava.components.jax import Component
from mava.core_jax import SystemTrainer


class Batch(NamedTuple):
    """A batch of data; all shapes are expected to be [B, ...]."""

    observations: Any
    actions: Any
    advantages: Any

    # Target value estimate used to bootstrap the value function.
    target_values: Any

    # Value estimate and action log-prob at behavior time.
    behavior_values: Any
    behavior_log_probs: Any


class MCTSBatch(NamedTuple):
    """A batch of data; all shapes are expected to be [B, ...]."""

    observations: Any
    search_policies: Any
    target_values: Any


class TrainingState(NamedTuple):
    """Training state consists of network parameters and optimiser state."""

    params: Any
    opt_states: Dict[str, optax.OptState]
    random_key: Any


class Utility(Component):
    @abc.abstractmethod
    def on_training_utility_fns(self, trainer: SystemTrainer) -> None:
        """[summary]"""


class Loss(Component):
    @abc.abstractmethod
    def on_training_loss_fns(self, trainer: SystemTrainer) -> None:
        """[summary]"""

    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "loss_fn"


class Step(Component):
    @abc.abstractmethod
    def on_training_step_fn(self, trainer: SystemTrainer) -> None:
        """[summary]"""

    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "step_fn"
