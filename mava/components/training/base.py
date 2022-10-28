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

from mava.components import Component
from mava.core_jax import SystemTrainer


class Batch(NamedTuple):
    """A batch of data; all shapes are expected to be [B, ...]."""

    observations: Any
    policy_states: Any
    actions: Any
    advantages: Any

    # Target value estimate used to bootstrap the value function.
    target_values: Any

    # Value estimate and action log-prob at behavior time.
    behavior_values: Any
    behavior_log_probs: Any


class TrainingState(NamedTuple):
    """Training state consists of network parameters and optimiser state."""

    policy_params: Any
    critic_params: Any
    policy_opt_states: Dict[str, optax.OptState]
    critic_opt_states: Dict[str, optax.OptState]
    random_key: Any
    target_value_stats: Any
    observation_stats: Any


class Utility(Component):
    @abc.abstractmethod
    def on_training_utility_fns(self, trainer: SystemTrainer) -> None:
        """Hook to override to define training utility functions."""
