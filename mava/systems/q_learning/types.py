# Copyright 2022 InstaDeep Ltd. All rights reserved.
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
from typing import Any, Callable, Dict, NamedTuple, Tuple  # noqa

import optax
from chex import PRNGKey
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
from flax.core.scope import FrozenVariableDict
from jax import Array
from jumanji.env import State
from typing_extensions import TypeAlias

from mava.types import Observation

Metrics = Dict[str, Array]


class Transition(NamedTuple):
    """Transition for recurrent Q-learning."""

    obs: Observation
    action: Array
    reward: Array
    terminal: Array
    term_or_trunc: Array
    next_obs: Observation


BufferState: TypeAlias = TrajectoryBufferState[Transition]


class DDQNParams(NamedTuple):
    """Double Q-learning network parameters."""

    online: FrozenVariableDict
    target: FrozenVariableDict


class LearnerState(NamedTuple):
    """State of the learner in an interaction-training loop."""

    # Interaction vars
    obs: Observation
    terminal: Array
    term_or_trunc: Array
    hidden_state: Array
    env_state: State
    time_steps: Array

    # Train vars
    train_steps: Array
    opt_state: optax.OptState

    # Shared vars
    buffer_state: TrajectoryBufferState
    params: DDQNParams
    key: PRNGKey


class ActionSelectionState(NamedTuple):
    """Everything used for action selection apart from the observation."""

    online_params: FrozenVariableDict
    hidden_state: Array
    time_steps: int
    key: PRNGKey


class ActionState(NamedTuple):
    """The carry in the interaction loop."""

    action_selection_state: ActionSelectionState
    env_state: State
    buffer_state: BufferState
    obs: Observation
    terminal: Array
    term_or_trunc: Array


class TrainState(NamedTuple):
    """The carry in the training loop."""

    buffer_state: BufferState
    params: DDQNParams
    opt_state: optax.OptState
    train_steps: Array
    key: PRNGKey
