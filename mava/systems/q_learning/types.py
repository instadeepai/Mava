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
from typing import Dict, Generic, TypeVar

import optax
from chex import PRNGKey
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
from flax.core.scope import FrozenVariableDict
from jax import Array
from jumanji.env import State
from typing_extensions import NamedTuple, TypeAlias

from mava.types import MavaObservation

Metrics = Dict[str, Array]


class Transition(NamedTuple):
    """Transition for recurrent Q-learning."""

    obs: MavaObservation
    action: Array
    reward: Array
    terminal: Array
    term_or_trunc: Array
    # Even though we use a trajectory buffer we need to store both obs and next_obs.
    # This is because of how the `AutoResetWrapper` returns obs at the end of an episode.
    next_obs: MavaObservation


BufferState: TypeAlias = TrajectoryBufferState[Transition]


class QNetParams(NamedTuple):
    """Double Q-learning network parameters."""

    online: FrozenVariableDict
    target: FrozenVariableDict


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
    obs: MavaObservation
    terminal: Array
    term_or_trunc: Array


class QMIXParams(NamedTuple):
    online: FrozenVariableDict
    target: FrozenVariableDict
    mixer_online: FrozenVariableDict
    mixer_target: FrozenVariableDict


QLearningParams = TypeVar("QLearningParams", QNetParams, QMIXParams)


class LearnerState(NamedTuple, Generic[QLearningParams]):
    """State of the learner in an interaction-training loop."""

    # Interaction vars
    obs: MavaObservation
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
    params: QLearningParams
    key: PRNGKey


class TrainState(NamedTuple, Generic[QLearningParams]):
    """The carry in the training loop."""

    buffer_state: BufferState
    params: QLearningParams
    opt_state: optax.OptState
    train_steps: Array
    key: PRNGKey


class SebulbaLearnerState(NamedTuple):
    """State of the learner for the Sebulba architecture."""

    params: QNetParams
    opt_states: optax.OptState
    step_counter: int
