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

from typing import Dict, Tuple

import flax.linen as nn
import optax
from chex import Array, PRNGKey
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
from flax.core.scope import FrozenVariableDict
from typing_extensions import NamedTuple, TypeAlias

from mava.types import State

Metrics: TypeAlias = Dict[str, Array]
Networks: TypeAlias = Tuple[nn.Module, nn.Module]
Optimisers: TypeAlias = Tuple[
    optax.GradientTransformation, optax.GradientTransformation, optax.GradientTransformation
]


class Qs(NamedTuple):
    q1: FrozenVariableDict
    q2: FrozenVariableDict


class QsAndTarget(NamedTuple):
    online: Qs
    targets: Qs


class SacParams(NamedTuple):
    actor: FrozenVariableDict
    q: QsAndTarget
    log_alpha: Array


class OptStates(NamedTuple):
    actor: optax.OptState
    q: optax.OptState
    alpha: optax.OptState


class Transition(NamedTuple):
    obs: Array
    action: Array
    reward: Array
    done: Array
    next_obs: Array


BufferState: TypeAlias = TrajectoryBufferState[Transition]


class LearnerState(NamedTuple):
    obs: Array
    env_state: State
    buffer_state: BufferState
    params: SacParams
    opt_states: OptStates
    t: Array
    key: PRNGKey
