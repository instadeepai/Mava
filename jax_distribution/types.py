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

from typing import Dict, NamedTuple

import chex
from flax.core.frozen_dict import FrozenDict
from jumanji.types import TimeStep
from optax._src.base import OptState

from jax_distribution.wrappers.jumanji import LogEnvState


class PPOTransition(NamedTuple):
    """Transition tuple for PPO."""

    done: chex.Array
    action: chex.Array
    value: chex.Array
    reward: chex.Array
    log_prob: chex.Array
    obs: chex.Array
    info: Dict


class RunnerState(NamedTuple):
    """State of the `Runner`."""

    params: FrozenDict
    opt_state: OptState
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep


class ExperimentOutput(NamedTuple):
    """Experiment output."""

    runner_state: RunnerState
    info: Dict[str, Dict[str, chex.Array]]
