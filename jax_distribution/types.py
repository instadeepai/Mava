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

from typing import Dict, NamedTuple, Union

import chex
from flax.core.frozen_dict import FrozenDict
from jumanji.environments.routing.robot_warehouse import State
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


class LearnerState(NamedTuple):
    """State of the learner."""

    params: FrozenDict
    opt_state: OptState
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep


class EvalState(NamedTuple):
    """State of the evaluator."""

    key: chex.PRNGKey
    env_state: State
    timestep: TimeStep
    step_count_: chex.Numeric = None
    return_: chex.Numeric = None


class ExperimentOutput(NamedTuple):
    """Experiment output."""

    episodes_info: Dict[str, chex.Array]
    learner_state: Union[LearnerState, None] = None
    total_loss: chex.Array = None
    value_loss: chex.Array = None
    loss_actor: chex.Array = None
    entropy: chex.Array = None
