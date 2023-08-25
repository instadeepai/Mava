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

from typing import Dict, NamedTuple, Optional

import chex
from flax.core.frozen_dict import FrozenDict
from jumanji.environments.routing.robot_warehouse import State
from jumanji.types import TimeStep
from optax._src.base import OptState

from mava.wrappers.jumanji import LogEnvState


class PPOTransition(NamedTuple):
    """Transition tuple for PPO."""

    done: chex.Array
    action: chex.Array
    value: chex.Array
    reward: chex.Array
    log_prob: chex.Array
    obs: chex.Array
    info: Dict


class Params(NamedTuple):
    """Parameters of an actor critic network."""

    actor_params: FrozenDict
    critic_params: FrozenDict


class OptStates(NamedTuple):
    """OptStates of actor critic learner."""

    actor_opt_state: OptState
    critic_opt_state: OptState


class HiddenStates(NamedTuple):
    """Hidden states for an actor critic learner."""

    policy_hidden_state: chex.Array
    critic_hidden_state: chex.Array


class LearnerState(NamedTuple):
    """State of the learner."""

    params: Params
    opt_states: OptStates
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep


class RNNLearnerState(NamedTuple):
    """State of the `Learner` for recurrent architectures."""

    params: Params
    opt_states: OptStates
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep
    dones: chex.Array
    hstates: HiddenStates


class EvalState(NamedTuple):
    """State of the evaluator."""

    key: chex.PRNGKey
    env_state: State
    timestep: TimeStep
    step_count_: chex.Numeric = None
    return_: chex.Numeric = None


class RNNEvalState(NamedTuple):
    """State of the evaluator for recurrent architectures."""

    key: chex.PRNGKey
    env_state: State
    timestep: TimeStep
    dones: chex.Array
    hstate: chex.Array
    step_count_: chex.Numeric = None
    return_: chex.Numeric = None


class ExperimentOutput(NamedTuple):
    """Experiment output."""

    episodes_info: Dict[str, chex.Array]
    learner_state: Optional[LearnerState] = None
    total_loss: chex.Array = None
    value_loss: chex.Array = None
    loss_actor: chex.Array = None
    entropy: chex.Array = None
