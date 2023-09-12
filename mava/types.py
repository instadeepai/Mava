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

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    Union,
)

import chex
from distrax import Distribution
from flax.core.frozen_dict import FrozenDict
from jumanji.types import TimeStep
from optax._src.base import OptState
from typing_extensions import TypeAlias

from mava.wrappers.jumanji import LogEnvState

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from flax.struct import dataclass

# Can't know the exact type of State or Timestep.
# Is there a better way to do this?
State: TypeAlias = Any
Observation: TypeAlias = Any

Action: TypeAlias = chex.Array
Value: TypeAlias = chex.Array
HiddenState: TypeAlias = chex.Array


class PPOTransition(NamedTuple):
    """Transition tuple for PPO."""

    done: chex.Array
    action: Action
    value: Value
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

    policy_hidden_state: HiddenState
    critic_hidden_state: HiddenState


# Question: we need this to be a dataclass because you can't
# subclass NamedTuple so should we make everything a dataclass?
# Could do unions of NamedTuples?
@dataclass
class LearnerState:
    """State of the learner."""

    params: Params
    opt_states: OptStates
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep


@dataclass
class RNNLearnerState(LearnerState):
    """State of the `Learner` for recurrent architectures."""

    dones: chex.Array
    hstates: HiddenStates


@dataclass
class EvalState(State):
    """State of the evaluator."""

    key: chex.PRNGKey
    env_state: State
    timestep: TimeStep
    step_count_: chex.Numeric = None
    return_: chex.Numeric = None


@dataclass
class RNNEvalState(EvalState):
    """State of the evaluator for recurrent architectures."""

    dones: chex.Array
    hstate: HiddenState


class ExperimentOutput(NamedTuple):
    """Experiment output."""

    episodes_info: Dict[str, chex.Array]
    learner_state: Optional[LearnerState] = None  # todo: why is this optional?
    total_loss: chex.Array = None
    value_loss: chex.Array = None
    loss_actor: chex.Array = None
    entropy: chex.Array = None


LearnerFn = Callable[[LearnerState], ExperimentOutput]

ActorApply = Callable[[FrozenDict, Observation], Distribution]
CriticApply = Callable[[FrozenDict, Observation], Value]
RecActorApply = Callable[[FrozenDict, HiddenState, Observation], Tuple[HiddenState, Distribution]]
RecCriticApply = Callable[[FrozenDict, HiddenState, Observation], Tuple[HiddenState, Value]]
