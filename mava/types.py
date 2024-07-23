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

from typing import Any, Callable, Dict, Generic, Tuple, TypeVar

import chex
from flax.core.frozen_dict import FrozenDict
from jumanji.types import TimeStep
from tensorflow_probability.substrates.jax.distributions import Distribution
from typing_extensions import NamedTuple, TypeAlias

Action: TypeAlias = chex.Array
Value: TypeAlias = chex.Array
Done: TypeAlias = chex.Array
HiddenState: TypeAlias = chex.Array
# Can't know the exact type of State.
State: TypeAlias = Any


class Observation(NamedTuple):
    """The observation that the agent sees.
    agents_view: the agent's view of the environment.
    action_mask: boolean array specifying, for each agent, which action is legal.
    step_count: the number of steps elapsed since the beginning of the episode.
    """

    agents_view: chex.Array  # (num_agents, num_obs_features)
    action_mask: chex.Array  # (num_agents, num_actions)
    step_count: chex.Array  # (num_agents, )


class ObservationGlobalState(NamedTuple):
    """The observation seen by agents in centralised systems.
    Extends `Observation` by adding a `global_state` attribute for centralised training.
    global_state: The global state of the environment, often a concatenation of agents' views.
    """

    agents_view: chex.Array  # (num_agents, num_obs_features)
    action_mask: chex.Array  # (num_agents, num_actions)
    global_state: chex.Array  # (num_agents, num_agents * num_obs_features)
    step_count: chex.Array  # (num_agents, )


RNNObservation: TypeAlias = Tuple[Observation, Done]
RNNGlobalObservation: TypeAlias = Tuple[ObservationGlobalState, Done]


class EvalState(NamedTuple):
    """State of the evaluator."""

    key: chex.PRNGKey
    env_state: State
    timestep: TimeStep
    step_count: chex.Array
    episode_return: chex.Array


class RNNEvalState(NamedTuple):
    """State of the evaluator for recurrent architectures."""

    key: chex.PRNGKey
    env_state: State
    timestep: TimeStep
    dones: chex.Array
    hstate: HiddenState
    step_count: chex.Array
    episode_return: chex.Array


# `MavaState` is the main type passed around in our systems. It is often used as a scan carry.
# Types like: `EvalState` | `LearnerState` (mava/systems/<system_name>/types.py) are `MavaState`s.
MavaState = TypeVar("MavaState")


class ExperimentOutput(NamedTuple, Generic[MavaState]):
    """Experiment output."""

    learner_state: MavaState
    episode_metrics: Dict[str, chex.Array]
    train_metrics: Dict[str, chex.Array]


LearnerFn = Callable[[MavaState], ExperimentOutput[MavaState]]
EvalFn = Callable[[FrozenDict, chex.PRNGKey], ExperimentOutput[MavaState]]

ActorApply = Callable[[FrozenDict, Observation], Distribution]
CriticApply = Callable[[FrozenDict, Observation], Value]
RecActorApply = Callable[
    [FrozenDict, HiddenState, RNNObservation], Tuple[HiddenState, Distribution]
]
RecCriticApply = Callable[[FrozenDict, HiddenState, RNNObservation], Tuple[HiddenState, Value]]
