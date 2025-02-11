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

from functools import cached_property
from typing import Any, Callable, Dict, Generic, Optional, Protocol, Tuple, TypeVar, Union

import chex
import jumanji.specs as specs
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
Metrics: TypeAlias = Dict[str, chex.Array]


class MarlEnv(Protocol):
    """The API used by mava for environments.

    A mava environment simply uses the Jumanji env API with a few added attributes.
    For examples of how to add custom environments to Mava see `mava/wrappers/jumanji.py`.
    Jumanji API docs: https://instadeepai.github.io/jumanji/#basic-usage
    """

    num_agents: int
    time_limit: int
    action_dim: int

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Resets the environment to an initial state.

        Args:
            key: random key used to reset the environment.

        Returns:
            state: State object corresponding to the new state of the environment,
            timestep: TimeStep object corresponding the first timestep returned by the environment,
        """
        ...

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        """Run one timestep of the environment's dynamics.

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the action to take.

        Returns:
            state: State object corresponding to the next state of the environment,
            timestep: TimeStep object corresponding the timestep returned by the environment,
        """
        ...

    @cached_property
    def observation_spec(self) -> specs.Spec:
        """Returns the observation spec.

        Returns:
            observation_spec: a NestedSpec tree of spec.
        """
        ...

    @cached_property
    def action_spec(self) -> specs.Spec:
        """Returns the action spec.

        Returns:
            action_spec: a NestedSpec tree of spec.
        """
        ...

    @cached_property
    def reward_spec(self) -> specs.Array:
        """Describes the reward returned by the environment. By default, this is assumed to be a
        single float.

        Returns:
            reward_spec: a `specs.Array` spec.
        """
        ...

    @cached_property
    def discount_spec(self) -> specs.BoundedArray:
        """Describes the discount returned by the environment. By default, this is assumed to be a
        single float between 0 and 1.

        Returns:
            discount_spec: a `specs.BoundedArray` spec.
        """
        ...

    @property
    def unwrapped(self) -> Any:
        """Retuns: the innermost environment (without any wrappers applied)."""
        ...


class Observation(NamedTuple):
    """The observation that the agent sees.

    agents_view: the agent's view of the environment.
    action_mask: boolean array specifying, for each agent, which action is legal.
    step_count: the number of steps elapsed since the beginning of the episode.
    """

    agents_view: chex.Array  # (num_agents, num_obs_features)
    action_mask: chex.Array  # (num_agents, num_actions)
    step_count: Optional[chex.Array] = None  # (num_agents, )


class ObservationGlobalState(NamedTuple):
    """The observation seen by agents in centralised systems.

    Extends `Observation` by adding a `global_state` attribute for centralised training.
    global_state: The global state of the environment, often a concatenation of agents' views.
    """

    agents_view: chex.Array  # (num_agents, num_obs_features)
    action_mask: chex.Array  # (num_agents, num_actions)
    global_state: chex.Array  # (num_agents, num_agents * num_obs_features)
    step_count: Optional[chex.Array] = None  # (num_agents, )


RNNObservation: TypeAlias = Tuple[Observation, Done]
RNNGlobalObservation: TypeAlias = Tuple[ObservationGlobalState, Done]
MavaObservation: TypeAlias = Union[Observation, ObservationGlobalState]

# `MavaState` is the main type passed around in our systems. It is often used as a scan carry.
# Types like: `LearnerState` (mava/systems/<system_name>/types.py) are `MavaState`s.
MavaState = TypeVar("MavaState")
MavaTransition = TypeVar("MavaTransition")


class ExperimentOutput(NamedTuple, Generic[MavaState]):
    """Experiment output."""

    learner_state: MavaState
    episode_metrics: Metrics
    train_metrics: Metrics


LearnerFn = Callable[[MavaState], ExperimentOutput[MavaState]]
SebulbaLearnerFn = Callable[[MavaState, MavaTransition], Tuple[MavaState, Metrics]]
ActorApply = Callable[[FrozenDict, Observation], Distribution]
CriticApply = Callable[[FrozenDict, Observation], Value]
RecActorApply = Callable[
    [FrozenDict, HiddenState, RNNObservation], Tuple[HiddenState, Distribution]
]
RecCriticApply = Callable[[FrozenDict, HiddenState, RNNObservation], Tuple[HiddenState, Value]]
