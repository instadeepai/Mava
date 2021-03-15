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

"""The base system interface."""

from typing import List, Dict

import mava
import acme
from acme import types

# Internal imports.

import dm_env
import numpy as np


def _calculate_num_learner_steps(
    num_observations: int, min_observations: int, observations_per_step: float
) -> int:
    """Calculates the number of learner steps to do at step=num_observations."""
    n = num_observations - min_observations
    if n < 0:
        # Do not do any learner steps until you have seen min_observations.
        return 0
    if observations_per_step > 1:
        # One batch every 1/obs_per_step observations, otherwise zero.
        return int(n % int(observations_per_step) == 0)
    else:
        # Always return 1/obs_per_step batches every observation.
        return int(1 / observations_per_step)


# TODO: NEED TO CHANGE THIS TO MARL
class System(mava.core.Executor, acme.core.VariableSource):
    """Agent class which combines acting and learning.
    This provides an implementation of the `Actor` interface which acts and
    learns. It takes as input instances of both `acme.Actor` and `acme.Learner`
    classes, and implements the policy, observation, and update methods which
    defer to the underlying actor and learner.
    The only real logic implemented by this class is that it controls the number
    of observations to make before running a learner step. This is done by
    passing the number of `min_observations` to use and a ratio of
    `observations_per_step` := num_actor_actions / num_learner_steps.
    Note that the number of `observations_per_step` can also be in the range[0, 1]
    in order to allow the agent to take more than 1 learner step per action.
    """

    def __init__(
        self,
        executor: mava.core.Executor,
        trainer: mava.core.Trainer,
        min_observations: int,
        observations_per_step: float,
    ):
        self._executor = executor
        self._trainer = trainer
        self._min_observations = min_observations
        self._observations_per_step = observations_per_step
        self._num_observations = 0

    def select_action(
        self, agent_id: str, observation: types.NestedArray
    ) -> types.NestedArray:
        return self._executor.agent_select_action(agent_id, observation)

    def agent_observe_first(self, agent_id: str, timestep: dm_env.TimeStep):
        self._executor.agent_observe_first(agent_id, timestep)

    def agent_observe(
        self, agent_id: str, action: types.NestedArray, next_timestep: dm_env.TimeStep
    ):
        self._num_observations += 1
        self._executor.observe(agent_id, action, next_timestep)

    def agent_update(self, agent_id: str):
        num_steps = _calculate_num_learner_steps(
            num_observations=self._num_observations,
            min_observations=self._min_observations,
            observations_per_step=self._observations_per_step,
        )
        for _ in range(num_steps):
            # Run learner steps (usually means gradient steps).
            self._trainer.agent_step(agent_id)
        if num_steps > 0:
            # Update the actor weights when learner updates.
            self._executor.agent_update(agent_id)

    def agent_get_variables(
        self, agent_id: str, names: List[str]
    ) -> List[List[np.ndarray]]:
        return self._trainer.agent_get_variables(agent_id, names)

    # TODO(arnu) finish there functions for the multi-agent case
    def select_actions(
        self, observations: Dict[str, types.NestedArray]
    ) -> Dict[str, types.NestedArray]:
        actions = {}
        for agent_id, agent in self._actors.items():
            actions[agent_id] = agent.select_action(observations[agent_id])
        return actions

    def observe_first(self, timestep: dm_env.TimeStep):
        self._executor.observe_first(timestep)

    def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
        self._num_observations += 1
        self._actor.observe(action, next_timestep)

    def update(self):
        num_steps = _calculate_num_learner_steps(
            num_observations=self._num_observations,
            min_observations=self._min_observations,
            observations_per_step=self._observations_per_step,
        )
        for _ in range(num_steps):
            # Run learner steps (usually means gradient steps).
            self._learner.step()
        if num_steps > 0:
            # Update the actor weights when learner updates.
            self._actor.update()

    def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
        return self._learner.get_variables(names)


# Internal class.
