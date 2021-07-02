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

from typing import Any, Dict, Sequence, Tuple, Union

import dm_env
from acme import types

import mava


def _calculate_num_learner_steps(
    num_observations: int,
    min_observations: int,
    observations_per_step: float,
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


class System(mava.core.Executor, mava.core.VariableSource):
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
        return self._executor.select_action(agent_id, observation)

    def select_actions(
        self, observations: Dict[str, types.NestedArray]
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]:
        return self._executor.select_actions(observations)

    def observe_first(
        self,
        timestep: dm_env.TimeStep,
        extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        self._executor.observe_first(timestep, extras)

    def observe(
        self,
        actions: Dict[str, types.NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        self._num_observations += 1
        self._executor.observe(actions, next_timestep, next_extras)

    def agent_observe(
        self,
        agent: str,
        action: Union[float, int, types.NestedArray],
        next_timestep: dm_env.TimeStep,
    ) -> None:
        self._num_observations += 1
        self._executor.agent_observe(agent, action, next_timestep)

    def update(self) -> None:
        num_steps = _calculate_num_learner_steps(
            num_observations=self._num_observations,
            min_observations=self._min_observations,
            observations_per_step=self._observations_per_step,
        )
        for _ in range(num_steps):
            # Run learner steps (usually means gradient steps).
            self._trainer.step()
        if num_steps > 0:
            # Update the actor weights when learner updates.
            self._executor.update()

    def get_variables(
        self, names: Sequence[str]
    ) -> Dict[str, Dict[str, types.NestedArray]]:
        return self._trainer.get_variables(names)
