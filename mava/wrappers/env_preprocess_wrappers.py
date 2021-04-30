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

from typing import Any, Dict, List, Union

import gym
from supersuit.aec_wrappers import ObservationWrapper as SeqObservationWrapper
from supersuit.parallel_wrappers import ObservationWrapper as ParObservationWrapper

from mava.types import Observation
from mava.wrappers.pettingzoo import (
    PettingZooAECEnvWrapper,
    PettingZooParallelEnvWrapper,
)

"""
GYM Preprocess Wrappers.

Other gym preprocess wrappers:
    https://github.com/PettingZoo-Team/SuperSuit/blob/1f02289e8f51082aa50a413b34700b67042410c6/supersuit/gym_wrappers.py
    https://github.com/openai/gym/tree/master/gym/wrappers
"""

PettingZooEnv = Union[PettingZooAECEnvWrapper, PettingZooParallelEnvWrapper]


class StandardizeObservationGym(gym.ObservationWrapper):
    """
    Standardize observations
    Ensures mean of 0 and standard deviation of 1 (unit variance) for obs.
    From https://github.com/ikostrikov/pytorch-a3c/blob/e898f7514a03de73a2bf01e7b0f17a6f93963389/envs.py # noqa: E501
    """

    def __init__(self, env: gym.Env = None) -> None:
        super(StandardizeObservationGym, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def _observation(self, observation: Any) -> Any:
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + observation.mean() * (
            1 - self.alpha
        )
        self.state_std = self.state_std * self.alpha + observation.std() * (
            1 - self.alpha
        )

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)


"""
Petting Zoo Env Preprocess Wrappers.

Other PZ preprocess wrappers:
    https://github.com/PettingZoo-Team/SuperSuit
"""


class StandardizeObservation:
    """
    Standardize observations
    Ensures mean of 0 and standard deviation of 1 (unit variance) for obs.
    Adapted from https://github.com/ikostrikov/pytorch-a3c/blob/e898f7514a03de73a2bf01e7b0f17a6f93963389/envs.py # noqa: E501
    """

    def __init__(self, env: PettingZooEnv = None, load_params: Dict = None):
        self._ini_params(load_params)
        self.env = env

    def _ini_params(self, load_params: Dict = None) -> None:
        if load_params:
            self._internal_state = load_params
        else:
            params = {}
            for agent in self.env.possible_agents:  # type:ignore
                params[agent] = {
                    "state_mean": 0,
                    "state_std": 0,
                    "alpha": 0.9999,
                    "num_steps": 0,
                }
            self._internal_state = params

    def _get_updated_observation(
        self,
        agent: List,
        observation: Observation,
    ) -> Observation:

        state_mean = self._internal_state[agent].get("state_mean")
        state_std = self._internal_state[agent].get("state_std")
        alpha = self._internal_state[agent].get("alpha")
        num_steps = self._internal_state[agent].get("num_steps")

        state_mean = state_mean * alpha + observation.mean() * (  # type:ignore
            1 - alpha
        )
        state_std = state_std * alpha + observation.std() * (1 - alpha)  # type:ignore

        steps = num_steps
        # If steps is zero, this would result in div by zero error.
        if steps == 0:
            steps = 1

        unbiased_mean = state_mean / (1 - pow(alpha, steps))
        unbiased_std = state_std / (1 - pow(alpha, steps))

        self._internal_state[agent]["state_mean"] = state_mean
        self._internal_state[agent]["state_std"] = state_std
        self._internal_state[agent]["alpha"] = alpha

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)


class StandardizeObservationSeq(SeqObservationWrapper, StandardizeObservation):
    """Standardize Obs in Sequential Env"""

    def __init__(self, env: PettingZooEnv) -> None:
        super().__init__(env)

    def reset(self) -> None:
        self._ini_params()
        super().reset()

    def _modify_observation(self, agent: List, observation: Observation) -> Observation:
        self._internal_state[agent]["num_steps"] = (
            int(self._internal_state[agent]["num_steps"]) + 1
        )
        return self._get_updated_observation(agent, observation)

    def _check_wrapper_params(self) -> None:
        return

    def _modify_spaces(self) -> None:
        return


class StandardizeObservationPar(ParObservationWrapper, StandardizeObservation):
    """Standardize Obs in Parallel Env"""

    def __init__(self, env: PettingZooEnv) -> None:
        super().__init__(env)

    def reset(self) -> None:
        self._ini_params()
        return super().reset()

    def _modify_observation(self, agent: List, observation: Observation) -> Observation:
        self._internal_state[agent]["num_steps"] = (
            int(self._internal_state[agent]["num_steps"]) + 1
        )
        return self._get_updated_observation(agent, observation)

    def _check_wrapper_params(self) -> None:
        return

    def _modify_spaces(self) -> None:
        return
