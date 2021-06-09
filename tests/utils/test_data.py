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

from typing import Dict

from dm_env import StepType, TimeStep

from mava.types import OLT
from mava.utils.wrapper_utils import SeqTimestepDict


def get_seq_timesteps_1() -> TimeStep:
    return TimeStep(
        step_type=StepType.FIRST,
        reward=0.0,
        discount=1.0,
        observation=OLT(observation=[0.1, 0.3, 0.7], legal_actions=[1], terminal=[0.0]),
    )


def get_expected_parallel_timesteps_1() -> TimeStep:
    return TimeStep(
        step_type=StepType.FIRST,
        reward={"agent_0": 0.0, "agent_1": 0.0, "agent_2": 0.0},
        discount={"agent_0": 1.0, "agent_1": 1.0, "agent_2": 1.0},
        observation={
            "agent_0": OLT(
                observation=[0.1, 0.3, 0.7],
                legal_actions=[1],
                terminal=[0.0],
            ),
            "agent_1": OLT(
                observation=[0.1, 0.3, 0.7],
                legal_actions=[1],
                terminal=[0.0],
            ),
            "agent_2": OLT(
                observation=[0.1, 0.3, 0.7],
                legal_actions=[1],
                terminal=[0.0],
            ),
        },
    )


def get_seq_timesteps_dict_2() -> Dict[str, SeqTimestepDict]:
    return {
        "agent_0": {
            "timestep": TimeStep(
                step_type=StepType.FIRST,
                reward=-1,
                discount=0.8,
                observation=OLT(
                    observation=[0.1, 0.5, 0.7], legal_actions=[1], terminal=[0.0]
                ),
            ),
            "action": 0,
        },
        "agent_1": {
            "timestep": TimeStep(
                step_type=StepType.FIRST,
                reward=0.0,
                discount=0.8,
                observation=OLT(
                    observation=[0.8, 0.3, 0.7], legal_actions=[1], terminal=[0.0]
                ),
            ),
            "action": 2,
        },
        "agent_2": {
            "timestep": TimeStep(
                step_type=StepType.FIRST,
                reward=1,
                discount=1.0,
                observation=OLT(
                    observation=[0.9, 0.9, 0.8], legal_actions=[1], terminal=[0.0]
                ),
            ),
            "action": 1,
        },
    }


def get_expected_parallel_timesteps_2() -> TimeStep:
    return TimeStep(
        step_type=StepType.FIRST,
        reward={"agent_0": -1, "agent_1": 0.0, "agent_2": 1},
        discount={"agent_0": 0.8, "agent_1": 0.8, "agent_2": 1.0},
        observation={
            "agent_0": OLT(
                observation=[0.1, 0.5, 0.7],
                legal_actions=[1],
                terminal=[0.0],
            ),
            "agent_1": OLT(
                observation=[0.8, 0.3, 0.7],
                legal_actions=[1],
                terminal=[0.0],
            ),
            "agent_2": OLT(
                observation=[0.9, 0.9, 0.8],
                legal_actions=[1],
                terminal=[0.0],
            ),
        },
    )
