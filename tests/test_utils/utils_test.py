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


import numpy as np

from mava.utils.wrapper_utils import (
    broadcast_timestep_to_all_agents,
    convert_seq_timestep_and_actions_to_parallel,
)
from tests.test_utils.test_data import (
    get_expected_parallel_timesteps_1,
    get_expected_parallel_timesteps_2,
    get_seq_timesteps_1,
    get_seq_timesteps_dict_2,
)


class TestUtils:
    # Test broadcast_timestep_to_all_agents function.
    def test_broadcast_timestep_to_all_agents(self) -> None:
        seq_timesteps = get_seq_timesteps_1()

        expected_parallel_timesteps = get_expected_parallel_timesteps_1()

        possible_agents = ["agent_0", "agent_1", "agent_2"]
        parallel_timesteps = broadcast_timestep_to_all_agents(
            seq_timesteps, possible_agents
        )

        assert np.array_equal(
            expected_parallel_timesteps,
            parallel_timesteps,
        ), "Failed to broadcast timesteps."

    # Test convert_seq_timestep_and_actions_to_parallel
    def test_convert_seq_timestep_and_actions_to_parallel(self) -> None:

        timesteps = get_seq_timesteps_dict_2()
        expected_parallel_timesteps = get_expected_parallel_timesteps_2()

        possible_agents = ["agent_0", "agent_1", "agent_2"]
        expected_actions = {"agent_0": 0, "agent_1": 2, "agent_2": 1}
        (
            parallel_actions,
            parallel_timesteps,
        ) = convert_seq_timestep_and_actions_to_parallel(timesteps, possible_agents)

        assert np.array_equal(
            expected_parallel_timesteps,
            parallel_timesteps,
        ), "Failed to convert seq timesteps to parallel."

        assert np.array_equal(
            parallel_actions,  # type: ignore
            expected_actions,  # type: ignore
        ), "Failed to convert seq actions to parallel."
