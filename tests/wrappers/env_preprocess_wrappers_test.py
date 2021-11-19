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
import pytest
from _pytest.monkeypatch import MonkeyPatch
from supersuit import dtype_v0, normalize_obs_v0, reward_lambda_v0

from mava.wrappers.env_preprocess_wrappers import (
    StandardizeObservationParallel,
    StandardizeObservationSequential,
    StandardizeRewardParallel,
    StandardizeRewardSequential,
)
from tests.conftest import EnvSpec, EnvType, Helpers


@pytest.mark.parametrize(
    "env_spec",
    [
        EnvSpec("pettingzoo.atari.pong_v2", EnvType.Parallel),
        EnvSpec("pettingzoo.atari.pong_v2", EnvType.Sequential),
    ],
)
class TestEnvPreprocessWrapper:
    """Test env preprocessors"""

    # Test normalizing obs to be within range 0 and 1
    def test_preprocess_wrapper_obs_0_normalize(
        self,
        env_spec: EnvSpec,
        helpers: Helpers,
        monkeypatch: MonkeyPatch,
    ) -> None:

        wrapped_env, _ = helpers.get_wrapped_env(
            env_spec,
            env_preprocess_wrappers=[
                (dtype_v0, {"dtype": np.float32}),
                (normalize_obs_v0, None),
            ],
        )
        timestep = wrapped_env.reset()
        if type(timestep) == tuple:
            initial_dm_env_timestep, env_extras = timestep
        else:
            initial_dm_env_timestep = timestep

        agents = wrapped_env.agents

        helpers.verify_observations_are_normalized(
            initial_dm_env_timestep.observation, agents, env_spec
        )

        # Parallel env_types
        if env_spec.env_type == EnvType.Parallel:
            for i in range(50):
                test_agents_actions = {
                    agent: wrapped_env.action_spaces[agent].sample() for agent in agents
                }
                curr_dm_timestep = wrapped_env.step(test_agents_actions)
                helpers.verify_observations_are_normalized(
                    curr_dm_timestep.observation, agents, env_spec
                )

        # Sequential env_types
        elif env_spec.env_type == EnvType.Sequential:
            for i in range(50):
                for agent in agents:
                    test_agent_actions = wrapped_env.action_spaces[agent].sample()
                    curr_dm_timestep = wrapped_env.step(test_agent_actions)
                    helpers.verify_observations_are_normalized(
                        curr_dm_timestep.observation, agents, env_spec
                    )

    # Test standardize obs to have mean 0 and std 1
    def test_preprocess_wrapper_obs_1_standardize(
        self,
        env_spec: EnvSpec,
        helpers: Helpers,
        monkeypatch: MonkeyPatch,
    ) -> None:

        # Parallel env_types
        if env_spec.env_type == EnvType.Parallel:
            StandardizeObservation = StandardizeObservationParallel
            # StandardizeObservation = parallel_wrapper_fn(
            # StandardizeObservationSequential
            # )
            # StandardizeObservation = StandardizeObservationSequential

        # Sequential env_types
        elif env_spec.env_type == EnvType.Sequential:
            StandardizeObservation = StandardizeObservationSequential

        wrapped_env, _ = helpers.get_wrapped_env(
            env_spec,
            env_preprocess_wrappers=[
                (dtype_v0, {"dtype": np.float32}),
                (StandardizeObservation, None),
            ],
        )
        # if env_spec.env_type == EnvType.Parallel:
        #     wrapped_env = to_parallel(StandardizeObservation(wrapped_env))
        #     # wrapped_env = to_parallel(wrapped_env)

        _ = wrapped_env.reset()

        agents = wrapped_env.agents

        # Parallel env_types
        if env_spec.env_type == EnvType.Parallel:
            test_agents_actions = {
                agent: wrapped_env.action_spaces[agent].sample() for agent in agents
            }
            for i in range(50):
                curr_dm_timestep = wrapped_env.step(test_agents_actions)
                helpers.verify_observations_are_standardized(
                    curr_dm_timestep.observation, agents, env_spec
                )

        # Sequential env_types
        elif env_spec.env_type == EnvType.Sequential:
            for i in range(50):
                for agent in agents:
                    test_agent_actions = wrapped_env.action_spaces[agent].sample()
                    curr_dm_timestep = wrapped_env.step(test_agent_actions)
                    helpers.verify_observations_are_standardized(
                        curr_dm_timestep.observation, agents, env_spec
                    )

    # Test normalizing rewards to be within range 0.2 and 1
    # TODO(Kale-ab): Test more than min and max.
    def test_preprocess_wrapper_reward_0_normalize(
        self,
        env_spec: EnvSpec,
        helpers: Helpers,
        monkeypatch: MonkeyPatch,
    ) -> None:

        min = 0.2
        max = 1

        # Parallel env_types
        if env_spec.env_type == EnvType.Parallel:
            StandardizeReward = StandardizeRewardParallel

        # Sequential env_types
        elif env_spec.env_type == EnvType.Sequential:
            StandardizeReward = StandardizeRewardSequential

        wrapped_env, _ = helpers.get_wrapped_env(
            env_spec,
            env_preprocess_wrappers=[
                (dtype_v0, {"dtype": np.float32}),
                (StandardizeReward, {"lower_bound": min, "upper_bound": max}),
            ],
        )

        _ = wrapped_env.reset()
        agents = wrapped_env.agents

        # Parallel env_types
        if env_spec.env_type == EnvType.Parallel:
            for i in range(50):
                test_agents_actions = {
                    agent: wrapped_env.action_spaces[agent].sample() for agent in agents
                }
                curr_dm_timestep = wrapped_env.step(test_agents_actions)
                helpers.verify_reward_is_normalized(
                    curr_dm_timestep.reward, agents, env_spec, min=min, max=max
                )

        # Sequential env_types
        elif env_spec.env_type == EnvType.Sequential:
            for i in range(50):
                for agent in agents:
                    test_agent_actions = wrapped_env.action_spaces[agent].sample()
                    curr_dm_timestep = wrapped_env.step(test_agent_actions)
                    helpers.verify_reward_is_normalized(
                        curr_dm_timestep.reward, agents, env_spec, min=min, max=max
                    )

    # Test custom reward shaping function
    def test_preprocess_wrapper_reward_1_custom_function(
        self,
        env_spec: EnvSpec,
        helpers: Helpers,
        monkeypatch: MonkeyPatch,
    ) -> None:

        wrapped_env, _ = helpers.get_wrapped_env(
            env_spec,
            env_preprocess_wrappers=[
                (dtype_v0, {"dtype": np.float32}),
                (reward_lambda_v0, {"change_reward_fn": lambda r: r + 100}),
            ],
        )

        _ = wrapped_env.reset()
        agents = wrapped_env.agents

        # Parallel env_types
        if env_spec.env_type == EnvType.Parallel:
            test_agents_actions = {
                agent: wrapped_env.action_spaces[agent].sample() for agent in agents
            }
            curr_dm_timestep = wrapped_env.step(test_agents_actions)
            for agent in agents:
                assert (
                    curr_dm_timestep.reward[agent] >= 100
                ), "Failed custom reward shaping. "

        # Sequential env_types
        elif env_spec.env_type == EnvType.Sequential:
            for agent in agents:
                test_agent_actions = wrapped_env.action_spaces[agent].sample()
                curr_dm_timestep = wrapped_env.step(test_agent_actions)
                assert curr_dm_timestep.reward >= 100, "Failed custom reward shaping. "
