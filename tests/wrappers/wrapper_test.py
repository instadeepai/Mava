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

from typing import Any, Dict
from unittest.mock import patch

import dm_env
import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch

from mava import types
from tests.conftest import EnvSpec, EnvType, Helpers
from tests.enums import EnvSource

"""
TestEnvWrapper is a general purpose test class that runs tests for environment wrappers.
This is meant to flexibily test various environments wrappers.

    It is parametrize by an EnvSpec object:
        env_name: [name of env]
        env_type: [EnvType.Parallel/EnvType.Sequential]
        env_source: [What is source env - e.g. PettingZoo, RLLibMultiEnv or Flatland]
            - Used in confest to determine which envs and wrappers to load.

    For new environments - you might need to update the Helpers class in conftest.py.
"""


# TODO (Kale-ab): Test dying agents.
@pytest.mark.parametrize(
    "env_spec",
    [
        EnvSpec("pettingzoo.mpe.simple_spread_v2", EnvType.Parallel),
        EnvSpec("pettingzoo.mpe.simple_spread_v2", EnvType.Sequential),
        EnvSpec("pettingzoo.sisl.multiwalker_v7", EnvType.Parallel),
        EnvSpec("pettingzoo.sisl.multiwalker_v7", EnvType.Sequential),
        EnvSpec("flatland", EnvType.Parallel, EnvSource.Flatland),
        EnvSpec("tic_tac_toe", EnvType.Sequential, EnvSource.OpenSpiel),
    ],
)
class TestEnvWrapper:
    # Test that we can load a env module and that it contains agents,
    #   agents and possible_agents.
    def test_loadmodule(self, env_spec: EnvSpec, helpers: Helpers) -> None:
        env = helpers.get_env(env_spec)
        props_which_should_not_be_none = [env, env.agents, env.possible_agents]
        assert helpers.verify_all_props_not_none(
            props_which_should_not_be_none
        ), "Failed to load module"

    #  Test initialization of env wrapper, which should have
    #   a nested environment, an observation and action space for each agent.
    def test_wrapper_initialization(self, env_spec: EnvSpec, helpers: Helpers) -> None:
        wrapped_env, _ = helpers.get_wrapped_env(env_spec)
        num_agents = len(wrapped_env.agents)

        props_which_should_not_be_none = [
            wrapped_env,
            wrapped_env.environment,
            wrapped_env.observation_spec(),
            wrapped_env.action_spec(),
            wrapped_env.reward_spec(),
            wrapped_env.discount_spec(),
        ]

        assert helpers.verify_all_props_not_none(
            props_which_should_not_be_none
        ), "Failed to ini wrapped env."
        assert (
            len(wrapped_env.observation_spec()) == num_agents
        ), "Failed to generate observation specs for all agents."
        assert (
            len(wrapped_env.action_spec()) == num_agents
        ), "Failed to generate action specs for all agents."
        assert (
            len(wrapped_env.reward_spec()) == num_agents
        ), "Failed to generate reward specs for all agents."
        assert (
            len(wrapped_env.discount_spec()) == num_agents
        ), "Failed to generate discount specs for all agents."

    # Test of reset of wrapper and that dm_env_timestep has basic props.
    def test_wrapper_env_reset(self, env_spec: EnvSpec, helpers: Helpers) -> None:
        wrapped_env, _ = helpers.get_wrapped_env(env_spec)
        num_agents = len(wrapped_env.agents)

        timestep = wrapped_env.reset()
        if type(timestep) == tuple:
            dm_env_timestep, env_extras = timestep
        else:
            dm_env_timestep = timestep
        props_which_should_not_be_none = [dm_env_timestep, dm_env_timestep.observation]

        assert helpers.verify_all_props_not_none(
            props_which_should_not_be_none
        ), "Failed to ini dm_env_timestep."
        assert (
            dm_env_timestep.step_type == dm_env.StepType.FIRST
        ), "Failed to have correct StepType."
        if (
            env_spec.env_name == "tic_tac_toe"
            and env_spec.env_source == EnvSource.OpenSpiel
            and env_spec.env_type == EnvType.Sequential
        ):
            pytest.skip(
                "This test is only applicable to parralel wrappers and only works "
                "for the provided PZ sequential envs because they have 3 agents, and"
                "an OLT has length of 3 (a bug, i'd say)"
            )
        assert (
            len(dm_env_timestep.observation) == num_agents
        ), "Failed to generate observation for all agents."
        assert wrapped_env._reset_next_step is False, "_reset_next_step not set."

        helpers.assert_env_reset(wrapped_env, dm_env_timestep, env_spec)

    # Test that observations from petting zoo get converted to
    #   dm observations correctly. This only runs
    #   if wrapper has a _convert_observations or _convert_observation functions.
    def test_covert_env_to_dm_env_0_no_action_mask(
        self, env_spec: EnvSpec, helpers: Helpers
    ) -> None:
        wrapped_env, _ = helpers.get_wrapped_env(env_spec)

        # Does the wrapper have the functions we want to test
        if hasattr(wrapped_env, "_convert_observations") or hasattr(
            wrapped_env, "_convert_observation"
        ):
            #  Get agent names from env and mock out data
            agents = wrapped_env.agents
            test_agents_observations = {
                agent: np.random.rand(*wrapped_env.observation_spaces[agent].shape)
                for agent in agents
            }

            # Parallel env_types
            if env_spec.env_type == EnvType.Parallel:
                dm_env_timestep = wrapped_env._convert_observations(
                    test_agents_observations, dones={agent: False for agent in agents}
                )

                for agent in wrapped_env.agents:
                    np.testing.assert_array_equal(
                        test_agents_observations[agent],
                        dm_env_timestep[agent].observation,
                    )

                    assert (
                        bool(dm_env_timestep[agent].terminal) is False
                    ), "Failed to set terminal."

            # Sequential env_types
            elif env_spec.env_type == EnvType.Sequential:
                for agent in agents:
                    dm_env_timestep = wrapped_env._convert_observation(
                        agent, test_agents_observations[agent], done=False
                    )

                    np.testing.assert_array_equal(
                        test_agents_observations[agent],
                        dm_env_timestep.observation,
                    )
                    assert (
                        bool(dm_env_timestep.terminal) is False
                    ), "Failed to set terminal."

    # Test that observations from petting zoo get converted to
    #   dm observations correctly when empty obs are returned.
    def test_covert_env_to_dm_env_0_empty_obs(
        self, env_spec: EnvSpec, helpers: Helpers
    ) -> None:
        wrapped_env, _ = helpers.get_wrapped_env(env_spec)

        # Does the wrapper have the functions we want to test
        if hasattr(wrapped_env, "_convert_observations") or hasattr(
            wrapped_env, "_convert_observation"
        ):
            #  Get agent names from env and mock out data
            agents = wrapped_env.agents
            test_agents_observations: Dict = {}

            # Parallel env_types
            if env_spec.env_type == EnvType.Parallel:
                dm_env_timestep = wrapped_env._convert_observations(
                    test_agents_observations, dones={agent: False for agent in agents}
                )

                # We have empty OLT for all agents
                for agent in wrapped_env.agents:
                    np.testing.assert_array_equal(
                        dm_env_timestep[agent].observation,
                        np.zeros(
                            wrapped_env.observation_spaces[agent].shape,
                            dtype=wrapped_env.observation_spaces[agent].dtype,
                        ),
                    )

                    np.testing.assert_array_equal(
                        dm_env_timestep[agent].legal_actions,
                        np.ones(
                            wrapped_env.action_spaces[agent].shape,
                            dtype=wrapped_env.action_spaces[agent].dtype,
                        ),
                    )

    # Test that observations **with actions masked** from petting zoo get
    #   converted to dm observations correctly. This only runs
    #   if wrapper has a _convert_observations or _convert_observation functions.
    def test_covert_env_to_dm_env_2_with_action_mask(
        self, env_spec: EnvSpec, helpers: Helpers
    ) -> None:
        wrapped_env, _ = helpers.get_wrapped_env(env_spec)

        # Does the wrapper have the functions we want to test
        if hasattr(wrapped_env, "_convert_observations") or hasattr(
            wrapped_env, "_convert_observation"
        ):
            #  Get agent names from env and mock out data
            agents = wrapped_env.agents
            test_agents_observations = {}
            for agent in agents:
                # TODO If cont action space masking is implemented - Update
                test_agents_observations[agent] = {
                    "observation": np.random.rand(
                        *wrapped_env.observation_spaces[agent].shape
                    ),
                    "action_mask": np.random.randint(
                        2, size=wrapped_env.action_spaces[agent].shape
                    ),
                }
            # Parallel env_types
            if env_spec.env_type == EnvType.Parallel:
                dm_env_timestep = wrapped_env._convert_observations(
                    test_agents_observations,
                    dones={agent: False for agent in agents},
                )

                for agent in wrapped_env.agents:
                    np.testing.assert_array_equal(
                        test_agents_observations[agent].get("observation"),
                        dm_env_timestep[agent].observation,
                    )
                    np.testing.assert_array_equal(
                        test_agents_observations[agent].get("action_mask"),
                        dm_env_timestep[agent].legal_actions,
                    )
                    assert (
                        bool(dm_env_timestep[agent].terminal) is False
                    ), "Failed to set terminal."

            # Sequential env_types
            elif env_spec.env_type == EnvType.Sequential:
                for agent in agents:
                    dm_env_timestep = wrapped_env._convert_observation(
                        agent, test_agents_observations[agent], done=False
                    )

                    np.testing.assert_array_equal(
                        test_agents_observations[agent].get("observation"),
                        dm_env_timestep.observation,
                    )

                    np.testing.assert_array_equal(
                        test_agents_observations[agent].get("action_mask"),
                        dm_env_timestep.legal_actions,
                    )
                    assert (
                        bool(dm_env_timestep.terminal) is False
                    ), "Failed to set terminal."

    # Test we can take a action and it updates observations
    def test_step_0_valid_when_env_not_done(
        self, env_spec: EnvSpec, helpers: Helpers
    ) -> None:
        wrapped_env, _ = helpers.get_wrapped_env(env_spec)

        # Seed environment since we are sampling actions.
        # We need to seed env and action space.
        random_seed = 42
        wrapped_env.seed(random_seed)
        helpers.seed_action_space(wrapped_env, random_seed)

        #  Get agent names from env
        agents = wrapped_env.agents

        timestep = wrapped_env.reset()
        if type(timestep) == tuple:
            initial_dm_env_timestep, env_extras = timestep
        else:
            initial_dm_env_timestep = timestep
        # Parallel env_types
        if env_spec.env_type == EnvType.Parallel:
            test_agents_actions = {
                agent: wrapped_env.action_spaces[agent].sample() for agent in agents
            }
            curr_dm_timestep = wrapped_env.step(test_agents_actions)

            for agent in wrapped_env.agents:
                assert not np.array_equal(
                    initial_dm_env_timestep.observation[agent].observation,
                    curr_dm_timestep.observation[agent].observation,
                ), "Failed to update observations."

        # Sequential env_types
        elif env_spec.env_type == EnvType.Sequential:
            curr_dm_timestep = initial_dm_env_timestep
            for agent in agents:
                if env_spec.env_source == EnvSource.OpenSpiel:
                    test_agent_actions = np.random.choice(
                        np.where(curr_dm_timestep.observation.legal_actions)[0]
                    )
                else:
                    test_agent_actions = wrapped_env.action_spaces[agent].sample()

                curr_dm_timestep = wrapped_env.step(test_agent_actions)

                assert not np.array_equal(
                    initial_dm_env_timestep.observation.observation,
                    curr_dm_timestep.observation.observation,
                ), "Failed to update observations."

        assert (
            wrapped_env._reset_next_step is False
        ), "Failed to set _reset_next_step correctly."
        assert curr_dm_timestep.reward is not None, "Failed to set rewards."
        assert (
            curr_dm_timestep.step_type is dm_env.StepType.MID
        ), "Failed to update step type."

    # Test we only step in our env once.
    def test_step_1_valid_when_env_not_done(
        self, env_spec: EnvSpec, helpers: Helpers
    ) -> None:
        wrapped_env, _ = helpers.get_wrapped_env(env_spec)

        # Seed environment since we are sampling actions.
        # We need to seed env and action space.
        random_seed = 42
        wrapped_env.seed(random_seed)
        helpers.seed_action_space(wrapped_env, random_seed)

        #  Get agent names from env
        agents = wrapped_env.agents

        # Parallel env_types
        if env_spec.env_type == EnvType.Parallel:
            test_agents_actions = {
                agent: wrapped_env.action_spaces[agent].sample() for agent in agents
            }
            with patch.object(wrapped_env, "step") as parallel_step:
                parallel_step.return_value = None, None, None, None
                _ = wrapped_env.step(test_agents_actions)
                parallel_step.assert_called_once_with(test_agents_actions)

        # Sequential env_types
        elif env_spec.env_type == EnvType.Sequential:
            for agent in agents:
                with patch.object(wrapped_env, "step") as seq_step:
                    seq_step.return_value = None
                    test_agent_action = wrapped_env.action_spaces[agent].sample()
                    _ = wrapped_env.step(test_agent_action)
                    seq_step.assert_called_once_with(test_agent_action)

    # Test if all agents are done, env is set to done
    def test_step_2_invalid_when_env_done(
        self, env_spec: EnvSpec, helpers: Helpers, monkeypatch: MonkeyPatch
    ) -> None:
        wrapped_env, _ = helpers.get_wrapped_env(env_spec)

        if env_spec.env_source == EnvSource.OpenSpiel:
            pytest.skip("Open Spiel does not use the .last() method")

        # Seed environment since we are sampling actions.
        # We need to seed env and action space.
        random_seed = 42
        wrapped_env.seed(random_seed)
        helpers.seed_action_space(wrapped_env, random_seed)

        #  Get agent names from env
        _ = wrapped_env.reset()
        agents = wrapped_env.agents

        # Parallel env_types
        if env_spec.env_type == EnvType.Parallel:
            test_agents_actions = {
                agent: wrapped_env.action_spaces[agent].sample() for agent in agents
            }

            monkeypatch.setattr(wrapped_env, "env_done", helpers.mock_done)

            curr_dm_timestep = wrapped_env.step(test_agents_actions)

            helpers.assert_env_reset(wrapped_env, curr_dm_timestep, env_spec)

        # Sequential env_types
        # TODO (Kale-ab): Make this part below less reliant on PZ.
        elif env_spec.env_type == EnvType.Sequential:
            n_agents = wrapped_env.num_agents

            # Mock functions to act like PZ environment is done
            def mock_environment_last() -> Any:
                observe = wrapped_env.observation_spaces[agent].sample()
                reward = 0.0
                done = True
                info: Dict = {}
                return observe, reward, done, info

            def mock_step(action: types.Action) -> None:
                return

            # Mocks certain functions - if functions don't exist, error is not thrown.
            monkeypatch.setattr(
                wrapped_env._environment, "last", mock_environment_last, raising=False
            )
            monkeypatch.setattr(
                wrapped_env._environment, "step", mock_step, raising=False
            )

            for index, (agent) in enumerate(wrapped_env.agent_iter(n_agents)):
                test_agent_actions = wrapped_env.action_spaces[agent].sample()

                # Mock whole env being done when you reach final agent
                if index == n_agents - 1:
                    monkeypatch.setattr(
                        wrapped_env,
                        "env_done",
                        helpers.mock_done,
                    )

                # Mock update has occurred in step
                monkeypatch.setattr(
                    wrapped_env._environment, "_has_updated", True, raising=False
                )

                curr_dm_timestep = wrapped_env.step(test_agent_actions)

                # Check each agent is on last step
                assert (
                    curr_dm_timestep.step_type is dm_env.StepType.LAST
                ), "Failed to update step type."

            helpers.assert_env_reset(wrapped_env, curr_dm_timestep, env_spec)

        assert (
            wrapped_env._reset_next_step is True
        ), "Failed to set _reset_next_step correctly."
        assert (
            curr_dm_timestep.step_type is dm_env.StepType.LAST
        ), "Failed to update step type."
