import dm_env
import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch

from tests.conftest import EnvSpec, EnvType, Helpers

"""
TestEnvWrapper is a general purpose test class that runs tests for environment wrappers.
This is meant to flexibily test various environments wrappers.

    It is parametrize by an EnvSpec object:
        env_name: [name of env]
        env_type: [EnvType.Parallel/EnvType.Sequential]

    For new environments - you might need to update the Helpers class in conftest.py.
"""


@pytest.mark.parametrize(
    "env_spec",
    [
        EnvSpec("pettingzoo.mpe.simple_spread_v2", EnvType.Parallel),
        EnvSpec("pettingzoo.mpe.simple_spread_v2", EnvType.Sequential),
        EnvSpec("pettingzoo.sisl.multiwalker_v6", EnvType.Parallel),
        EnvSpec("pettingzoo.sisl.multiwalker_v6", EnvType.Sequential),
    ],
)
class TestEnvWrapper:
    # Test that we can load a env module and that it contains agents,
    #   action_spaces and observation_spaces.
    def test_loadmodule(self, env_spec: EnvSpec, helpers: Helpers) -> None:
        env, _ = helpers.get_env(env_spec)
        props_which_should_not_be_none = [
            env,
            env.agents,
            env.action_spaces,
            env.observation_spaces,
        ]
        assert helpers.verify_all_props_not_none(
            props_which_should_not_be_none
        ), "Failed to load module"

    #  Test initialization of env wrapper, which should have
    #   a nested environment, an observation and action space for each agent.
    def testwrapper_initialization(self, env_spec: EnvSpec, helpers: Helpers) -> None:
        env, num_agents = helpers.get_env(env_spec)
        wrapper_func = helpers.get_wrapper(env_spec)
        wrapped_env = wrapper_func(env)
        props_which_should_not_be_none = [
            wrapped_env,
            wrapped_env.environment,
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

    # Test of reset of wrapper and that dm_env_timestep has basic props.
    def testwrapper_env_reset(self, env_spec: EnvSpec, helpers: Helpers) -> None:
        env, num_agents = helpers.get_env(env_spec)
        wrapper_func = helpers.get_wrapper(env_spec)
        wrapped_env = wrapper_func(env)

        dm_env_timestep = wrapped_env.reset()
        props_which_should_not_be_none = [dm_env_timestep, dm_env_timestep.observation]

        assert helpers.verify_all_props_not_none(
            props_which_should_not_be_none
        ), "Failed to ini dm_env_timestep."
        assert (
            dm_env_timestep.step_type == dm_env.StepType.FIRST
        ), "Failed to have correct StepType."
        assert (
            len(dm_env_timestep.observation) == num_agents
        ), "Failed to generate observation for all agents."
        assert wrapped_env._reset_next_step is False, "_reset_next_step not set."

        assert dm_env_timestep.reward is None, "Failed to reset reward."
        assert dm_env_timestep.discount is None, "Failed to reset discount."

    # Test that observations from petting zoo get converted to
    #   dm observations correctly. This only runs
    #   if wrapper has a _convert_observations or _convert_observation functions.
    def test_covert_env_to_dm_env_0_no_action_mask(
        self, env_spec: EnvSpec, helpers: Helpers
    ) -> None:
        env, num_agents = helpers.get_env(env_spec)
        wrapper_func = helpers.get_wrapper(env_spec)
        wrapped_env = wrapper_func(env)

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

    # Test that observations **with actions masked** from petting zoo get
    #   converted to dm observations correctly. This only runs
    #   if wrapper has a _convert_observations or _convert_observation functions.
    def test_covert_env_to_dm_env_1_with_action_mask(
        self, env_spec: EnvSpec, helpers: Helpers
    ) -> None:
        env, num_agents = helpers.get_env(env_spec)
        wrapper_func = helpers.get_wrapper(env_spec)
        wrapped_env = wrapper_func(env)

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
        env, num_agents = helpers.get_env(env_spec)
        wrapper_func = helpers.get_wrapper(env_spec)
        wrapped_env = wrapper_func(env)

        # Seed environment since we are sampling actions.
        # We need to seed env and action space.
        random_seed = 42
        wrapped_env.seed(random_seed)
        helpers.seed_action_space(wrapped_env, random_seed)

        #  Get agent names from env
        agents = wrapped_env.agents

        initial_dm_env_timestep = wrapped_env.reset()
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
            for agent in agents:
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

    # Test if all agents are done, env is set to done
    def test_step_1_invalid_when_env_done(
        self, env_spec: EnvSpec, helpers: Helpers, monkeypatch: MonkeyPatch
    ) -> None:
        env, num_agents = helpers.get_env(env_spec)
        wrapper_func = helpers.get_wrapper(env_spec)
        wrapped_env = wrapper_func(env)

        # Seed environment since we are sampling actions.
        # We need to seed env and action space.
        random_seed = 42
        wrapped_env.seed(random_seed)
        helpers.seed_action_space(wrapped_env, random_seed)

        #  Get agent names from env
        agents = wrapped_env.agents

        _ = wrapped_env.reset()

        # Parallel env_types
        if env_spec.env_type == EnvType.Parallel:
            test_agents_actions = {
                agent: wrapped_env.action_spaces[agent].sample() for agent in agents
            }

            expected_reward = {agent: 0 for agent in wrapped_env.agents}
            # Mock being done
            monkeypatch.setattr(
                wrapped_env._environment.aec_env,
                "agents",
                [],
            )

            curr_dm_timestep = wrapped_env.step(test_agents_actions)

            assert (
                curr_dm_timestep.reward == expected_reward
            ), "Failed to correctly set reward. "

        # Sequential env_types
        elif env_spec.env_type == EnvType.Sequential:

            for index, agent in enumerate(agents):
                test_agent_actions = wrapped_env.action_spaces[agent].sample()

                # Mock being done when you reach final agent
                if index == len(agents) - 1:
                    monkeypatch.setattr(
                        wrapped_env._environment,
                        "dones",
                        {agent: True for agent in wrapped_env.agents},
                    )
                    monkeypatch.setattr(
                        wrapped_env._environment,
                        "agents",
                        [],
                    )

                curr_dm_timestep = wrapped_env.step(test_agent_actions)

            assert curr_dm_timestep.reward == 0, "Failed to correctly set reward. "

        assert (
            wrapped_env._reset_next_step is True
        ), "Failed to set _reset_next_step correctly."
        assert (
            curr_dm_timestep.step_type is dm_env.StepType.LAST
        ), "Failed to update step type."
