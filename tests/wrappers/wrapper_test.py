import dm_env
import numpy as np
import pytest

from tests.conftest import EnvSpec, EnvType

"""
TestEnvWrapper is a general purpose test class that runs tests for environment wrappers.
This is meant to flexibily test various environments wrappers.

    It is parametrize by an EnvSpec object:
        env_name: [name of env]
        env_type: [EnvType.Parallel/EnvType.Sequential]
"""


@pytest.mark.parametrize(
    "env_spec",
    [
        EnvSpec("pettingzoo.mpe.simple_spread_v2", EnvType.Parallel),
        EnvSpec("pettingzoo.mpe.simple_spread_v2", EnvType.Sequential),
        # TODO Uncomment once we have working legal_actions for continuous envs
        # EnvSpec("pettingzoo.sisl.multiwalker_v6", EnvType.Parallel),
        # EnvSpec("pettingzoo.sisl.multiwalker_v6", EnvType.Sequential),
    ],
)
class TestEnvWrapper:
    # Test that we can load a env module and that it contains agents,
    #   action_spaces and observation_spaces.
    def test_loadmodule(self, env_spec: EnvSpec, helpers: pytest.fixture) -> None:
        env, _ = helpers.retrieve_env(env_spec)
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
    def testwrapper_initialization(
        self, env_spec: EnvSpec, helpers: pytest.fixture
    ) -> None:
        env, num_agents = helpers.retrieve_env(env_spec)
        wrapper_func = helpers.retrieve_wrapper(env_spec)
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
    def testwrapper_env_reset(self, env_spec: EnvSpec, helpers: pytest.fixture) -> None:
        env, num_agents = helpers.retrieve_env(env_spec)
        wrapper_func = helpers.retrieve_wrapper(env_spec)
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
    def test_covertenv_to_dm_ev_0_no_action_mask(
        self, env_spec: EnvSpec, helpers: pytest.fixture
    ) -> None:
        env, num_agents = helpers.retrieve_env(env_spec)
        wrapper_func = helpers.retrieve_wrapper(env_spec)
        wrapped_env = wrapper_func(env)

        # Does the wrapper have the functions we want to test
        if hasattr(wrapped_env, "_convert_observations") or hasattr(
            wrapped_env, "_convert_observation"
        ):
            agents = ["agent_0", "agent_1", "agent_2"]
            test_agents_observations = {agent: np.random.rand(5, 5) for agent in agents}

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
    def test_covertenv_to_dm_ev_1_with_action_mask(
        self, env_spec: EnvSpec, helpers: pytest.fixture
    ) -> None:
        env, num_agents = helpers.retrieve_env(env_spec)
        wrapper_func = helpers.retrieve_wrapper(env_spec)
        wrapped_env = wrapper_func(env)

        # Does the wrapper have the functions we want to test
        if hasattr(wrapped_env, "_convert_observations") or hasattr(
            wrapped_env, "_convert_observation"
        ):
            agents = ["agent_0", "agent_1", "agent_2"]
            test_agents_observations = {}
            for agent in agents:
                test_agents_observations[agent] = {
                    "observation": np.random.rand(5, 5),
                    "action_mask": np.random.randint(2, size=5),
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
