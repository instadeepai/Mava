import pytest
from tests.conftest import EnvSpec, EnvType, Helpers

# from mocks import MockedExecutor

from tests.mocks import MockedSystem, Environment, get_mocked_env_spec

# from acme.testing.fakes import Environment
import dm_env

from acme import specs


import numpy as np

# import pprint
from pprint import pprint


@pytest.mark.parametrize(
    "env_spec",
    [
        EnvSpec("pettingzoo.mpe.simple_spread_v2", EnvType.Parallel),
        # EnvSpec("pettingzoo.mpe.simple_spread_v2", EnvType.Sequential),
        # EnvSpec("pettingzoo.sisl.multiwalker_v6", EnvType.Parallel),
        # EnvSpec("pettingzoo.sisl.multiwalker_v6", EnvType.Sequential),
    ],
)
class TestEnvironmentLoop:
    # Test that we can load a env loop and that it contains
    #   an env, executor, counter, logger and should_update.
    # def test_initialize_env_loop(self, env_spec: EnvSpec, helpers: Helpers) -> None:
    #     env, _ = helpers.get_env(env_spec)
    #     env_loop_func = helpers.get_env_loop(env_spec)

    #     wrapper_func = helpers.get_wrapper(env_spec)
    #     wrapped_env = wrapper_func(env)

    #     env_loop = env_loop_func(
    #         env, MockedExecutor(get_mocked_env_spec(wrapped_env), env_spec.env_type)
    #     )

    #     props_which_should_not_be_none = [
    #         env_loop,
    #         env_loop._environment,
    #         env_loop._executor,
    #         env_loop._counter,
    #         env_loop._logger,
    #         env_loop._should_update,
    #     ]
    #     assert helpers.verify_all_props_not_none(
    #         props_which_should_not_be_none
    #     ), "Failed to initialize env loop."

    def test_get_actions(self, env_spec: EnvSpec, helpers: Helpers) -> None:
        print(env_spec.env_type)
        env, _ = helpers.get_env(env_spec)
        env_loop_func = helpers.get_env_loop(env_spec)

        wrapper_func = helpers.get_wrapper(env_spec)
        wrapped_env = wrapper_func(env)

        env_loop = env_loop_func(
            wrapped_env,
            MockedSystem(get_mocked_env_spec(wrapped_env)._specs, env_spec.env_type),
        )

        print(env_loop.run_episode())
