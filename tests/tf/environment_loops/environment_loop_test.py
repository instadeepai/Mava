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

import pytest

from tests.tf.conftest import EnvSpec, EnvType, Helpers, MockedEnvironments
from tests.tf.mocks import MockedSystem


@pytest.mark.parametrize(
    "env_spec",
    [
        # Mocked environments
        EnvSpec(MockedEnvironments.Mocked_Dicrete, EnvType.Parallel),
        EnvSpec(MockedEnvironments.Mocked_Dicrete, EnvType.Sequential),
        EnvSpec(MockedEnvironments.Mocked_Continous, EnvType.Parallel),
        EnvSpec(MockedEnvironments.Mocked_Continous, EnvType.Sequential),
        # Real Environments
        EnvSpec("pettingzoo.mpe.simple_spread_v2", EnvType.Parallel),
        EnvSpec("pettingzoo.mpe.simple_spread_v2", EnvType.Sequential),
        EnvSpec("pettingzoo.sisl.multiwalker_v8", EnvType.Parallel),
        EnvSpec("pettingzoo.sisl.multiwalker_v8", EnvType.Sequential),
    ],
)
class TestEnvironmentLoop:
    # Test that we can load a env loop and that it contains
    #   an env, executor, counter, logger and should_update.
    def test_initialize_env_loop(self, env_spec: EnvSpec, helpers: Helpers) -> None:
        wrapped_env, specs = helpers.get_wrapped_env(env_spec)
        env_loop_func = helpers.get_env_loop(env_spec)

        env_loop = env_loop_func(
            wrapped_env,
            MockedSystem(specs),
        )

        props_which_should_not_be_none = [
            env_loop,
            env_loop._environment,
            env_loop._executor,
            env_loop._counter,
            env_loop._logger,
            env_loop._should_update,
        ]
        assert helpers.verify_all_props_not_none(
            props_which_should_not_be_none
        ), "Failed to initialize env loop."

    # Test that we can run an episode and that the episode returns valid data.
    def test_valid_episode(self, env_spec: EnvSpec, helpers: Helpers) -> None:
        wrapped_env, specs = helpers.get_wrapped_env(env_spec)
        env_loop_func = helpers.get_env_loop(env_spec)

        env_loop = env_loop_func(
            wrapped_env,
            MockedSystem(specs),
        )

        result = env_loop.run_episode()

        helpers.assert_valid_episode(result)
