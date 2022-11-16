# python3
# Copyright 2022 InstaDeep Ltd. All rights reserved.
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

"""Environment loop unit test"""

import pytest

from tests.conftest import EnvSpec, Helpers, MockedEnvironments
from tests.mocks import MockedSystem


@pytest.mark.parametrize(
    "env_spec",
    [
        # Mocked environments
        EnvSpec(MockedEnvironments.Mocked_Dicrete),
        EnvSpec(MockedEnvironments.Mocked_Dicrete),
        EnvSpec(MockedEnvironments.Mocked_Continous),
        EnvSpec(MockedEnvironments.Mocked_Continous),
        # Real Environments
        EnvSpec("pettingzoo.mpe.simple_spread_v2"),
        EnvSpec("pettingzoo.mpe.simple_spread_v2"),
        EnvSpec("pettingzoo.sisl.multiwalker_v8"),
        EnvSpec("pettingzoo.sisl.multiwalker_v8"),
    ],
)
class TestEnvironmentLoop:
    """Test that we can load a env loop"""

    def test_initialize_env_loop(self, env_spec: EnvSpec, helpers: Helpers) -> None:
        """Test initialization of the environmnet loop"""
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

    def test_valid_episode(self, env_spec: EnvSpec, helpers: Helpers) -> None:
        """Test that we can run an episode and that the episode returns valid data."""
        wrapped_env, specs = helpers.get_wrapped_env(env_spec)
        env_loop_func = helpers.get_env_loop(env_spec)

        env_loop = env_loop_func(
            wrapped_env,
            MockedSystem(specs),
        )

        result = env_loop.run_episode()

        helpers.assert_valid_episode(result)
