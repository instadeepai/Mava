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

import numpy as np
import pytest

from mava.utils.environments import debugging_utils, pettingzoo_utils, smac_utils
from mava.utils.environments.flatland_utils import flatland_env_factory

try:
    from flatland.envs.observations import TreeObsForRailEnv
    from flatland.envs.rail_generators import sparse_rail_generator
    from flatland.envs.schedule_generators import sparse_schedule_generator
except ModuleNotFoundError:
    pass

rail_gen_cfg: Dict = {
    "max_num_cities": 4,
    "max_rails_between_cities": 2,
    "max_rails_in_city": 3,
    "grid_mode": True,
}


@pytest.mark.parametrize(
    "env",
    [
        # env_factory, env_params
        (debugging_utils.make_environment, {}),
        (pettingzoo_utils.make_environment, {}),
        (
            flatland_env_factory,
            {
                "env_config": {
                    "number_of_agents": 2,
                    "width": 25,
                    "height": 25,
                    "rail_generator": sparse_rail_generator(**rail_gen_cfg),
                    "schedule_generator": sparse_schedule_generator(),
                    "obs_builder_object": TreeObsForRailEnv(max_depth=2),
                }
            },
        ),
        # This requires install sc2 and setting SC2Path,
        # we don't do this for the tests on CI, only when testing locally,
        # so commented out.
        # (smac_utils.make_environment,{})
    ],
)
class TestEnvUtils:
    def test_env_reproducibility_0_seed_same_observation(self, env: Any) -> None:
        """Test with seeds, obs are the same.

        Args:
            env_factory (Any): env factory.
        """
        test_seed = 42
        env_factory, env_params = env

        wrapped_env = env_factory(random_seed=test_seed, **env_params)
        reset_result = wrapped_env.reset()
        if type(reset_result) is tuple:
            timestep, _ = reset_result
        else:
            timestep = reset_result

        wrapped_env2 = env_factory(random_seed=test_seed, **env_params)
        reset_result2 = wrapped_env2.reset()
        if type(reset_result2) is tuple:
            timestep2, _ = reset_result2
        else:
            timestep2 = reset_result2

        for agent in wrapped_env.agents:
            np.testing.assert_array_equal(
                timestep.observation[agent].observation,
                timestep2.observation[agent].observation,
            )

    def test_env_reproducibility_1_no_seed_different_observation(
        self, env: Any
    ) -> None:
        """Test with no seeds, obs are different.

        Args:
            env_factory (Any): env factory.
        """
        env_factory, env_params = env

        # This test doesn't work with flatland and SC2, since FL uses
        # a default seed (1) and SC2 (5), even when a seed is not provided.
        if (
            env_factory == flatland_env_factory
            or env_factory == smac_utils.make_environment
        ):
            pytest.skip("Skipping no seed test for flatland and SC2.")

        wrapped_env = env_factory(**env_params)
        reset_result = wrapped_env.reset()
        if type(reset_result) is tuple:
            timestep, _ = reset_result
        else:
            timestep = reset_result

        wrapped_env2 = env_factory(**env_params)
        reset_result2 = wrapped_env2.reset()
        if type(reset_result2) is tuple:
            timestep2, _ = reset_result2
        else:
            timestep2 = reset_result2

        for agent in wrapped_env.agents:
            assert not np.array_equal(
                timestep.observation[agent].observation,
                timestep2.observation[agent].observation,
            )

    def test_env_reproducibility_1_different_seed_different_observation(
        self, env: Any
    ) -> None:
        """Test with different seeds, obs are different.

        Args:
            env_factory (Any): env factory.
        """
        env_factory, env_params = env
        test_seed1 = 42
        test_seed2 = 43

        # This test doesn't work with flatland, since FL seeds
        # at ini for SparseRailGen .
        if env_factory == flatland_env_factory:
            pytest.skip("Skipping diff seed test for flatland.")

        wrapped_env = env_factory(random_seed=test_seed1, **env_params)
        reset_result = wrapped_env.reset()
        if type(reset_result) is tuple:
            timestep, _ = reset_result
        else:
            timestep = reset_result

        wrapped_env2 = env_factory(random_seed=test_seed2, **env_params)
        reset_result2 = wrapped_env2.reset()
        if type(reset_result2) is tuple:
            timestep2, _ = reset_result2
        else:
            timestep2 = reset_result2

        # The random seed used in SC2 doesn't necessarily result
        # in diff observations. According to doc, the seed results in slight randomness.
        # Therefore for SC2, we test the seed is set correctly,
        # not that the obs are different.
        if env_factory == smac_utils.make_environment:
            assert wrapped_env._environment._seed == test_seed1
            assert wrapped_env2._environment._seed == test_seed2
            return

        for agent in wrapped_env.agents:
            assert not np.array_equal(
                timestep.observation[agent].observation,
                timestep2.observation[agent].observation,
            )
