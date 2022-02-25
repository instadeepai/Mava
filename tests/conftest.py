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

import importlib
import typing
from typing import Any, Dict, List, Tuple, Union

import acme
import dm_env
import numpy as np
import numpy.testing as npt
import pytest

try:
    from mava.utils.environments import flatland_utils
    from mava.wrappers.flatland import FlatlandEnvWrapper
except ModuleNotFoundError:
    pass

try:
    from pettingzoo.utils.env import AECEnv, ParallelEnv
except ModuleNotFoundError:
    pass

from mava import specs as mava_specs
from mava.environment_loop import ParallelEnvironmentLoop, SequentialEnvironmentLoop
from mava.types import Observation, Reward

try:
    from mava.utils.environments.open_spiel_utils import load_open_spiel_env
    from mava.wrappers.open_spiel import OpenSpielSequentialWrapper
except ImportError:
    pass
from mava.utils.wrapper_utils import convert_np_type
from mava.wrappers.pettingzoo import (
    PettingZooAECEnvWrapper,
    PettingZooParallelEnvWrapper,
)
from tests.enums import EnvSource, EnvSpec, EnvType, MockedEnvironments
from tests.mocks import (
    ParallelMAContinuousEnvironment,
    ParallelMADiscreteEnvironment,
    SequentialMAContinuousEnvironment,
    SequentialMADiscreteEnvironment,
)

# flatland environment config
flatland_env_config = {
    "n_agents": 3,
    "x_dim": 30,
    "y_dim": 30,
    "n_cities": 2,
    "max_rails_between_cities": 2,
    "max_rails_in_city": 3,
    "seed": 0,
    "malfunction_rate": 1 / 200,
    "malfunction_min_duration": 20,
    "malfunction_max_duration": 50,
    "observation_max_path_depth": 30,
    "observation_tree_depth": 2,
}

"""
Helpers contains re-usable test functions.
"""

# TODO(Kale-ab): Better structure helper funcs


class Helpers:
    @staticmethod
    def verify_all_props_not_none(props_which_should_not_be_none: list) -> bool:
        """Check all props are not none

        Args:
            props_which_should_not_be_none : vars which should have a value.

        Returns:
            bool indicating if vars are not none.
        """
        return all(prop is not None for prop in props_which_should_not_be_none)

    @staticmethod
    def get_env(env_spec: EnvSpec) -> Union[AECEnv, ParallelEnv]:
        """Return an env based on an env spec.

        Args:
            env_spec : decription of env.

        Raises:
            Exception: No appropriate env found.

        Returns:
            an envrionment.
        """
        env = None
        if env_spec.env_source == EnvSource.PettingZoo:
            mod = importlib.import_module(env_spec.env_name)
            if env_spec.env_type == EnvType.Parallel:
                env = mod.parallel_env()  # type:ignore
            elif env_spec.env_type == EnvType.Sequential:
                env = mod.env()  # type:ignore
        elif env_spec.env_source == EnvSource.Flatland:
            env = flatland_utils.make_environment(**flatland_env_config)  # type:ignore
        elif env_spec.env_source == EnvSource.OpenSpiel:
            env = load_open_spiel_env(env_spec.env_name)
        else:
            raise Exception("Env_spec is not valid.")
        env.reset()  # type:ignore
        return env

    @staticmethod
    def get_wrapper_function(
        env_spec: EnvSpec,
    ) -> dm_env.Environment:
        """Returns a wrapper function.

        Args:
            env_spec : decription of env.

        Raises:
            Exception: No env wrapper found.

        Returns:
             an envrionment wrapper.
        """
        wrapper: dm_env.Environment = None
        if env_spec.env_source == EnvSource.PettingZoo:
            if env_spec.env_type == EnvType.Parallel:
                wrapper = PettingZooParallelEnvWrapper
            elif env_spec.env_type == EnvType.Sequential:
                wrapper = PettingZooAECEnvWrapper
        elif env_spec.env_source == EnvSource.Flatland:
            wrapper = FlatlandEnvWrapper
        elif env_spec.env_source == EnvSource.OpenSpiel:
            wrapper = OpenSpielSequentialWrapper
        else:
            raise Exception("Env_spec is not valid.")
        return wrapper

    @staticmethod
    def get_env_loop(
        env_spec: EnvSpec,
    ) -> acme.core.Worker:
        """Returns an env loop.

        Args:
            env_spec : decription of env.

        Raises:
            Exception: Unable to find env loop.

        Returns:
            env loop.
        """
        env_loop = None
        if env_spec.env_type == EnvType.Parallel:
            env_loop = ParallelEnvironmentLoop
        elif env_spec.env_type == EnvType.Sequential:
            env_loop = SequentialEnvironmentLoop
        else:
            raise Exception("Env_spec is not valid.")
        return env_loop

    @staticmethod
    def get_mocked_env(
        env_spec: EnvSpec,
    ) -> Union[
        ParallelMADiscreteEnvironment,
        SequentialMADiscreteEnvironment,
        ParallelMAContinuousEnvironment,
        SequentialMAContinuousEnvironment,
    ]:
        """Function that retrieves a mocked env.

        Args:
            env_spec : decription of env.

        Raises:
            Exception: no valid env found.

        Returns:
            a mocked environment.
        """
        env_name = env_spec.env_name

        if env_name is MockedEnvironments.Mocked_Dicrete:
            if env_spec.env_type == EnvType.Parallel:
                env = ParallelMADiscreteEnvironment(
                    num_actions=18,
                    num_observations=2,
                    obs_shape=(84, 84, 4),
                    obs_dtype=np.float32,
                    episode_length=10,
                )
            elif env_spec.env_type == EnvType.Sequential:
                env = SequentialMADiscreteEnvironment(
                    num_actions=18,
                    num_observations=2,
                    obs_shape=(84, 84, 4),
                    obs_dtype=np.float32,
                    episode_length=10,
                )
        elif env_name is MockedEnvironments.Mocked_Continous:
            if env_spec.env_type == EnvType.Parallel:
                env = ParallelMAContinuousEnvironment(
                    action_dim=2,
                    observation_dim=2,
                    bounded=True,
                    episode_length=10,
                )
            elif env_spec.env_type == EnvType.Sequential:
                env = SequentialMAContinuousEnvironment(
                    action_dim=2,
                    observation_dim=2,
                    bounded=True,
                    episode_length=10,
                )

        if env is None:
            raise Exception("Env_spec is not valid.")
        return env

    @staticmethod
    def is_mocked_env(
        env_name: str,
    ) -> bool:
        """Returns bool indicating if env is mocked or not.

        Args:
            env_spec : decription of env.

        Returns:
            bool indicating if env is mocked or not.
        """
        mock = False
        if (
            env_name is MockedEnvironments.Mocked_Continous
            or env_name is MockedEnvironments.Mocked_Dicrete
        ):
            mock = True
        return mock

    @staticmethod
    def get_wrapped_env(
        env_spec: EnvSpec, **kwargs: Any
    ) -> Tuple[dm_env.Environment, acme.specs.EnvironmentSpec]:
        """Returns a wrapped env and specs.

        Args:
            env_spec : decription of env.

        Returns:
            a wrapped env and specs.
        """
        specs = None
        if Helpers.is_mocked_env(env_spec.env_name):
            wrapped_env = Helpers.get_mocked_env(env_spec)
            specs = wrapped_env._specs
        else:
            env = Helpers.get_env(env_spec)
            wrapper_func = Helpers.get_wrapper_function(env_spec)
            wrapped_env = wrapper_func(env, **kwargs)
            specs = Helpers.get_pz_env_spec(wrapped_env)._specs
        return wrapped_env, specs

    #
    @staticmethod
    def get_pz_env_spec(environment: dm_env.Environment) -> dm_env.Environment:
        """Returns a petting zoo environment spec.

        Args:
            environment : an env.

        Returns:
            a petting zoo environment spec.
        """
        return mava_specs.MAEnvironmentSpec(environment)

    @staticmethod
    def seed_action_space(
        env_wrapper: Union[PettingZooAECEnvWrapper, PettingZooParallelEnvWrapper],
        random_seed: int,
    ) -> None:
        """Seeds action space.

        Args:
            env_wrapper : an env wrapper.
            random_seed : random seed to be used.
        """
        [
            env_wrapper.action_spaces[agent].seed(random_seed)
            for agent in env_wrapper.agents
        ]

    @staticmethod
    def compare_dicts(dictA: Dict, dictB: Dict) -> bool:
        """Function that check if two dicts are equal.

        Args:
            dictA : dict A.
            dictB : dict B.

        Returns:
            bool indicating if dicts are equal or not.
        """
        typesA = [type(k) for k in dictA.values()]
        typesB = [type(k) for k in dictB.values()]

        return (dictA == dictB) and (typesA == typesB)

    @staticmethod
    def assert_valid_episode(episode_result: Dict) -> None:
        """Function that checks if a valid episode was run.

        Args:
            episode_result : result dict from an episode.
        """
        assert (
            episode_result["episode_length"] > 0
            and episode_result["mean_episode_return"] is not None
            and episode_result["steps_per_second"] is not None
            and episode_result["episodes"] == 1
            and episode_result["steps"] > 0
        )

    @staticmethod
    def assert_env_reset(
        wrapped_env: dm_env.Environment,
        dm_env_timestep: dm_env.TimeStep,
        env_spec: EnvSpec,
    ) -> None:
        """Assert env are reset correctly.

        Args:
            wrapped_env : wrapped env.
            dm_env_timestep : timestep.
            env_spec : env spec.
        """
        if env_spec.env_type == EnvType.Parallel:
            rewards_spec = wrapped_env.reward_spec()
            expected_rewards = {
                agent: convert_np_type(rewards_spec[agent].dtype, 0)
                for agent in wrapped_env.agents
            }

            discount_spec = wrapped_env.discount_spec()
            expected_discounts = {
                agent: convert_np_type(rewards_spec[agent].dtype, 1)
                for agent in wrapped_env.agents
            }

            Helpers.compare_dicts(
                dm_env_timestep.reward,
                expected_rewards,
            ), "Failed to reset reward."
            Helpers.compare_dicts(
                dm_env_timestep.discount,
                expected_discounts,
            ), "Failed to reset discount."

        elif env_spec.env_type == EnvType.Sequential:
            for agent in wrapped_env.agents:
                rewards_spec = wrapped_env.reward_spec()
                expected_reward = convert_np_type(rewards_spec[agent].dtype, 0)

                discount_spec = wrapped_env.discount_spec()
                expected_discount = convert_np_type(discount_spec[agent].dtype, 1)

                assert dm_env_timestep.reward == expected_reward and type(
                    dm_env_timestep.reward
                ) == type(expected_reward), "Failed to reset reward."
                assert dm_env_timestep.discount == expected_discount and type(
                    dm_env_timestep.discount
                ) == type(expected_discount), "Failed to reset discount."

    @staticmethod
    @typing.no_type_check
    # TODO(Kale-ab) Sort out typing issues.
    def verify_observations_are_normalized(
        observations: Observation,
        agents: List,
        env_spec: EnvSpec,
        min: int = 0,
        max: int = 1,
    ) -> None:
        """Verify observations are normalized.

        Args:
            observations : env obs.
            agents : env agents.
            env_spec : env spec.
            min : min for normalization.
            max : max for normalization.
        """
        if env_spec.env_type == EnvType.Parallel:
            for agent in agents:
                assert (
                    observations[agent].observation.min() >= min
                    and observations[agent].observation.max() <= max
                ), "Failed to normalize observations."

        elif env_spec.env_type == EnvType.Sequential:
            assert (
                observations.observation.min() >= min
                and observations.observation.max() <= max
            ), "Failed to normalize observations."

    @staticmethod
    @typing.no_type_check
    # TODO(Kale-ab) Sort out typing issues.
    def verify_reward_is_normalized(
        rewards: Reward, agents: List, env_spec: EnvSpec, min: int = 0, max: int = 1
    ) -> None:
        """Verify reward is normalized.

        Args:
            rewards : rewards.
            agents : env agents.
            env_spec : env spec.
            min : min for normalization.
            max : max for normalization.
        """
        if env_spec.env_type == EnvType.Parallel:
            for agent in agents:
                assert (
                    rewards[agent] >= min and rewards[agent] <= max
                ), "Failed to normalize reward."

        elif env_spec.env_type == EnvType.Sequential:
            assert rewards >= min and rewards <= max, "Failed to normalize reward."

    @staticmethod
    def verify_observations_are_standardized(
        observations: Observation, agents: List, env_spec: EnvSpec
    ) -> None:
        """Verify obs are standardized.

        Args:
            observations : env observations.
            agents : env agents.
            env_spec : env spec.
        """
        if env_spec.env_type == EnvType.Parallel:
            for agent in agents:
                npt.assert_almost_equal(
                    observations[agent].observation.mean(), 0, decimal=2  # type: ignore
                )
                npt.assert_almost_equal(
                    observations[agent].observation.std(), 1, decimal=2  # type: ignore
                )

        elif env_spec.env_type == EnvType.Sequential:
            npt.assert_almost_equal(
                observations.observation.mean(), 0, decimal=2  # type: ignore
            )
            npt.assert_almost_equal(
                observations.observation.std(), 1, decimal=2  # type: ignore
            )

    @staticmethod
    def mock_done() -> bool:
        """Mock env being done.

        Returns:
            returns true.
        """
        return True


@typing.no_type_check
@pytest.fixture
def helpers() -> Helpers:
    """Return helper class.

    Returns:
        helpers class.
    """
    return Helpers
