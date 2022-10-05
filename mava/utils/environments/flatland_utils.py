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

"""Utils for making Flatland environment."""
from typing import Dict, Optional, Tuple

from mava.utils.jax_training_utils import set_jax_double_precision
from mava.wrappers.env_preprocess_wrappers import (
    ConcatAgentIdToObservation,
    ConcatPrevActionToObservation,
)
from mava.wrappers.flatland import FlatlandEnvWrapper


def check_flatland_import() -> bool:
    """Consistent way to check if flatland has been installed.

    Returns:
        whether flatland exists or not.
    """
    try:
        from flatland.envs.line_generators import sparse_line_generator

        # Delete unused var
        del sparse_line_generator
        _found_flatland = True

    except ModuleNotFoundError:
        _found_flatland = False
    return _found_flatland


_found_flatland = check_flatland_import()
if _found_flatland:
    from flatland.envs.line_generators import sparse_line_generator
    from flatland.envs.malfunction_generators import (
        MalfunctionParameters,
        ParamMalfunctionGen,
    )
    from flatland.envs.observations import TreeObsForRailEnv
    from flatland.envs.predictions import ShortestPathPredictorForRailEnv
    from flatland.envs.rail_env import RailEnv
    from flatland.envs.rail_generators import sparse_rail_generator

    def _create_rail_env_with_tree_obs(
        n_agents: int = 5,
        x_dim: int = 30,
        y_dim: int = 30,
        n_cities: int = 2,
        max_rails_between_cities: int = 2,
        max_rails_in_city: int = 3,
        seed: Optional[int] = 0,
        malfunction_rate: float = 1 / 200,
        malfunction_min_duration: int = 20,
        malfunction_max_duration: int = 50,
        observation_max_path_depth: int = 30,
        observation_tree_depth: int = 2,
    ) -> RailEnv:
        """Create a Flatland RailEnv with TreeObservation.

        Args:
            n_agents: Number of trains. Defaults to 5.
            x_dim: Width of map. Defaults to 30.
            y_dim: Height of map. Defaults to 30.
            n_cities: Number of cities. Defaults to 2.
            max_rails_between_cities: Max rails between cities. Defaults to 2.
            max_rails_in_city: Max rails in cities. Defaults to 3.
            seed: Random seed. Defaults to 0.
            malfunction_rate: Malfunction rate. Defaults to 1/200.
            malfunction_min_duration: Min malfunction duration. Defaults to 20.
            malfunction_max_duration: Max malfunction duration. Defaults to 50.
            observation_max_path_depth: Shortest path predictor depth. Defaults to 30.
            observation_tree_depth: TreeObs depth. Defaults to 2.

        Returns:
            RailEnv: A Flatland RailEnv.
        """

        # Break agents from time to time
        malfunction_parameters = MalfunctionParameters(
            malfunction_rate=malfunction_rate,
            min_duration=malfunction_min_duration,
            max_duration=malfunction_max_duration,
        )

        # Observation builder
        predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
        tree_observation = TreeObsForRailEnv(
            max_depth=observation_tree_depth, predictor=predictor
        )

        rail_env = RailEnv(
            width=x_dim,
            height=y_dim,
            rail_generator=sparse_rail_generator(
                max_num_cities=n_cities,
                grid_mode=False,
                max_rails_between_cities=max_rails_between_cities,
                max_rail_pairs_in_city=max_rails_in_city // 2,
            ),
            line_generator=sparse_line_generator(),
            number_of_agents=n_agents,
            malfunction_generator=ParamMalfunctionGen(malfunction_parameters),
            obs_builder_object=tree_observation,
            random_seed=seed,
        )

        return rail_env

    def make_flatland_task_name(x_dim: int, y_dim: int, num_agents: int) -> str:
        """A simple helper function to create a flatland task name.

        The task name will be a string created as:
        `<map_width>x<map_height>_<num_agents>_trains`. For example on a
        5x5 maps with 3 agents the task name will be `5x5_3_trains`.
        """

        env_width = str(x_dim)
        env_height = str(y_dim)
        num_trains = str(num_agents)

        task_name = f"{env_width}x{env_height}_{num_trains}_trains"

        return task_name

    def make_environment(
        n_agents: int = 10,
        x_dim: int = 30,
        y_dim: int = 30,
        n_cities: int = 2,
        max_rails_between_cities: int = 2,
        max_rails_in_city: int = 3,
        seed: int = 0,
        malfunction_rate: float = 1 / 200,
        malfunction_min_duration: int = 20,
        malfunction_max_duration: int = 50,
        observation_max_path_depth: int = 30,
        observation_tree_depth: int = 2,
        concat_prev_actions: bool = True,
        concat_agent_id: bool = False,
        evaluation: bool = False,
        random_seed: Optional[int] = None,
    ) -> Tuple[FlatlandEnvWrapper, Dict[str, str]]:
        """Loads a flatand environment and wraps it using the flatland wrapper"""

        del evaluation  # since it has same behaviour for both train and eval

        # Env uses int64 action space due to the use of spac.Discrete.
        set_jax_double_precision()

        env = _create_rail_env_with_tree_obs(
            n_agents=n_agents,
            x_dim=x_dim,
            y_dim=y_dim,
            n_cities=n_cities,
            max_rails_between_cities=max_rails_between_cities,
            max_rails_in_city=max_rails_in_city,
            seed=random_seed,
            malfunction_rate=malfunction_rate,
            malfunction_min_duration=malfunction_min_duration,
            malfunction_max_duration=malfunction_max_duration,
            observation_max_path_depth=observation_max_path_depth,
            observation_tree_depth=observation_tree_depth,
        )

        env = FlatlandEnvWrapper(env)

        if concat_prev_actions:
            env = ConcatPrevActionToObservation(env)

        if concat_agent_id:
            env = ConcatAgentIdToObservation(env)

        environment_task_name = {
            "environment_name": "flatland",
            "task_name": make_flatland_task_name(
                x_dim=x_dim, y_dim=y_dim, num_agents=n_agents
            ),
        }

        return env, environment_task_name
