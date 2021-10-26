from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from mava.wrappers.flatland3 import Flatland3EnvWrapper

def create_rail_env(
    n_agents=5,
    x_dim=30,
    y_dim=30,
    n_cities=2,
    max_rails_between_cities=2,
    max_rails_in_city=3,
    seed=0,
    malfunction_rate=1/200,
    observation_max_path_depth=30,
    observation_tree_depth=2,
    evaluation=True
):
    """Create Rail Env"""
    # Break agents from time to time
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=malfunction_rate,
        min_duration=20,
        max_duration=50
    )

    # Observation builder
    predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
    tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=predictor)

    rail_env = RailEnv(
        width=x_dim, height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            grid_mode=False,
            max_rails_between_cities=max_rails_between_cities,
            max_rail_pairs_in_city=max_rails_in_city//2
        ),
        line_generator=sparse_line_generator(),
        number_of_agents=n_agents,
        malfunction_generator=ParamMalfunctionGen(malfunction_parameters),
        obs_builder_object=tree_observation,
        random_seed=seed
    )

    return Flatland3EnvWrapper(rail_env)