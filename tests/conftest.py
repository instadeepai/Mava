import importlib
import typing
from enum import Enum
from typing import Dict, Tuple, Union

import dm_env
import pytest
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from pettingzoo.utils.env import AECEnv, ParallelEnv

from mava.wrappers.flatland import FlatlandEnvWrapper
from mava.wrappers.pettingzoo import (
    PettingZooAECEnvWrapper,
    PettingZooParallelEnvWrapper,
)


class EnvType(Enum):
    Sequential = 1
    Parallel = 2


class EnvSpec:
    def __init__(self, env_name: str, env_type: EnvType):
        self.env_name = env_name
        self.env_type = env_type


# flatland environment config
rail_gen_cfg: Dict = {
    "max_num_cities": 4,
    "max_rails_between_cities": 2,
    "max_rails_in_city": 3,
    "grid_mode": True,
    "seed": 42,
}

flatland_env_config: Dict = {
    "number_of_agents": 2,
    "width": 25,
    "height": 25,
    "rail_generator": sparse_rail_generator(**rail_gen_cfg),
    "schedule_generator": sparse_schedule_generator(),
    "obs_builder_object": TreeObsForRailEnv(max_depth=2),
}


# non-pettingzoo environment names
class NPZ_EnvName:
    FLATLAND = "flatland"


# non-pettingzoo environment wrappers
npz_env_wrappers: Dict = {NPZ_EnvName.FLATLAND: FlatlandEnvWrapper}

"""
Helpers contains re-usable test functions.
"""

# TODO(Kale-ab): Better structure helper funcs


class Helpers:
    # Check all props are not none
    @staticmethod
    def verify_all_props_not_none(props_which_should_not_be_none: list) -> bool:
        return all(prop is not None for prop in props_which_should_not_be_none)

    # Return an env - currently Pettingzoo envs.
    @staticmethod
    def get_env(env_spec: EnvSpec) -> Tuple[Union[AECEnv, ParallelEnv], int]:
        env, num_agents = None, None
        if env_spec.env_name == NPZ_EnvName.FLATLAND:
            env = RailEnv(**flatland_env_config)
            # flatland does not specify action and observation spaces
            # we'll specify dummy spaces here
            env.action_spaces = {"": None}
            env.observation_spaces = {"": None}
        elif env_spec.env_type == EnvType.Parallel:
            mod = importlib.import_module(env_spec.env_name)
            env = mod.parallel_env()  # type: ignore
        elif env_spec.env_type == EnvType.Sequential:
            mod = importlib.import_module(env_spec.env_name)
            env = mod.env()  # type: ignore
        else:
            raise Exception("Env_spec is not valid.")
        env.reset()  # type: ignore
        num_agents = len(env.agents)  # type: ignore
        return env, num_agents

    # Returns a wrapper.
    @staticmethod
    def get_wrapper(
        env_spec: EnvSpec,
    ) -> dm_env.Environment:
        wrapper = None
        if env_spec.env_name in npz_env_wrappers.keys():
            wrapper = npz_env_wrappers[env_spec.env_name]
        elif env_spec.env_type == EnvType.Parallel:
            wrapper = PettingZooParallelEnvWrapper
        elif env_spec.env_type == EnvType.Sequential:
            wrapper = PettingZooAECEnvWrapper
        else:
            raise Exception("Env_spec is not valid.")
        return wrapper

    # Seeds action space
    @staticmethod
    def seed_action_space(
        env_wrapper: Union[PettingZooAECEnvWrapper, PettingZooParallelEnvWrapper],
        random_seed: int,
    ) -> None:
        [
            env_wrapper.action_spaces[agent].seed(random_seed)
            for agent in env_wrapper.agents
        ]

    @staticmethod
    def compare_dicts(dictA: Dict, dictB: Dict) -> bool:
        typesA = [type(k) for k in dictA.values()]
        typesB = [type(k) for k in dictB.values()]

        return (dictA == dictB) and (typesA == typesB)


@typing.no_type_check
@pytest.fixture
def helpers() -> Helpers:
    return Helpers
