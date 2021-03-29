import importlib
import typing
from enum import Enum
from typing import Tuple, Union

import dm_env
import pytest
from pettingzoo.utils.env import AECEnv, ParallelEnv

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


"""
Helpers contains re-usable test functions.
"""


class Helpers:
    # Check all props are not none
    @staticmethod
    def verify_all_props_not_none(props_which_should_not_be_none: list) -> bool:
        return all(prop is not None for prop in props_which_should_not_be_none)

    # Return an env - currently Pettingzoo envs.
    @staticmethod
    def retrieve_env(env_spec: EnvSpec) -> Tuple[Union[AECEnv, ParallelEnv], int]:
        env, num_agents = None, None
        if env_spec.env_type == EnvType.Parallel:
            mod = importlib.import_module(env_spec.env_name)
            env = mod.parallel_env()  # type: ignore
        elif env_spec.env_type == EnvType.Sequential:
            mod = importlib.import_module(env_spec.env_name)
            env = mod.env()  # type: ignore
        else:
            raise Exception("Env_spec is not valid.")
        env.reset()
        num_agents = len(env.agents)
        return env, num_agents

    # Returns a wrapper.
    @staticmethod
    def retrieve_wrapper(
        env_spec: EnvSpec,
    ) -> dm_env.Environment:
        wrapper = None
        if env_spec.env_type == EnvType.Parallel:
            wrapper = PettingZooParallelEnvWrapper
        elif env_spec.env_type == EnvType.Sequential:
            wrapper = PettingZooAECEnvWrapper
        else:
            raise Exception("Env_spec is not valid.")
        return wrapper


@typing.no_type_check
@pytest.fixture
def helpers() -> Helpers:
    return Helpers
