from typing import Dict, List, Tuple, Union

import dm_env
import numpy as np
from dm_env import specs

# Need to install typing_extensions since we support pre python 3.8
from typing_extensions import TypedDict

from mava import types
from mava.wrappers.pettingzoo import (
    PettingZooAECEnvWrapper,
    PettingZooParallelEnvWrapper,
)

SeqTimestepDict = TypedDict(
    "SeqTimestepDict",
    {"timestep": dm_env.TimeStep, "action": types.Action},
)


def generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
    return np.zeros(spec.shape, spec.dtype)


def convert_np_type(dtype: np.dtype, value: Union[int, float]) -> Union[int, float]:
    return np.dtype(dtype).type(value)


def parameterized_restart(
    reward: types.Reward,
    discount: types.Discount,
    observation: types.Observation,
) -> dm_env.TimeStep:
    """Returns a `TimeStep` with `step_type` set to `StepType.FIRST`.
    Differs from dm_env.restart, since reward and discount can be set to
    initial types."""
    return dm_env.TimeStep(dm_env.StepType.FIRST, reward, discount, observation)


"""Project single timestep to all agents."""


def broadcast_timestep_to_all_agents(
    timestep: dm_env.TimeStep, possible_agents: list
) -> dm_env.TimeStep:
    parallel_timestep = dm_env.TimeStep(
        observation={agent: timestep.observation for agent in possible_agents},
        reward={agent: timestep.reward for agent in possible_agents},
        discount={agent: timestep.discount for agent in possible_agents},
        step_type=timestep.step_type,
    )

    return parallel_timestep


"""Convert dict of seq timestep and actions to parallel"""


def convert_seq_timestep_and_actions_to_parallel(
    timesteps: Dict[str, SeqTimestepDict], possible_agents: list
) -> Tuple[dict, dm_env.TimeStep]:

    step_types = [timesteps[agent]["timestep"].step_type for agent in possible_agents]
    assert all(
        x == step_types[0] for x in step_types
    ), f"Step types should be identical - {step_types} "
    parallel_timestep = dm_env.TimeStep(
        observation={
            agent: timesteps[agent]["timestep"].observation for agent in possible_agents
        },
        reward={
            agent: timesteps[agent]["timestep"].reward for agent in possible_agents
        },
        discount={
            agent: timesteps[agent]["timestep"].discount for agent in possible_agents
        },
        step_type=step_types[0],
    )

    parallel_actions = {agent: timesteps[agent]["action"] for agent in possible_agents}

    return parallel_actions, parallel_timestep


def apply_env_wrapper_preprocessers(
    environment: Union[PettingZooAECEnvWrapper, PettingZooParallelEnvWrapper],
    env_preprocess_wrappers: List,
) -> Union[PettingZooAECEnvWrapper, PettingZooParallelEnvWrapper]:
    if env_preprocess_wrappers and isinstance(env_preprocess_wrappers, List):
        for (env_wrapper, params) in env_preprocess_wrappers:
            if params:
                environment = env_wrapper(environment, **params)
            else:
                environment = env_wrapper(environment)
    return environment


class RunningStatistics:
    """Helper class to comute running statistics such as
    the max, min, mean, variance and standard deviation of
    a specific quantity.
    """

    def __init__(self, label: str) -> None:
        self.count = 0
        self.old_mean = 0.0
        self.new_mean = 0.0
        self.old_var = 0.0
        self.new_var = 0.0

        self._max = -9999999.9
        self._min = 9999999.9

        self._label = label

    def push(self, x: float) -> None:
        self.count += 1

        if x > self._max:
            self._max = x

        if x < self._min:
            self._min = x

        if self.count == 1:
            self.old_mean = self.new_mean = x
            self.old_var = 0.0
        else:
            self.new_mean = self.old_mean + (x - self.old_mean) / self.count
            self.new_var = self.old_var + (x - self.old_mean) * (x - self.new_mean)

            self.old_mean = self.new_mean
            self.old_var = self.new_var

    def max(self) -> float:
        return self._max

    def min(self) -> float:
        return self._min

    def mean(self) -> float:
        return self.new_mean if self.count else 0.0

    def var(self) -> float:
        return self.new_var / (self.count - 1) if self.count > 1 else 0.0

    def std(self) -> float:
        return np.sqrt(self.var())
