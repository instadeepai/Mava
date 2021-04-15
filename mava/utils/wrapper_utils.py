from typing import Dict, Tuple, Union

import dm_env
import numpy as np
from dm_env import specs

# Need to install typing_extensions since we support pre python 3.8
from typing_extensions import TypedDict

from mava import types

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
