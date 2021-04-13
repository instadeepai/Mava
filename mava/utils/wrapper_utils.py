from typing import Dict, NamedTuple, Union, Tuple

import dm_env
import numpy as np
from acme import types
from dm_env import specs

# Need to install typing_extensions since we support pre python 3.8
from typing_extensions import TypedDict

SequentialTimestepActions = TypedDict(
    "AgentSeqInfo",
    {"timestep": dm_env.TimeStep, "action": Union[int, float, types.NestedArray]},
)


def generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
    return np.zeros(spec.shape, spec.dtype)


def convert_np_type(dtype: np.dtype, value: Union[int, float]) -> Union[int, float]:
    return np.dtype(dtype).type(value)


def parameterized_restart(
    reward: Union[int, float, types.NestedArray],
    discount: Union[int, float, types.NestedArray],
    observation: Union[Dict, types.NestedArray],
) -> dm_env.TimeStep:
    """Returns a `TimeStep` with `step_type` set to `StepType.FIRST`.
    Differs from dm_env.restart, since reward and discount can be set to
    initial types."""
    return dm_env.TimeStep(dm_env.StepType.FIRST, reward, discount, observation)


class OLT(NamedTuple):
    """Container for (observation, legal_actions, terminal) tuples."""

    observation: types.Nest
    legal_actions: types.Nest
    terminal: types.Nest


"""Project single timestep to all agents."""


def project_timestep_to_all_agents(
    timestep: dm_env.TimeStep, possible_agents: list
) -> dm_env.TimeStep:
    parallel_timestep = dm_env.TimeStep(
        observation={agent: timestep.observation for agent in possible_agents},
        reward={agent: timestep.reward for agent in possible_agents},
        discount={agent: timestep.discount for agent in possible_agents},
        step_type=timestep.step_type,
    )

    return parallel_timestep


"""Convert seq timestep and actions to parallel"""


def convert_seq_timestep_and_actions_to_parallel(
    timesteps: SequentialTimestepActions, possible_agents: list
) -> Tuple[dm_env.TimeStep, dict]:

    # Use each agents timestep
    parallel_timestep = dm_env.TimeStep(
        observation={
            agent: timesteps[agent].get("timestep").observation
            for agent in possible_agents
        },
        reward={
            agent: timesteps[agent].get("timestep").reward for agent in possible_agents
        },
        discount={
            agent: timesteps[agent].get("timestep").discount
            for agent in possible_agents
        },
        step_type={
            agent: timesteps[agent].get("timestep").step_type
            for agent in possible_agents
        },
    )

    parallel_actions = {
        agent: timesteps[agent].get("action") for agent in possible_agents
    }

    return parallel_actions, parallel_timestep
