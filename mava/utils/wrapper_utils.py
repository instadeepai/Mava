import collections
from typing import Any, Dict, List, Tuple, Union

import dm_env
import numpy as np
from dm_env import specs
from gym import Space
from pettingzoo.utils.conversions import ParallelEnv
from pettingzoo.utils.env import AECEnv

# Need to install typing_extensions since we support pre python 3.8
from typing_extensions import TypedDict

from mava import types

SeqTimestepDict = TypedDict(
    "SeqTimestepDict",
    {"timestep": dm_env.TimeStep, "action": types.Action},
)


def convert_dm_compatible_observations(
    observes: Dict[str, np.ndarray],
    dones: Dict[str, bool],
    action_spaces: Dict[str, Space],
    observation_spaces: Dict[str, Space],
    env_done: bool,
    possible_agents: List,
) -> types.Observation:
    """Convert Parallel observation so it's dm_env compatible.

    Args:
        observes (Dict[str, np.ndarray]): observations per agent.
        dones (Dict[str, bool]): dones per agent.
        action_spaces ( Dict[str, Space]): env action spaces.
        observation_spaces ( Dict[str, Space]): env observation spaces.
        env_done (bool): is env done.
        possible_agents (List): possible agents in env.

    Returns:
        types.Observation: dm compatible observation.
    """
    observations: Dict[str, types.OLT] = {}
    if observes:
        for agent, observation in observes.items():
            if isinstance(observation, dict) and "action_mask" in observation:
                legals = observation["action_mask"]
                observation = observation["observation"]
            else:
                # TODO Handle legal actions better for continous envs,
                # maybe have min and max for each action and clip the
                # agents actions  accordingly
                legals = np.ones(
                    action_spaces[agent].shape,
                    dtype=action_spaces[agent].dtype,
                )
            observations[agent] = types.OLT(
                observation=observation,
                legal_actions=legals,
                terminal=np.asarray([dones[agent]], dtype=np.float32),
            )
    # Handle empty observations - some envs return {} at last step.
    else:
        observations = {
            agent: types.OLT(
                observation=np.zeros(
                    observation_spaces[agent].shape,
                    dtype=observation_spaces[agent].dtype,
                ),
                legal_actions=np.ones(
                    action_spaces[agent].shape,
                    dtype=action_spaces[agent].dtype,
                ),
                terminal=np.asarray([env_done], dtype=np.float32),
            )
            for agent in possible_agents
        }
    return observations


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


def parameterized_termination(
    reward: types.Reward,
    discount: types.Discount,
    observation: types.Observation,
) -> dm_env.TimeStep:
    """Returns a `TimeStep` with `step_type` set to `StepType.LAST`."""
    return dm_env.TimeStep(dm_env.StepType.LAST, reward, discount, observation)


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
    environment: Any,
    env_preprocess_wrappers: List,
) -> Any:
    # Currently only supports PZ envs.
    if isinstance(environment, ParallelEnv) or isinstance(environment, AECEnv):
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

    # The queue_size is used to estimate a moving mean and variance value.
    def __init__(self, label: str, queue_size: int = 100) -> None:

        self.queue: collections.deque = collections.deque(maxlen=queue_size)
        self._max = -float("inf")
        self._min = float("inf")

        self._mean = 0.0

        self._var = 0.0

        self._label = label

        self._raw = 0.0

    def push(self, x: float) -> None:
        self._raw = x
        self.queue.append(x)

        if x > self._max:
            self._max = x

        if x < self._min:
            self._min = x

        if len(self.queue) == 1:
            self._mean = x
            self._var = 0
        else:
            self._mean = np.mean(self.queue)
            self._var = np.var(self.queue)

    def max(self) -> float:
        return self._max

    def min(self) -> float:
        return self._min

    def mean(self) -> float:
        return self._mean

    def var(self) -> float:
        return self._var

    def std(self) -> float:
        return np.sqrt(self._var)

    def raw(self) -> float:
        return self._raw


# Adapted From https://github.com/DLR-RM/stable-baselines3/blob/237223f834fe9b8143ea24235d087c4e32addd2f/stable_baselines3/common/running_mean_std.py # noqa: E501
class RunningMeanStd(object):
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update_batch(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr)
        batch_var = np.var(arr)
        batch_count = 1
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
