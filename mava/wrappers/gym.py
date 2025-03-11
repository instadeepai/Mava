# Copyright 2022 InstaDeep Ltd. All rights reserved.
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

import sys
import traceback
import warnings
from dataclasses import field
from enum import IntEnum
from multiprocessing import Queue
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium
import gymnasium.vector.async_vector_env
import numpy as np
from gymnasium import spaces
from gymnasium.spaces.utils import is_space_dtype_shape_equiv
from gymnasium.vector.utils import write_to_shared_memory
from numpy.typing import NDArray

from mava.types import MavaObservation, Observation, ObservationGlobalState

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

# Filter out the warnings
warnings.filterwarnings("ignore", module="gymnasium.utils.passive_env_checker")


# needed to avoid host -> device transfers when calling TimeStep.last()
class StepType(IntEnum):
    """Copy of Jumanji's step type but without jax arrays"""

    FIRST = 0
    MID = 1
    LAST = 2


@dataclass
class TimeStep:
    step_type: StepType
    reward: NDArray
    discount: NDArray
    observation: MavaObservation
    extras: Dict = field(default_factory=dict)

    def first(self) -> NDArray:
        return self.step_type == StepType.FIRST

    def mid(self) -> NDArray:
        return self.step_type == StepType.MID

    def last(self) -> NDArray:
        return self.step_type == StepType.LAST


class UoeWrapper(gymnasium.Wrapper):
    """A base wrapper for multi-agent environments developed by the University of Edinburgh.
    This wrapper is compatible with the RobotWarehouse and Level-Based Foraging environments.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        use_shared_rewards: bool = True,
        add_global_state: bool = False,
    ):
        """Initialize the gym wrapper
        Args:
            env (gymnasium.env): gymnasium env instance.
            use_shared_rewards (bool, optional): Use individual or shared rewards.
            Defaults to False.
            add_global_state (bool, optional) : Add global state information
            to observations.
        """
        super().__init__(env)
        self._env = env
        self.use_shared_rewards = use_shared_rewards
        self.add_global_state = add_global_state
        self.num_agents = len(self._env.action_space)
        self.num_actions = self._env.action_space[0].n

        # Tuple(Box(...) * N) --> Box(N, ...)
        single_obs = self.observation_space[0]  # type: ignore
        shape = (self.num_agents, *single_obs.shape)
        low = np.tile(single_obs.low, (self.num_agents, 1))
        high = np.tile(single_obs.high, (self.num_agents, 1))
        self.observation_space = spaces.Box(low=low, high=high, shape=shape, dtype=single_obs.dtype)

        # Tuple(Discrete(...) * N) --> MultiDiscrete(... * N)
        self.action_space = spaces.MultiDiscrete([self.num_actions] * self.num_agents)

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[NDArray, Dict]:
        if seed is not None:
            self.env.unwrapped.seed(seed)

        agents_view, info = self._env.reset()

        info["action_mask"] = self.get_action_mask(info)
        if self.add_global_state:
            info["global_obs"] = self.get_global_obs(agents_view)

        return np.array(agents_view), info

    def step(self, actions: List) -> Tuple[NDArray, NDArray, NDArray, NDArray, Dict]:
        agents_view, reward, terminated, truncated, info = self._env.step(actions)

        info["action_mask"] = self.get_action_mask(info)
        if self.add_global_state:
            info["global_obs"] = self.get_global_obs(agents_view)

        if self.use_shared_rewards:
            reward = np.array([np.array(reward).sum()] * self.num_agents)
        else:
            reward = np.array(reward)

        return agents_view, reward, terminated, truncated, info

    def get_action_mask(self, info: Dict) -> NDArray:
        if "action_mask" in info:
            return np.array(info["action_mask"])
        return np.ones((self.num_agents, self.num_actions), dtype=np.float32)

    def get_global_obs(self, obs: NDArray) -> NDArray:
        global_obs = np.concatenate(obs, axis=0)
        return np.tile(global_obs, (self.num_agents, 1))


class SmacWrapper(UoeWrapper):
    """A wrapper that converts actions to integers."""

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[NDArray, Dict]:
        agents_view, info = super().reset()
        info["won_episode"] = info["battle_won"]
        return agents_view, info

    def step(self, actions: List) -> Tuple[NDArray, NDArray, NDArray, NDArray, Dict]:
        # Convert actions to integers before passing them to the environment
        actions = [int(action) for action in actions]

        agents_view, reward, terminated, truncated, info = super().step(actions)
        info["won_episode"] = info["battle_won"]

        return agents_view, reward, terminated, truncated, info

    def get_action_mask(self, info: Dict) -> NDArray:
        return np.array(self._env.unwrapped.get_avail_actions())


class GymRecordEpisodeMetrics(gymnasium.Wrapper):
    """Record the episode returns and lengths."""

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        self._env = env
        self.running_count_episode_return = 0.0
        self.running_count_episode_length = 0.0

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[NDArray, Dict]:
        agents_view, info = self._env.reset(seed, options)

        # Reset the metrics
        self.running_count_episode_return = 0.0
        self.running_count_episode_length = 0.0

        # Create the metrics dict
        metrics = {
            "episode_return": self.running_count_episode_return,
            "episode_length": self.running_count_episode_length,
            "is_terminal_step": False,
        }

        info["metrics"] = metrics

        return agents_view, info

    def step(self, actions: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray, Dict]:
        agents_view, reward, terminated, truncated, info = self._env.step(actions)

        self.running_count_episode_return += float(np.mean(reward))
        self.running_count_episode_length += 1

        metrics = {
            "episode_return": self.running_count_episode_return,
            "episode_length": self.running_count_episode_length,
            "is_terminal_step": np.logical_or(terminated, truncated).all().item(),
        }

        info["metrics"] = metrics

        return agents_view, reward, terminated, truncated, info


class GymAgentIDWrapper(gymnasium.Wrapper):
    """Add one hot agent IDs to observation."""

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)

        self.agent_ids = np.eye(self.env.num_agents)
        self.observation_space = self.modify_space(self.env.observation_space)

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[NDArray, Dict]:
        """Reset the environment."""
        obs, info = self.env.reset(seed, options)
        obs = np.concatenate([self.agent_ids, obs], axis=1)
        return obs, info

    def step(self, action: list) -> Tuple[NDArray, float, bool, bool, Dict]:
        """Step the environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = np.concatenate([self.agent_ids, obs], axis=1)
        return obs, reward, terminated, truncated, info

    def modify_space(self, space: spaces.Space) -> spaces.Space:
        if isinstance(space, spaces.Box):
            new_shape = (space.shape[0], space.shape[1] + self.env.num_agents)
            high = np.concatenate((space.high, np.ones_like(self.agent_ids)), axis=1)
            low = np.concatenate((space.low, np.zeros_like(self.agent_ids)), axis=1)
            return spaces.Box(low=low, high=high, shape=new_shape, dtype=space.dtype)
        elif isinstance(space, spaces.Tuple):
            return spaces.Tuple(self.modify_space(s) for s in space)
        else:
            raise ValueError(f"Space {type(space)} is not currently supported.")


class GymToJumanji:
    """Converts from the Gym API to the Jumanji API."""

    def __init__(self, env: gymnasium.vector.VectorEnv):
        self.env = env
        self.single_action_space = env.unwrapped.single_action_space
        self.single_observation_space = env.unwrapped.single_observation_space
        self.num_agents = len(self.env.single_action_space)

    def reset(self, seed: Optional[list[int]] = None, options: Optional[dict] = None) -> TimeStep:
        obs, info = self.env.reset(seed=seed, options=options)  # type: ignore

        num_envs = self.env.num_envs

        step_type = np.full(num_envs, StepType.FIRST)
        rewards = np.zeros((num_envs, self.num_agents), dtype=float)
        terminated = np.zeros((num_envs, self.num_agents), dtype=float)

        timestep = self._create_timestep(obs, step_type, terminated, rewards, info)

        return timestep

    def step(self, action: list) -> TimeStep:
        obs, rewards, terminated, truncated, info = self.env.step(action)

        ep_done = np.logical_or(terminated, truncated)
        step_type = np.where(ep_done, StepType.LAST, StepType.MID)
        terminated = np.repeat(
            terminated[..., np.newaxis], repeats=self.num_agents, axis=-1
        )  # (B,) --> (B, N)

        timestep = self._create_timestep(obs, step_type, terminated, rewards, info)

        return timestep

    def _format_observation(
        self,
        obs: NDArray,
        action_mask: Tuple[NDArray],
        global_obs: Tuple[Union[NDArray, None]] = (None,),
    ) -> Union[Observation, ObservationGlobalState]:
        """Create an observation from the raw observation and environment state."""

        action_mask = np.stack(action_mask)
        obs_data = {"agents_view": obs, "action_mask": action_mask}

        if global_obs[0] is not None:
            global_obs = np.array(global_obs)
            obs_data["global_state"] = global_obs
            return ObservationGlobalState(**obs_data)
        else:
            return Observation(**obs_data)

    def _create_timestep(
        self, obs: NDArray, step_type: NDArray, terminated: NDArray, rewards: NDArray, info: Dict
    ) -> TimeStep:
        observation = self._format_observation(
            obs, info["action_mask"], info.get("global_obs", (None,))
        )
        # Filter out the masks and auxiliary data
        extras = {}
        extras["episode_metrics"] = {
            key: value for key, value in info["metrics"].items() if key[0] != "_"
        }
        extras["real_next_obs"] = self._format_observation(  # type: ignore
            info["real_next_obs"], info["real_next_action_mask"], info["real_next_global_obs"]
        )

        if "won_episode" in info:
            extras["won_episode"] = info["won_episode"]

        return TimeStep(
            step_type=step_type,
            reward=rewards,
            discount=1.0 - terminated,
            observation=observation,
            extras=extras,
        )

    def close(self) -> None:
        self.env.close()


# Copied from Gymnasium/blob/main/gymnasium/vector/async_vector_env.py
# Modified to work with multiple agents
# Note: The worker handles auto-resetting the environments.
# Each environment resets when all of its agents have either terminated or been truncated.
def async_multiagent_worker(  # CCR001
    index: int,
    env_fn: Callable,
    pipe: Connection,
    parent_pipe: Connection,
    shared_memory: Union[NDArray, dict[str, Any], tuple[Any, ...]],
    error_queue: Queue,
) -> None:
    env = env_fn()
    observation_space = env.observation_space
    action_space = env.action_space
    parent_pipe.close()

    try:
        while True:
            command, data = pipe.recv()

            if command == "reset":
                observation, info = env.reset(**data)
                info["real_next_obs"] = observation
                info["real_next_action_mask"] = info["action_mask"]
                info["real_next_global_obs"] = info.get("global_obs", None)
                if shared_memory:
                    write_to_shared_memory(observation_space, index, observation, shared_memory)
                    observation = None
                pipe.send(((observation, info), True))
            elif command == "step":
                # Modified the step function to align with 'AutoResetWrapper'.
                # The environment resets when all agents have either terminated or truncated.
                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(data)
                info["real_next_obs"] = observation
                info["real_next_action_mask"] = info["action_mask"]
                info["real_next_global_obs"] = info.get("global_obs", None)
                if np.logical_or(terminated, truncated).all():
                    observation, new_info = env.reset()
                    info["action_mask"] = new_info["action_mask"]
                    info["global_obs"] = new_info.get("global_obs", None)

                if shared_memory:
                    write_to_shared_memory(observation_space, index, observation, shared_memory)
                    observation = None

                pipe.send(((observation, reward, terminated, truncated, info), True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "close", "_setattr", "_check_spaces"]:
                    raise ValueError(
                        f"Trying to call function `{name}` with \
                        `call`, use `{name}` directly instead."
                    )

                attr = env.get_wrapper_attr(name)
                if callable(attr):
                    pipe.send((attr(*args, **kwargs), True))
                else:
                    pipe.send((attr, True))
            elif command == "_setattr":
                name, value = data
                env.set_wrapper_attr(name, value)
                pipe.send((None, True))
            elif command == "_check_spaces":
                obs_mode, single_obs_space, single_action_space = data
                pipe.send(
                    (
                        (
                            (
                                single_obs_space == observation_space
                                if obs_mode == "same"
                                else is_space_dtype_shape_equiv(single_obs_space, observation_space)
                            ),
                            single_action_space == action_space,
                        ),
                        True,
                    )
                )
            else:
                raise RuntimeError(
                    f"Received unknown command `{command}`. Must be one of \
                    [`reset`, `step`, `close`, `_call`, `_setattr`, `_check_spaces`]."
                )
    except (KeyboardInterrupt, Exception):
        error_type, error_message, _ = sys.exc_info()
        trace = traceback.format_exc()

        error_queue.put((index, error_type, error_message, trace))
        pipe.send((None, False))
    finally:
        env.close()
