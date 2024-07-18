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
from multiprocessing import Queue
from multiprocessing.connection import Connection
from typing import Any, Callable, Dict, Optional, Tuple, Union

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import write_to_shared_memory
from numpy.typing import NDArray

# Filter out the warnings
warnings.filterwarnings("ignore", module="gymnasium.utils.passive_env_checker")


class GymWrapper(gymnasium.Wrapper):
    """Wrapper for gym environments."""

    def __init__(
        self,
        env: gymnasium.Env,
        use_shared_rewards: bool = True,
        add_global_state: bool = False,
    ):
        """Initialise the gym wrapper
        Args:
            env (gymnasium.env): gymnasium env instance.
            use_shared_rewards (bool, optional): Use individual or shared rewards.
            Defaults to False.
            add_global_state (bool, optional) : Create global observations. Defaults to False.
        """
        super().__init__(env)
        self._env = env
        self.use_shared_rewards = use_shared_rewards
        self.add_global_state = add_global_state
        self.num_agents = len(self._env.action_space)
        self.num_actions = self._env.action_space[0].n

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[NDArray, Dict]:

        if seed is not None:
            self.env.seed(seed)

        agents_view, info = self._env.reset()

        info = {"actions_mask": self.get_actions_mask(info)}
        if self.add_global_state:
            info["global_obs"] = self.get_global_obs(agents_view)

        return np.array(agents_view), info

    def step(self, actions: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray, Dict]:

        agents_view, reward, terminated, truncated, info = self._env.step(actions)

        info = {"actions_mask": self.get_actions_mask(info)}
        if self.add_global_state:
            info["global_obs"] = self.get_global_obs(agents_view)

        if self.use_shared_rewards:
            reward = np.array([np.array(reward).sum()] * self.num_agents)
        else:
            reward = np.array(reward)

        return agents_view, reward, terminated, truncated, info

    def get_actions_mask(self, info: Dict) -> NDArray:
        if "action_mask" in info:
            return np.array(info["action_mask"])
        return np.ones((self.num_agents, self.num_actions), dtype=np.float32)

    def get_global_obs(self, obs: NDArray) -> NDArray:
        global_obs = np.concatenate(obs, axis=0)
        return np.tile(global_obs, (self.num_agents, 1))


class GymLBFWrapper(GymWrapper):
    """Wrapper for LBF gym environments"""

    def __init__(
        self,
        env: gymnasium.Env,
        use_shared_rewards: bool = True,
        add_global_state: bool = False,
    ):
        """Initialise the gym wrapper
        Args:
            env (gymnasium.env): gymnasium env instance.
            use_shared_rewards (bool, optional): Use individual or shared rewards.
            Defaults to False.
            add_global_state (bool, optional) : Create global observations. Defaults to False.
        """
        super().__init__(env, use_shared_rewards, add_global_state)

    def step(self, actions: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray, Dict]:

        agents_view, reward, terminated, truncated, info = super().step(actions)

        truncated = np.repeat(truncated, self.num_agents)
        terminated = np.repeat(terminated, self.num_agents)

        return agents_view, reward, terminated, truncated, info


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

        # Reset the env
        agents_view, info = self._env.reset(seed, options)

        # Create the metrics dict
        metrics = {
            "episode_return": self.running_count_episode_return,
            "episode_length": self.running_count_episode_length,
            "is_terminal_step": True,
        }

        # Reset the metrics
        self.running_count_episode_return = 0.0
        self.running_count_episode_length = 0

        if "won_episode" in info:
            metrics["won_episode"] = info["won_episode"]

        info["metrics"] = metrics

        return agents_view, info

    def step(self, actions: NDArray) -> Tuple:

        # Step the env
        agents_view, reward, terminated, truncated, info = self._env.step(actions)

        self.running_count_episode_return += float(np.mean(reward))
        self.running_count_episode_length += 1

        metrics = {
            "episode_return": self.running_count_episode_return,
            "episode_length": self.running_count_episode_length,
            "is_terminal_step": False,
        }
        if "won_episode" in info:
            metrics["won_episode"] = info["won_episode"]

        info["metrics"] = metrics

        return agents_view, reward, terminated, truncated, info


class GymAgentIDWrapper(gymnasium.Wrapper):
    """Add one hot agent IDs to observation."""

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)

        self.agent_ids = np.eye(self.env.num_agents)
        observation_space = self.env.observation_space[0]
        _obs_low, _obs_high, _obs_dtype, _obs_shape = (
            observation_space.low[0],
            observation_space.high[0],
            observation_space.dtype,
            observation_space.shape,
        )
        _new_obs_shape = (_obs_shape[0] + self.env.num_agents,)
        _observation_boxs = [
            spaces.Box(low=_obs_low, high=_obs_high, shape=_new_obs_shape, dtype=_obs_dtype)
        ] * self.env.num_agents
        self.observation_space = spaces.Tuple(_observation_boxs)

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


# Copied form Gymnasium/blob/main/gymnasium/vector/async_vector_env.py
# Modified to work with multiple agents
def async_multiagent_worker(  # noqa CCR001
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
    autoreset = False

    parent_pipe.close()

    try:
        while True:
            command, data = pipe.recv()

            if command == "reset":
                observation, info = env.reset(**data)
                if shared_memory:
                    write_to_shared_memory(observation_space, index, observation, shared_memory)
                    observation = None
                    autoreset = False
                pipe.send(((observation, info), True))
            elif command == "step":
                if autoreset:
                    observation, info = env.reset()
                    reward, terminated, truncated = 0, False, False
                else:
                    (
                        observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = env.step(data)
                autoreset = np.logical_or(terminated, truncated).all()

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
                pipe.send(
                    (
                        (data[0] == observation_space, data[1] == action_space),
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
