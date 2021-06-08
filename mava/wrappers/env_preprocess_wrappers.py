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

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import gym
import numpy as np
from pettingzoo.utils.wrappers import OrderEnforcingWrapper as PettingzooWrapper
from supersuit.aec_wrappers import ObservationWrapper as SequentialObservationWrapper
from supersuit.parallel_wrappers import ObservationWrapper as ParallelObservationWrapper
from supersuit.parallel_wrappers import ParallelWraper as ParallelEnvPettingZoo

from mava.types import Observation, Reward
from mava.utils.wrapper_utils import RunningMeanStd
from mava.wrappers.env_wrappers import ParallelEnvWrapper, SequentialEnvWrapper

# Prevent circular import issue.
if TYPE_CHECKING:
    from mava.wrappers.pettingzoo import (
        PettingZooAECEnvWrapper,
        PettingZooParallelEnvWrapper,
    )

PettingZooEnv = Union["PettingZooAECEnvWrapper", "PettingZooParallelEnvWrapper"]

# TODO(Kale-ab): Make wrapper more general
# Should Works across any SequentialEnvWrapper or ParallelEnvWrapper.
"""
GYM Preprocess Wrappers.

Other gym preprocess wrappers:
    https://github.com/PettingZoo-Team/SuperSuit/blob/1f02289e8f51082aa50a413b34700b67042410c6/supersuit/gym_wrappers.py
    https://github.com/openai/gym/tree/master/gym/wrappers
"""


class StandardizeObservationGym(gym.ObservationWrapper):
    """
    Standardize observations
    Ensures mean of 0 and standard deviation of 1 (unit variance) for obs.
    From https://github.com/ikostrikov/pytorch-a3c/blob/e898f7514a03de73a2bf01e7b0f17a6f93963389/envs.py # noqa: E501
    """

    def __init__(self, env: gym.Env = None) -> None:
        super(StandardizeObservationGym, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def _observation(self, observation: Any) -> Any:
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + observation.mean() * (
            1 - self.alpha
        )
        self.state_std = self.state_std * self.alpha + observation.std() * (
            1 - self.alpha
        )

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)


"""
Petting Zoo Env Preprocess Wrappers.

Other PZ preprocess wrappers:
    https://github.com/PettingZoo-Team/SuperSuit
"""


# TODO(Kale-ab): Configure this to use RunningMeanStd.
class StandardizeObservation:
    """
    Standardize observations
    Ensures mean of 0 and standard deviation of 1 (unit variance) for obs.
    Adapted from https://github.com/ikostrikov/pytorch-a3c/blob/e898f7514a03de73a2bf01e7b0f17a6f93963389/envs.py # noqa: E501

        :env Env to wrap.
        :load_params Params to load.
        :alpha To avoid division by zero.
    """

    def __init__(
        self,
        env: PettingZooEnv = None,
        load_params: Dict = None,
        alpha: float = 0.999,
    ):
        self.env: Optional[PettingZooEnv] = env
        self.alpha = alpha
        self._ini_params(load_params)

    def _ini_params(self, load_params: Dict = None) -> None:
        if load_params:
            self._internal_state = load_params
        else:
            params = {
                agent: {
                    "state_mean": 0,
                    "state_std": 0,
                    "num_steps": 0,
                }
                for agent in self.env.possible_agents  # type:ignore
            }
            self._internal_state = params

    def _get_updated_observation(
        self,
        agent: str,
        observation: Observation,
    ) -> Observation:

        state_mean = self._internal_state[agent].get("state_mean")
        state_std = self._internal_state[agent].get("state_std")
        num_steps = self._internal_state[agent].get("num_steps")

        state_mean = state_mean * self.alpha + observation.mean() * (  # type:ignore
            1 - self.alpha
        )
        state_std = state_std * self.alpha + observation.std() * (  # type:ignore
            1 - self.alpha
        )

        unbiased_mean = state_mean / (1 - pow(self.alpha, num_steps))
        unbiased_std = state_std / (1 - pow(self.alpha, num_steps))

        self._internal_state[agent]["state_mean"] = state_mean
        self._internal_state[agent]["state_std"] = state_std

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)


class StandardizeObservationSequential(
    SequentialObservationWrapper, StandardizeObservation
):
    """Standardize Obs in Sequential Env"""

    def __init__(
        self, env: PettingZooEnv, load_params: Dict = None, alpha: float = 0.999
    ) -> None:
        SequentialObservationWrapper.__init__(self, env)
        StandardizeObservation.__init__(self, env, load_params, alpha)

    def _modify_observation(self, agent: str, observation: Observation) -> Observation:
        self._internal_state[agent]["num_steps"] = (
            int(self._internal_state[agent]["num_steps"]) + 1
        )
        return self._get_updated_observation(agent, observation)

    def _check_wrapper_params(self) -> None:
        return

    def _modify_spaces(self) -> None:
        return


class StandardizeObservationParallel(
    ParallelObservationWrapper, StandardizeObservation
):
    """Standardize Obs in Parallel Env"""

    def __init__(
        self, env: PettingZooEnv, load_params: Dict = None, alpha: float = 0.999
    ) -> None:
        ParallelObservationWrapper.__init__(self, env)
        StandardizeObservation.__init__(self, env, load_params, alpha)

    def _modify_observation(self, agent: str, observation: Observation) -> Observation:
        self._internal_state[agent]["num_steps"] = (
            int(self._internal_state[agent]["num_steps"]) + 1
        )
        return self._get_updated_observation(agent, observation)

    def _check_wrapper_params(self) -> None:
        return

    def _modify_spaces(self) -> None:
        return


class StandardizeReward:
    """
    Standardize rewards
    We rescale rewards, but don't shift the mean - http://joschu.net/docs/nuts-and-bolts.pdf .
    Adapted from https://github.com/DLR-RM/stable-baselines3/blob/237223f834fe9b8143ea24235d087c4e32addd2f/stable_baselines3/common/vec_env/vec_normalize.py # noqa: E501
    and https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py.
        :env Env to wrap.
        :load_params Params to load.
        :upper_bound: Max value for discounted reward.
        :lower_bound: Min value for discounted reward.
        :alpha To avoid division by zero.
    """

    def __init__(
        self,
        env: PettingZooEnv = None,
        load_params: Dict = None,
        lower_bound: float = -10.0,
        upper_bound: float = 10.0,
        alpha: float = 0.999,
    ):
        self._ini_params(load_params)
        self.env = env
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.alpha = alpha

    def _ini_params(self, load_params: Dict = None) -> None:
        if load_params:
            self._internal_state = load_params
        else:
            params = {}
            gamma_dict = None
            if hasattr(self, "_discounts"):
                gamma_dict = self._discount  # type: ignore

            for agent in self.env.possible_agents:  # type:ignore
                gamma = gamma_dict[agent] if gamma_dict else 1
                params[agent] = {
                    "return": 0,
                    "ret_rms": RunningMeanStd(shape=()),
                    "gamma": gamma,
                }
            self._internal_state = params

    def _update_reward(self, agent: str, reward: Reward) -> None:
        """Update reward normalization statistics."""
        agent_dict = self._internal_state[agent]
        ret = agent_dict["return"]
        gamma = agent_dict["gamma"]
        ret_rms = agent_dict["ret_rms"]

        ret = ret * gamma + reward
        ret_rms.update(ret)

        agent_dict["return"] = ret
        agent_dict["ret_rms"] = ret_rms
        self._internal_state[agent] = agent_dict

    def normalize_reward(self, agent: str, reward: Reward) -> Reward:
        """
        Normalize rewards using rewards statistics.
        Calling this method does not update statistics.
        """
        agent_dict = self._internal_state[agent]
        ret_rms = agent_dict["ret_rms"]
        reward = np.clip(
            reward / np.sqrt(ret_rms.var + self.alpha),
            self.lower_bound,
            self.upper_bound,
        )
        return reward

    def _get_updated_reward(
        self,
        agent: str,
        reward: Reward,
    ) -> Reward:
        self._update_reward(agent, reward)
        reward = self.normalize_reward(agent, reward)
        return reward


class StandardizeRewardSequential(PettingzooWrapper, StandardizeReward):
    def __init__(
        self,
        env: SequentialEnvWrapper,
        load_params: Dict = None,
        lower_bound: float = -10.0,
        upper_bound: float = 10.0,
        alpha: float = 0.999,
    ) -> None:
        PettingzooWrapper.__init__(self, env)
        StandardizeReward.__init__(
            self, env, load_params, lower_bound, upper_bound, alpha
        )

    def reset(self) -> None:
        # Reset returns, but not running scores.
        for stats in self._internal_state.values():
            stats["return"] = 0

        super().reset()
        self.rewards = {
            agent: self._get_updated_reward(agent, reward)
            for agent, reward in self.rewards.items()  # type: ignore
        }
        self.__cumulative_rewards = {a: 0 for a in self.agents}
        self._accumulate_rewards()

    def step(self, action: np.ndarray) -> None:
        agent = self.env.agent_selection  # type: ignore
        super().step(action)
        self.rewards = {
            agent: self._get_updated_reward(agent, reward)
            for agent, reward in self.rewards.items()
        }
        self.__cumulative_rewards[agent] = 0
        self._cumulative_rewards = self.__cumulative_rewards
        self._accumulate_rewards()


class StandardizeRewardParallel(
    ParallelEnvPettingZoo,
    StandardizeReward,
):
    def __init__(
        self,
        env: ParallelEnvWrapper,
        load_params: Dict = None,
        lower_bound: float = -10.0,
        upper_bound: float = 10.0,
        alpha: float = 0.999,
    ) -> None:
        ParallelEnvPettingZoo.__init__(self, env)
        StandardizeReward.__init__(
            self, env, load_params, lower_bound, upper_bound, alpha
        )

    def reset(self) -> Observation:
        # Reset returns, but not running scores.
        for stats in self._internal_state.values():
            stats["return"] = 0

        obs = self.env.reset()  # type: ignore
        self.agents = self.env.agents  # type: ignore
        return obs

    def step(self, actions: Dict) -> Any:
        obs, rew, done, info = super().step(actions)
        rew = {
            agent: self._get_updated_reward(agent, rew) for agent, rew in rew.items()
        }
        return obs, rew, done, info
