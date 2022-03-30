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

import dm_env
import gym
import numpy as np

from mava.types import OLT, Action, Observation, Reward
from mava.utils.wrapper_utils import RunningMeanStd
from mava.wrappers.env_wrappers import ParallelEnvWrapper, SequentialEnvWrapper

try:
    from supersuit.utils.base_aec_wrapper import BaseWrapper

    _has_supersuit = True
except ModuleNotFoundError:
    _has_supersuit = False


try:
    import pettingzoo  # noqa: F401

    _has_petting_zoo = True
except ModuleNotFoundError:
    _has_petting_zoo = False

if _has_petting_zoo:
    from pettingzoo.utils import BaseParallelWraper

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
    https://github.com/PettingZoo-Team/SuperSuit/blob/1f02289e8f51082aa50a413b34700b67042410c6/supersuit/gym_wrappers.py # noqa: E501
    https://github.com/openai/gym/tree/master/gym/wrappers # noqa: E501
"""


class StandardizeObservationGym(gym.ObservationWrapper):
    """
    Standardize observations.

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


if _has_supersuit:

    class StandardizeObservationSequential(BaseWrapper, StandardizeObservation):
        """Standardize Obs in Sequential Env"""

        def __init__(
            self, env: PettingZooEnv, load_params: Dict = None, alpha: float = 0.999
        ) -> None:
            BaseWrapper.__init__(self, env)
            StandardizeObservation.__init__(self, env, load_params, alpha)

        def _modify_observation(
            self, agent: str, observation: Observation
        ) -> Observation:
            self._internal_state[agent]["num_steps"] = (
                int(self._internal_state[agent]["num_steps"]) + 1
            )
            return self._get_updated_observation(agent, observation)

        def _modify_action(self, agent: str, action: Action) -> Action:
            return action


if _has_petting_zoo:

    class StandardizeObservationParallel(BaseParallelWraper, StandardizeObservation):
        """Standardize Obs in Parallel Env"""

        def __init__(
            self, env: PettingZooEnv, load_params: Dict = None, alpha: float = 0.999
        ) -> None:
            BaseParallelWraper.__init__(self, env)
            StandardizeObservation.__init__(self, env, load_params, alpha)

        def _modify_observation(
            self, agent: str, observation: Observation
        ) -> Observation:
            self._internal_state[agent]["num_steps"] = (
                int(self._internal_state[agent]["num_steps"]) + 1
            )
            return self._get_updated_observation(agent, observation)

        def _modify_action(self, action: Action) -> Action:
            return action

        def reset(self) -> Dict:
            obss = super().reset()
            return {
                agent: self._modify_observation(agent, obs)
                for agent, obs in obss.items()
            }

        def step(self, actions: Action) -> Any:
            obss, rew, done, info = super().step(actions)
            obss = {
                agent: self._modify_observation(agent, obs)
                for agent, obs in obss.items()
            }
            return obss, rew, done, info


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


if _has_supersuit:

    class StandardizeRewardSequential(BaseWrapper, StandardizeReward):
        def __init__(
            self,
            env: SequentialEnvWrapper,
            load_params: Dict = None,
            lower_bound: float = -10.0,
            upper_bound: float = 10.0,
            alpha: float = 0.999,
        ) -> None:
            BaseWrapper.__init__(self, env)
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

        def _modify_observation(
            self, agent: str, observation: Observation
        ) -> Observation:
            return observation

        def _modify_action(self, agent: str, action: Action) -> Action:
            return action


if _has_petting_zoo:

    class StandardizeRewardParallel(
        BaseParallelWraper,
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
            BaseParallelWraper.__init__(self, env)
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
                agent: self._get_updated_reward(agent, rew)
                for agent, rew in rew.items()
            }
            return obs, rew, done, info

        def _modify_observation(self, observation: Observation) -> Observation:
            return observation

        def _modify_action(self, action: Action) -> Action:
            return action


class ConcatAgentIdToObservation:
    """Concat one-hot vector of agent ID to obs.

    We assume the environment has an ordered list
    self.possible_agents.
    """

    def __init__(self, environment: Any) -> None:
        self._environment = environment
        self._num_agents = len(environment.possible_agents)

    def reset(self) -> dm_env.TimeStep:
        """Reset environment and concat agent ID."""
        timestep, extras = self._environment.reset()
        old_observations = timestep.observation

        new_observations = {}

        for agent_id, agent in enumerate(self._environment.possible_agents):
            agent_olt = old_observations[agent]

            agent_observation = agent_olt.observation
            agent_one_hot = np.zeros(self._num_agents, dtype=agent_observation.dtype)
            agent_one_hot[agent_id] = 1

            new_observations[agent] = OLT(
                observation=np.concatenate([agent_one_hot, agent_observation]),
                legal_actions=agent_olt.legal_actions,
                terminal=agent_olt.terminal,
            )

        return (
            dm_env.TimeStep(
                timestep.step_type, timestep.reward, timestep.discount, new_observations
            ),
            extras,
        )

    def step(self, actions: Dict) -> dm_env.TimeStep:
        """Step the environment and concat agent ID"""
        timestep, extras = self._environment.step(actions)

        old_observations = timestep.observation
        new_observations = {}
        for agent_id, agent in enumerate(self._environment.possible_agents):
            agent_olt = old_observations[agent]

            agent_observation = agent_olt.observation
            agent_one_hot = np.zeros(self._num_agents, dtype=agent_observation.dtype)
            agent_one_hot[agent_id] = 1

            new_observations[agent] = OLT(
                observation=np.concatenate([agent_one_hot, agent_observation]),
                legal_actions=agent_olt.legal_actions,
                terminal=agent_olt.terminal,
            )

        return (
            dm_env.TimeStep(
                timestep.step_type, timestep.reward, timestep.discount, new_observations
            ),
            extras,
        )

    def observation_spec(self) -> Dict[str, OLT]:
        """Observation spec.

        Returns:
            types.Observation: spec for environment.
        """
        timestep, extras = self.reset()
        observations = timestep.observation
        return observations

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment.

        Args:
            name (str): attribute.

        Returns:
            Any: return attribute from env or underlying env.
        """
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)


class ConcatPrevActionToObservation:
    """Concat one-hot vector of agent prev_action to obs.

    We assume the environment has discreet actions.

    TODO (Claude) support continuous actions.
    """

    def __init__(self, environment: Any):
        self._environment = environment

    def reset(self) -> dm_env.TimeStep:
        """Reset the environment and add zero action."""
        timestep, extras = self._environment.reset()
        old_observations = timestep.observation
        action_spec = self._environment.action_spec()
        new_observations = {}
        # TODO double check this, because possible agents could shrink
        for agent in self._environment.possible_agents:
            agent_olt = old_observations[agent]
            agent_observation = agent_olt.observation
            agent_one_hot_action = np.zeros(
                action_spec[agent].num_values, dtype=np.float32
            )

            new_observations[agent] = OLT(
                observation=np.concatenate([agent_one_hot_action, agent_observation]),
                legal_actions=agent_olt.legal_actions,
                terminal=agent_olt.terminal,
            )

        return (
            dm_env.TimeStep(
                timestep.step_type, timestep.reward, timestep.discount, new_observations
            ),
            extras,
        )

    def step(self, actions: Dict) -> dm_env.TimeStep:
        """Step the environment and concat prev actions."""
        timestep, extras = self._environment.step(actions)
        old_observations = timestep.observation
        action_spec = self._environment.action_spec()
        new_observations = {}
        for agent in self._environment.possible_agents:
            agent_olt = old_observations[agent]
            agent_observation = agent_olt.observation
            agent_one_hot_action = np.zeros(
                action_spec[agent].num_values, dtype=np.float32
            )
            agent_one_hot_action[actions[agent]] = 1

            new_observations[agent] = OLT(
                observation=np.concatenate([agent_one_hot_action, agent_observation]),
                legal_actions=agent_olt.legal_actions,
                terminal=agent_olt.terminal,
            )

        return (
            dm_env.TimeStep(
                timestep.step_type, timestep.reward, timestep.discount, new_observations
            ),
            extras,
        )

    def observation_spec(self) -> Dict[str, OLT]:
        """Observation spec.

        Returns:
            types.Observation: spec for environment.
        """
        timestep, extras = self.reset()
        observations = timestep.observation
        return observations

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment.

        Args:
            name (str): attribute.

        Returns:
            Any: return attribute from env or underlying env.
        """
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)
