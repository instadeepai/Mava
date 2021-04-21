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

"""Wraps a RLLib Multi-Agent environment to be used as a dm_env environment."""
from typing import Any, Dict

import dm_env
import numpy as np
from acme import specs
from acme.wrappers.gym_wrapper import _convert_to_spec
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from mava import types
from mava.utils.wrapper_utils import convert_np_type, parameterized_restart
from mava.wrappers.pettingzoo import PettingZooParallelEnvWrapper


class RLLibMultiAgentEnvWrapper(PettingZooParallelEnvWrapper):
    """Environment wrapper for RLLib MA environments."""

    # Note: we don't inherit from base.EnvironmentWrapper because that class
    # assumes that the wrapped environment is a dm_env.Environment.
    def __init__(self, environment: MultiAgentEnv):
        self._environment = environment
        self._reset_next_step = True
        self.num_agents = len(environment.agents)
        self.possible_agents = [f"agent_{x}" for x in range(self.num_agents)]
        self.agents = [f"agent_{x}" for x in range(self.num_agents)]
        self.action_spaces = {agent: environment.action_space for agent in self.agents}
        self.observation_spaces = {
            agent: environment.observation_space for agent in self.agents
        }

    def reset(self) -> dm_env.TimeStep:
        """Resets the episode."""
        self._reset_next_step = False
        self.done = False
        self._step_type = dm_env.StepType.FIRST

        obs = self._environment.reset()
        done = {k: False for k in obs.keys()}
        observation: dict = self._convert_observation(obs, done)

        self._discount: dict = {}
        reward: dict = {}
        for agent, discount in self.discount_spec().items():
            self._discount[agent] = convert_np_type(np.float32, 1)
            reward[agent] = convert_np_type(np.float32, 0)

        return parameterized_restart(reward, self._discount, observation)

    def step(self, action: dict) -> dm_env.TimeStep:
        """Steps the environment."""
        if self._reset_next_step:
            return self.reset()

        action = {eval(k.split("_")[1]): v for k, v in action.items()}
        observe, reward, done, info = self._environment.step(action)
        if observe:
            observe = self._convert_observation(observe, done)

        if reward:
            reward = {
                f"agent_{k}": np.asarray(v, dtype=np.float32) for k, v in reward.items()
            }
        else:
            reward = {
                agent: np.asarray(0, dtype=np.float32) for agent in self.possible_agents
            }

        if done:
            self.done = done["__all__"]

        if self.done:
            self._step_type = dm_env.StepType.LAST
            self._reset_next_step = True
        else:
            self._step_type = dm_env.StepType.MID

        return dm_env.TimeStep(
            observation=observe,
            reward=reward,
            discount=self._discount,
            step_type=self._step_type,
        )

    # Convert RLLib multi-agent observation to it's dm_env compatible. Also, the list
    # of legal actions must be converted to a legal actions mask.
    def _convert_observation(self, observe: dict, done: dict) -> dict:

        observe = {
            f"agent_{k}": np.asarray(v, dtype=np.float32) for k, v in observe.items()
        }

        dones: dict = {
            f"agent_{k}": np.asarray([v], dtype=np.float32) for k, v in done.items()
        }

        observation: dict = {}
        for agent in observe.keys():
            legals = np.ones(
                self._environment.action_space.shape,
                dtype=self._environment.action_space.dtype,
            )

            observation[agent] = types.OLT(
                observation=observe[agent],
                legal_actions=legals,
                terminal=dones[agent],
            )

        return observation

    def observation_spec(self) -> Dict[str, types.OLT]:
        observation_specs = {}
        for i in range(len(self._environment.agents)):
            observation_specs[f"agent_{i}"] = types.OLT(
                observation=_convert_to_spec(self._environment.observation_space),
                legal_actions=_convert_to_spec(self._environment.action_space),
                terminal=specs.Array((1,), np.float32),
            )
        return observation_specs

    def action_spec(self) -> Dict[str, specs.DiscreteArray]:
        action_specs = {}
        for i in range(len(self._environment.agents)):
            action_specs[f"agent_{i}"] = _convert_to_spec(
                self._environment.action_space
            )
        return action_specs

    def reward_spec(self) -> Dict[str, specs.Array]:
        reward_specs = {}
        for i in range(len(self._environment.agents)):
            reward_specs[f"agent_{i}"] = specs.Array((), np.float32)

        return reward_specs

    def discount_spec(self) -> Dict[str, specs.BoundedArray]:
        discount_specs = {}
        for i in range(len(self._environment.agents)):
            discount_specs[f"agent_{i}"] = specs.BoundedArray(
                (), np.float32, minimum=0, maximum=1.0
            )
        return discount_specs

    def extra_spec(self) -> dict:
        return {}

    def seed(self, seed: int = None) -> None:
        environments = (
            self._environment.agents
        )  # RLLib stores envs as agents in MultiEnv
        for env in environments:
            env.seed(seed)

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        return getattr(self._environment, name)
