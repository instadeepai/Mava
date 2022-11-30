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

"""Wraps a PettingZoo MARL environment to be used as a dm_env environment."""

from typing import Any, Dict, List, Optional, Union
import copy
import dm_env
import jax
import numpy as np
from acme import specs
from acme.wrappers.gym_wrapper import _convert_to_spec
from gym import spaces

from mava import types
from mava.utils.wrapper_utils import (
    apply_env_wrapper_preprocessors,
    convert_dm_compatible_observations,
    convert_np_type,
    parameterized_restart,
)
from mava.wrappers.env_wrappers import ParallelEnvWrapper


class DoubleIndependentGym(ParallelEnvWrapper):
    """Environment wrapper for gym environments."""

    def __init__(
        self,
        environment,
    ):
        """Constructor for double independent gym environment."""

        self._environment_0 = environment
        self._environment_1 = copy.deepcopy(environment)
        self._agents = ["agent_0", "agent_1"]

    def reset(self) -> dm_env.TimeStep:
        """Resets the env.

        Returns:
            dm_env.TimeStep: dm timestep.
        """

        self._reset_next_step = False
        self._step_type = dm_env.StepType.FIRST
        discount_spec = self.discount_spec()
        self._done = False

        obs_0 = self._environment_0.reset()
        obs_1 = self._environment_1.reset()
        obs = {"agent_0": obs_0, "agent_1": obs_1}

        self._discounts = {
            agent: convert_np_type(discount_spec[agent].dtype, 1)
            for agent in self.possible_agents
        }

        observations = self._convert_observations(
            obs, {agent: False for agent in self.possible_agents}
        )
        rewards_spec = self.reward_spec()
        rewards = {
            agent: convert_np_type(rewards_spec[agent].dtype, 0)
            for agent in self.possible_agents
        }

        return parameterized_restart(rewards, self._discounts, observations)

    def step(self, actions: Dict[str, np.ndarray]) -> dm_env.TimeStep:
        """Steps in env.

        Args:
            actions (Dict[str, np.ndarray]): actions per agent.

        Returns:
            dm_env.TimeStep: dm timestep
        """

        if self._reset_next_step:
            return self.reset()

        # Convert Jax device array actions to python integers
        if not all(type(value) == int for value in actions.values()):  # type: ignore
            actions = jax.tree_map(lambda x: x.tolist(), actions)

        obs_0, rew_0, done_0, infos = self._environment_0.step(actions["agent_0"])
        obs_1, rew_1, done_1, infos = self._environment_1.step(actions["agent_1"])

        rew = {"agent_0": rew_0, "agent_1": rew_1}
        obs = {"agent_0": obs_0, "agent_1": obs_1}
        dones = {"agent_0": done_0, "agent_1": done_1}

        obs = self._convert_observations(obs, dones)
        rew = self._convert_rewards(rew, dones)

        self._done = all(dones.values())

        if self._done:
            self._step_type = dm_env.StepType.LAST
            self._reset_next_step = True
            # Terminal discount should be 0.0 as per dm_env
            discount = {
                agent: convert_np_type(self.discount_spec()[agent].dtype, 0.0)
                for agent in self.possible_agents
            }
        else:
            self._step_type = dm_env.StepType.MID
            discount = self._discounts

        timestep = dm_env.TimeStep(
            observation=obs,
            reward=rew,
            discount=discount,
            step_type=self._step_type,
        )

        return timestep, {}

    def env_done(self):
        return self._done

    def extras_spec(self) -> Dict[str, specs.BoundedArray]:
        """Function returns extra spec (format) of the env.

        Returns:
            Dict[str, specs.BoundedArray]: extra spec.
        """
        return {}

    def _convert_rewards(self, rewards, dones):
        for agent, reward in rewards.items():
            rewards[agent] = np.array(reward, "float32") if not dones[agent] else np.array(0.0, "float32")
        return rewards

    def _convert_observations(
        self, observes: Dict[str, np.ndarray], dones: Dict[str, bool]
    ) -> types.Observation:
        """Convert PettingZoo observation so it's dm_env compatible.

        Args:
            observes (Dict[str, np.ndarray]): observations per agent.
            dones (Dict[str, bool]): dones per agent.

        Returns:
            types.Observation: dm compatible observations.
        """

        for agent, obs in observes.items():
            observes[agent] = observes[agent] if not dones[agent] else np.zeros_like(observes[agent])


        return convert_dm_compatible_observations(
            observes,
            dones,
            self.observation_spec(),
            all(dones.values()),
            self.possible_agents,
        )

    def observation_spec(self) -> Dict[str, types.OLT]:
        """Observation spec.

        Returns:
            types.Observation: spec for environment.
        """
        observation_specs = {}
        for agent in self.possible_agents:
            if type(self._environment_0.observation_space) == spaces.Box:
                observation = _convert_to_spec(
                    self._environment_0.observation_space
                )

                action_space = self._environment_0.action_space
                if type(action_space) == spaces.Discrete:
                    # legal action mask should be a vector of ones and zeros
                    legal_actions = specs.BoundedArray(
                        shape=(action_space.n,),
                        dtype=action_space.dtype,
                        minimum=np.zeros(action_space.shape),
                        maximum=np.zeros(action_space.shape) + 1,
                        name=None,
                    )
                else:
                    legal_actions = _convert_to_spec(
                        self._environment_0.action_space
                    )
            else:
                # For env like SC2 with action mask spec
                observation = _convert_to_spec(
                    self._environment_0.observation_space
                )
                legal_actions = _convert_to_spec(
                    self._environment_0.observation_space
                )

            observation_specs[agent] = types.OLT(
                observation=observation,
                legal_actions=legal_actions,
                terminal=specs.Array((1,), np.float32),
            )

        return observation_specs

    def action_spec(self) -> Dict[str, Union[specs.DiscreteArray, specs.BoundedArray]]:
        """Action spec.

        Returns:
            Dict[str, Union[specs.DiscreteArray, specs.BoundedArray]]: spec for actions.
        """
        action_specs = {}
        action_space = self._environment_0.action_space
        action_space.dtype = np.int32
        for agent in self.possible_agents:
            action_specs[agent] = _convert_to_spec(action_space)
        return action_specs

    def reward_spec(self) -> Dict[str, specs.Array]:
        """Reward spec.

        Returns:
            Dict[str, specs.Array]: spec for rewards.
        """
        reward_specs = {}
        for agent in self.possible_agents:
            reward_specs[agent] = specs.Array((), np.float32)

        return reward_specs

    def discount_spec(self) -> Dict[str, specs.BoundedArray]:
        """Discount spec.

        Returns:
            Dict[str, specs.BoundedArray]: spec for discounts.
        """
        discount_specs = {}
        for agent in self.possible_agents:
            discount_specs[agent] = specs.BoundedArray(
                (), np.float32, minimum=0, maximum=1.0
            )
        return discount_specs

    @property
    def agents(self) -> List:
        """Agents still alive in env (not done).

        Returns:
            List: alive agents in env.
        """
        return self._agents

    @property
    def possible_agents(self) -> List:
        """All possible agents in env.

        Returns:
            List: all possible agents in env.
        """
        return self._agents

    @property
    def environment(self) -> "ParallelEnv":
        """Returns the wrapped environment.

        Returns:
            ParallelEnv: parallel env.
        """
        return self._environment_0

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
            return getattr(self._environment_0, name)
