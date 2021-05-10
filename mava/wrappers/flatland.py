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

"""Wraps a Flatland MARL environment to be used as a dm_env environment."""
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Union

import dm_env
import numpy as np
from acme import specs
from acme.wrappers.gym_wrapper import _convert_to_spec
from flatland.envs.observations import GlobalObsForRailEnv, TreeObsForRailEnv
from flatland.envs.rail_env import RailEnv
from gym.spaces import Discrete

from mava.types import OLT, Observation
from mava.utils.wrapper_utils import convert_np_type, parameterized_restart
from mava.wrappers.env_wrappers import ParallelEnvWrapper
from mava.wrappers.flatland_wrapper_utils import (
    _agent_info_spec,
    _decorate_step_method,
    _get_agent_handle,
    _get_agent_id,
    _infer_observation_space,
    normalize_observation,
)


class FlatlandEnvWrapper(ParallelEnvWrapper):
    """Environment wrapper for Flatland environments.

    All environments would require an observation preprocessor, except for
    'GlobalObsForRailEnv'. This is because flatland gives users the
    flexibility of designing custom observation builders. 'TreeObsForRailEnv'
    would use the normalize_observation function from the flatland baselines
    if none is supplied.

    The supplied preprocessor should return either an array, tuple of arrays or
    a dictionary of arrays for an observation input.

    The obervation, for an agent, returned by this wrapper could consist of both
    the agent observation and agent info. This is because flatland also provides
    informationn about the agents at each step. This information include;
    'action_required', 'malfunction', 'speed', and 'status', and it can be appended
    to the observation, by this wrapper, as an array. action_required is a boolean,
    malfunction is an int denoting the number of steps for which the agent would
    remain motionless, speed is a float and status can be any of the below;

    READY_TO_DEPART = 0
    ACTIVE = 1
    DONE = 2
    DONE_REMOVED = 3

    This would be included in the observation if agent_info is set to True
    """

    # Note: we don't inherit from base.EnvironmentWrapper because that class
    # assumes that the wrapped environment is a dm_env.Environment.
    def __init__(
        self,
        environment: RailEnv,
        preprocessor: Callable[
            [Any], Union[np.ndarray, Tuple[np.ndarray], Dict[str, np.ndarray]]
        ] = None,
        agent_info: bool = True,
    ):
        self._environment = environment
        self._environment.aec_env = self
        _decorate_step_method(self._environment)

        self._agents = [_get_agent_id(i) for i in range(self.num_agents)]
        self._possible_agents = self.agents[:]

        self._reset_next_step = True
        self._step_type = dm_env.StepType.FIRST
        self.num_actions = 5
        self.action_spaces = {
            agent: Discrete(self.num_actions) for agent in self.possible_agents
        }

        # preprocessor must be for observation builders other than global obs
        # treeobs builders would use the default preprocessor if none is
        # supplied
        self.preprocessor: Callable[
            [Dict[int, Any]], Dict[int, Any]
        ] = self._obtain_preprocessor(preprocessor)

        self._include_agent_info = agent_info

        # observation space:
        # flatland defines no observation space for an agent. Here we try
        # to define the observation space. All agents are identical and would
        # have the same observation space.
        # Infer observation space based on returned observation
        obs, _ = self._environment.reset()
        obs = self.preprocessor(obs)
        self.observation_spaces = {
            _get_agent_id(i): _infer_observation_space(ob) for i, ob in obs.items()
        }

    @property
    def agents(self) -> List[str]:
        return self._agents

    @property
    def possiple_agents(self) -> List[str]:
        return self._possible_agents

    @property
    def env_done(self) -> bool:
        return self._environment.dones["__all__"] or not self.agents

    def reset(self) -> dm_env.TimeStep:
        """Resets the episode."""
        self._reset_next_step = False
        self._agents = self.possible_agents[:]
        self._discounts = {
            agent: np.dtype("float32").type(1.0) for agent in self.agents
        }
        observe, info = self._environment.reset()
        observations = self._create_observations(observe, info, self._environment.dones)
        rewards_spec = self.reward_spec()
        rewards = {
            agent: convert_np_type(rewards_spec[agent].dtype, 0)
            for agent in self.possible_agents
        }

        discount_spec = self.discount_spec()
        self._discounts = {
            agent: convert_np_type(discount_spec[agent].dtype, 1)
            for agent in self.possible_agents
        }
        return parameterized_restart(rewards, self._discounts, observations)

    def step(self, actions: Dict[str, np.ndarray]) -> dm_env.TimeStep:
        """Steps the environment."""
        self._pre_step()

        if self._reset_next_step:
            return self.reset()

        self._agents = [
            agent
            for agent in self.agents
            if not self._environment.dones[_get_agent_handle(agent)]
        ]

        observations, rewards, dones, infos = self._environment.step(actions)

        rewards_spec = self.reward_spec()
        #  Handle empty rewards
        if not rewards:
            rewards = {
                agent: convert_np_type(rewards_spec[agent].dtype, 0)
                for agent in self.possible_agents
            }
        else:
            rewards = {
                _get_agent_id(agent): convert_np_type(
                    rewards_spec[_get_agent_id(agent)].dtype, reward
                )
                for agent, reward in rewards.items()
            }

        if observations:
            observations = self._create_observations(observations, infos, dones)

        if self.env_done:
            step_type = self._step_type
            self._step_type = dm_env.StepType.LAST
        else:
            step_type = dm_env.StepType.MID
            self._step_type = step_type

        if step_type == dm_env.StepType.LAST:
            self._reset_next_step = True
            rewards = {
                agent: np.dtype("float32").type(0) for agent, _ in rewards.items()
            }

        return dm_env.TimeStep(
            observation=observations,
            reward=rewards,
            discount=self._discounts,
            step_type=step_type,
        )

    # Convert Flatland observation so it's dm_env compatible. Also, the list
    # of legal actions must be converted to a legal actions mask.
    def _convert_observations(
        self, observes: Dict[str, Tuple[np.array, np.ndarray]], dones: Dict[str, bool]
    ) -> Observation:
        observations: Dict[str, OLT] = {}
        for agent, observation in observes.items():
            if isinstance(observation, dict) and "action_mask" in observation:
                legals = observation["action_mask"]
                observation = observation["observation"]
            else:
                # TODO Handle legal actions better for continous envs,
                #  maybe have min and max for each action and clip the agents actions
                #  accordingly
                legals = np.ones(
                    self.action_spaces[agent].shape,
                    dtype=self.action_spaces[agent].dtype,
                )
            observations[agent] = OLT(
                observation=observation,
                legal_actions=legals,
                terminal=np.asarray([dones[agent]], dtype=np.float32),
            )

        return observations

    # collate agent info and observation into a tuple, making the agents obervation to
    # be a tuple of the observation from the env and the agent info
    def _collate_obs_and_info(
        self, observes: Dict[int, np.ndarray], info: Dict[str, Dict[int, Any]]
    ) -> Dict[str, Tuple[np.array, np.ndarray]]:
        observations: Dict[str, Tuple[np.array, np.ndarray]] = {}
        observes = self.preprocessor(observes)
        for agent, obs in observes.items():
            agent_id = _get_agent_id(agent)
            agent_info = np.array(
                [info[k][agent] for k in info.keys()], dtype=np.float32
            )
            obs = (obs, agent_info) if self._include_agent_info else obs
            observations[agent_id] = obs

        return observations

    def _create_observations(
        self,
        obs: Dict[int, np.ndarray],
        info: Dict[str, Dict[int, Any]],
        dones: Dict[int, bool],
    ) -> Observation:
        observations_ = self._collate_obs_and_info(obs, info)
        dones_ = {_get_agent_id(k): v for k, v in dones.items()}
        observations = self._convert_observations(observations_, dones_)
        return observations

    def _obtain_preprocessor(
        self, preprocessor: Any
    ) -> Callable[[Dict[int, Any]], Dict[int, np.ndarray]]:
        """Obtains the actual preprocessor to be used based on the supplied
        preprocessor and the env's obs_builder object"""
        if not isinstance(self.obs_builder, GlobalObsForRailEnv):
            _preprocessor = preprocessor if preprocessor else lambda x: x
            if isinstance(self.obs_builder, TreeObsForRailEnv):
                _preprocessor = (
                    partial(
                        normalize_observation, tree_depth=self.obs_builder.max_depth
                    )
                    if not preprocessor
                    else preprocessor
                )
            assert _preprocessor is not None
        else:

            def _preprocessor(
                x: Tuple[np.ndarray, np.ndarray, np.ndarray]
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
                return x

        def returned_preprocessor(obs: Dict[int, Any]) -> Dict[int, np.ndarray]:
            temp_obs = {}
            for agent_id, ob in obs.items():
                temp_obs[agent_id] = _preprocessor(ob)
            return temp_obs

        return returned_preprocessor

    # set all parameters that should be available before an environment step
    # if no available agent, then environment is done and should be reset
    def _pre_step(self) -> None:
        if not self.agents:
            self._step_type = dm_env.StepType.LAST

    def observation_spec(self) -> Dict[str, OLT]:
        observation_specs = {}
        for agent in self.agents:
            observation_specs[agent] = OLT(
                observation=tuple(
                    (
                        _convert_to_spec(self.observation_spaces[agent]),
                        _agent_info_spec(),
                    )
                )
                if self._include_agent_info
                else _convert_to_spec(self.observation_spaces[agent]),
                legal_actions=_convert_to_spec(self.action_spaces[agent]),
                terminal=specs.Array((1,), np.float32),
            )
        return observation_specs

    def action_spec(self) -> Dict[str, Union[specs.DiscreteArray, specs.BoundedArray]]:
        action_specs = {}
        action_spaces = self.action_spaces
        for agent in self.possible_agents:
            action_specs[agent] = _convert_to_spec(action_spaces[agent])
        return action_specs

    def reward_spec(self) -> Dict[str, specs.Array]:
        reward_specs = {}
        for agent in self.possible_agents:
            reward_specs[agent] = specs.Array((), np.float32)
        return reward_specs

    def discount_spec(self) -> Dict[str, specs.BoundedArray]:
        discount_specs = {}
        for agent in self.possible_agents:
            discount_specs[agent] = specs.BoundedArray(
                (), np.float32, minimum=0, maximum=1.0
            )
        return discount_specs

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        return {}

    def seed(self, seed: int = None) -> None:
        self._environment._seed(seed)

    @property
    def environment(self) -> RailEnv:
        """Returns the wrapped environment."""
        return self._environment

    @property
    def num_agents(self) -> int:
        """Returns the number of trains/agents in the flatland environment"""
        return int(self._environment.number_of_agents)

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        return getattr(self._environment, name)
