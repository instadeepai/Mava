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

from typing import Any, Dict, Iterator, List, Tuple, Union

import dm_env
import numpy as np
import pyspiel  # type: ignore
from acme import specs
from gym.spaces import Discrete
from gym.spaces.box import Box
from open_spiel.python import rl_environment  # type: ignore

from mava import types
from mava.utils.wrapper_utils import convert_np_type, parameterized_restart
from mava.wrappers.env_wrappers import SequentialEnvWrapper


class OpenSpielSequentialWrapper(SequentialEnvWrapper):
    def __init__(
        self,
        environment: rl_environment.Environment,
    ):
        self._environment = environment
        self._possible_agents = [
            f"player_{i}" for i in range(self._environment.num_players)
        ]
        self._agents = self._possible_agents[:]
        self.num_actions = self._environment.action_spec()["num_actions"]
        self.info_state = self._environment.observation_spec()["info_state"]
        self.action_spaces = {
            agent: Discrete(self.num_actions) for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: Box(0, 1, self.info_state, dtype=np.float64)
            for agent in self.possible_agents
        }

    def reset(self) -> dm_env.TimeStep:
        """Resets the episode."""
        self._reset_next_step = False
        self.dones = {agent: False for agent in self.possible_agents}
        self._agents = self._possible_agents[:]

        self._prev_timestep: rl_environment.TimeStep = None
        self._current_player_id = 0

        opnspl_tmstep = self._environment.reset()
        agent = self.current_agent
        done = self.dones[agent]

        observe = self._to_observation(opnspl_tmstep)
        observation = self._convert_observation(agent, observe, done)

        self._discount = convert_np_type(self.discount_spec()[agent].dtype, 1)

        reward = convert_np_type(
            self.reward_spec()[agent].dtype,
            (
                opnspl_tmstep.rewards[self.current_player_id]
                if opnspl_tmstep.rewards
                else 0
            ),
        )

        return parameterized_restart(reward, self._discount, observation)

    def _to_observation(
        self,
        timestep: rl_environment.TimeStep,
    ) -> Dict[str, np.ndarray]:

        obs = np.array(
            timestep.observations["info_state"][self.current_player_id], np.float32
        )

        action_mask = np.zeros((self._environment.action_spec()["num_actions"],))
        action_mask[timestep.observations["legal_actions"][self.current_player_id]] = 1

        observation = {
            "observation": obs,
            "action_mask": action_mask,
        }

        return observation

    def step(self, action_list: Tuple[np.ndarray]) -> dm_env.TimeStep:
        """Steps the environment."""
        if self._reset_next_step:
            return self.reset()

        # only action lists are accepted
        if not isinstance(action_list, (list, tuple)):
            action_list = [action_list]

        agent = self.current_agent

        # done agents should be removed and active agents should take steps
        if self.dones[agent]:
            self.agents.remove(agent)
            del self.dones[agent]

            # move to next agent, which should also be done
            self._current_player_id = (self._current_player_id + 1) % self.num_agents
            agent = self.current_agent

            opnspl_timestep = self._prev_timestep

            step_type = dm_env.StepType.LAST

        else:
            opnspl_timestep = self._environment.step(action_list)

            # after a step, a next agent becomes the current
            agent = self.current_agent

            if (
                self._environment.get_state.current_player()
                == pyspiel.PlayerId.TERMINAL
            ):
                # all agents get done at a terminal step in turn-based games
                # current agent/player is updated using _current_player_id
                self.dones = {agnt: True for agnt in self._possible_agents}
                self._current_player_id = (
                    self._current_player_id + 1
                ) % self.num_agents

                agent = self.current_agent
            else:
                self.dones[agent] = False

            step_type = (
                dm_env.StepType.LAST if self.dones[agent] else dm_env.StepType.MID
            )
            self._prev_timestep = opnspl_timestep

        observe = self._to_observation(opnspl_timestep)

        # Reset if all agents are done
        if self.env_done():
            self._reset_next_step = True
            reward = convert_np_type(
                self.reward_spec()[agent].dtype,
                0,
            )
            observation = self._convert_observation(agent, observe, True)
        else:
            #  observation for next agent
            reward = convert_np_type(
                self.reward_spec()[agent].dtype,
                opnspl_timestep.rewards[self.current_player_id],
            )
            observation = self._convert_observation(agent, observe, self.dones[agent])

        return dm_env.TimeStep(
            observation=observation,
            reward=reward,
            discount=self._discount,
            step_type=step_type,
        )

    def env_done(self) -> bool:
        return not self.agents

    def agent_iter(self, max_iter: int = 2 ** 63) -> Iterator:
        return AgentIterator(self, max_iter)

    # Convert OpenSpiel observation so it's dm_env compatible. Also, the list
    # of legal actions must be converted to a legal actions mask.
    def _convert_observation(  # type: ignore[override]
        self, agent: str, observe: Union[dict, np.ndarray], done: bool
    ) -> types.OLT:
        if isinstance(observe, dict):
            legals = np.array(observe["action_mask"], np.float32)
            observation = np.array(observe["observation"])
        else:
            legals = np.ones(self.num_actions, np.float32)
            observation = np.array(observe)

        observation = types.OLT(
            observation=observation,
            legal_actions=legals,
            terminal=np.asarray([done], dtype=np.float32),
        )

        return observation

    def observation_spec(self) -> types.Observation:
        observation_specs = {}
        for agent in self.possible_agents:
            spec = self._environment.observation_spec()
            observation_specs[agent] = types.OLT(
                observation=specs.Array(spec["info_state"], np.float32),
                legal_actions=specs.Array(spec["legal_actions"], np.float32),
                terminal=specs.Array((1,), np.float32),
            )
        return observation_specs

    def action_spec(self) -> Dict[str, specs.DiscreteArray]:
        action_specs = {}
        for agent in self.possible_agents:
            spec = self._environment.action_spec()
            action_specs[agent] = specs.DiscreteArray(spec["num_actions"], np.int64)
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

    @property
    def agents(self) -> List:
        return self._agents

    @property
    def possible_agents(self) -> List:
        return self._possible_agents

    @property
    def environment(self) -> rl_environment.Environment:
        """Returns the wrapped environment."""
        return self._environment

    @property
    def current_agent(self) -> str:

        agent = self.possible_agents[self.current_player_id]

        return agent

    @property
    def current_player_id(self) -> int:
        if self._environment.get_state.current_player() == pyspiel.PlayerId.TERMINAL:
            return self._current_player_id
        else:
            p_id = self._environment.get_state.current_player()
            self._current_player_id = p_id
            return p_id

    @property
    def num_agents(self) -> int:
        return len(self.possible_agents)

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)


class AgentIterator:
    def __init__(self, env: OpenSpielSequentialWrapper, max_iter: int) -> None:
        self.env = env
        self.countdown = max_iter

    def __iter__(self) -> Any:
        return self

    def __next__(self) -> str:
        if not self.env.agents or self.countdown <= 0:
            raise StopIteration
        self.countdown -= 1
        return self.env.current_agent
