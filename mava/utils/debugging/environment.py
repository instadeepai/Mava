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

# Adapted from https://github.com/openai/multiagent-particle-envs.
# TODO (dries): Try using this class directly from PettingZoo and delete this file.

import copy
from typing import Any, Callable, Dict, List, Tuple, Union

import gym
import numpy as np
from gym import spaces

from mava.utils.debugging.core import Agent, World

from .multi_discrete import MultiDiscrete


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        world: World,
        action_space: str,
        reset_callback: Callable = None,
        reward_callback: Callable = None,
        observation_callback: Callable = None,
        info_callback: Callable = None,
        done_callback: Callable = None,
        shared_viewer: bool = True,
    ) -> None:

        self.world = world
        # Generate IDs and convert agent list to dictionary format.
        agent_dict = {}
        self.env_done = False
        self.agent_ids = []

        agent_list = self.world.policy_agents
        for a_i in range(len(agent_list)):
            agent_id = "agent_" + str(a_i)
            self.agent_ids.append(agent_id)
            agent_dict[agent_id] = agent_list[a_i]

        self.possible_agents = self.agent_ids

        self.num_agents = len(self.agent_ids)
        self.agents = agent_dict

        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        assert action_space in ["continuous", "discrete"]

        self.discrete_action_space = action_space == "discrete"

        # if true, action is a number 0...N, otherwise
        # action is a one-hot N-dimensional vector
        self.discrete_action_input = action_space == "discrete"
        self.time = 0

        self.render_geoms: Union[List, None] = None
        self.render_geoms_xform: Union[List, None] = None

        # configure spaces
        self.action_spaces = {}
        self.observation_spaces = {}
        for a_i, agent_id in enumerate(self.agent_ids):
            agent = self.agents[agent_id]
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(
                    low=-agent.u_range,
                    high=+agent.u_range,
                    shape=(world.dim_p,),
                    dtype=np.float32,
                )
            if agent.movable:
                total_action_space.append(u_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to
                # MultiDiscrete action space
                if all(
                    isinstance(act_space, spaces.Discrete)
                    for act_space in total_action_space
                ):
                    act_space = MultiDiscrete(
                        [[0, act_space.n - 1] for act_space in total_action_space]
                    )
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_spaces[agent_id] = act_space
            else:
                self.action_spaces[agent_id] = total_action_space[0]
            # observation space
            if observation_callback is not None:
                obs_dim = len(observation_callback(agent, a_i, self.world))
            else:
                raise ValueError("Observation callback is None.")
            self.observation_spaces[agent_id] = spaces.Box(
                low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32
            )

        # rendering
        self.shared_viewer = shared_viewer
        self.viewers: List = [None]
        if not self.shared_viewer:
            self.viewers = [None] * self.n
        self._reset_render()

    def step(
        self, action_n: Dict[str, Union[int, List[float]]]
    ) -> Tuple[
        Dict[str, Union[np.array, Any]],
        Union[dict, Dict[str, Union[float, Any]]],
        Dict[str, Any],
        Dict[str, dict],
    ]:
        obs_n = {}
        reward_n = {}
        done_n = {}
        # set action for each agent
        for agent_id in self.agent_ids:
            agent = self.agents[agent_id]
            agent_action = copy.deepcopy(action_n[agent_id])
            self._set_action(agent_action, agent, self.action_spaces[agent_id])
        # advance world state
        self.world.step()
        # record observation for each agent
        for a_i, agent_id in enumerate(self.agent_ids):
            agent = self.agents[agent_id]
            obs_n[agent_id] = self._get_obs(a_i, agent)
            reward_n[agent_id] = self._get_reward(a_i, agent)
            done_n[agent_id] = self._get_done(agent)

            if done_n[agent_id]:
                self.env_done = True

        state_n = self._get_state()

        return obs_n, reward_n, done_n, state_n

    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # reset world
        if self.reset_callback is not None:
            self.reset_callback(self.world)
        else:
            raise ValueError("self.reset_callback is still None!")
        # reset renderer
        self._reset_render()
        self.env_done = False
        # record observations for each agent
        obs_n = {}
        for a_i, agent_id in enumerate(self.agent_ids):
            agent = self.agents[agent_id]
            obs_n[agent_id] = self._get_obs(a_i, agent)
        state_n = self._get_state()
        return obs_n, {"s_t": state_n}

    # get info used for benchmarking
    def _get_info(self, agent: Agent) -> Dict:
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, a_i: int, agent: Agent) -> np.array:
        if self.observation_callback is None:
            return np.zeros(0)
        else:
            return self.observation_callback(agent, a_i, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent: Agent) -> bool:
        if self.done_callback is None:
            raise ValueError("No done callback specified.")
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, a_i: int, agent: Agent) -> float:
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, a_i, self.world)

    def _get_state(self) -> np.array:
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in self.world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos)
        # entity colors

        agent_pos = []
        agent_vel = []
        for i, agent in enumerate(self.world.agents):
            agent_pos.append(agent.state.p_pos)
            agent_vel.append(agent.state.p_vel)

        return np.array(
            np.concatenate(
                [[self.world.current_step / 50]] + entity_pos + agent_pos + agent_vel
            ),
            dtype=np.float32,
        )

    # set env action for a particular agent
    def _set_action(
        self,
        action: np.array,
        agent: Agent,
        action_space: spaces,
    ) -> None:
        agent.action.u = np.zeros(self.world.dim_p)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index : (index + s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1:
                    agent.action.u[0] = -1.0
                if action[0] == 2:
                    agent.action.u[0] = +1.0
                if action[0] == 3:
                    agent.action.u[1] = -1.0
                if action[0] == 4:
                    agent.action.u[1] = +1.0
            else:
                # if self.force_discrete_action:
                #     d = np.argmax(action[0])
                #     action[0][:] = 0.0
                #     action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    agent.action.u = action[0]
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self) -> None:
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode: str = "human") -> List[np.array]:

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't
                # import for headless machines)
                # from gym.envs.classic_control import rendering
                from . import rendering

                self.viewers[i] = rendering.Viewer(700, 700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import
            # for headless machines)
            # from gym.envs.classic_control import rendering
            from . import rendering

            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()

                if entity.color is not None:
                    r, g, b = entity.color
                    if "agent" in entity.name:
                        geom.set_color(r, g, b, alpha=0.5)
                    else:
                        geom.set_color(r, g, b)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[self.agent_ids[i]].state.p_pos
            self.viewers[i].set_bounds(
                pos[0] - cam_range,
                pos[0] + cam_range,
                pos[1] - cam_range,
                pos[1] + cam_range,
            )
            # update geometry positions
            if self.render_geoms_xform is not None:
                for e, entity in enumerate(self.world.entities):
                    self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            else:
                raise ValueError("self.render_geoms_xform is still None!")
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array=mode == "rgb_array"))

        return np.squeeze(results)

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self) -> List[np.array]:
        receptor_type = "polar"
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == "polar":
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == "grid":
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x, y]))
        return dx
