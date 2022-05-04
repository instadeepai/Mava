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
# type: ignore
import copy
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple, Union

import gym
import jax
import jax.numpy as jnp
import numpy as np
from acme import specs
from acme.wrappers.gym_wrapper import _convert_to_spec
from gym import spaces
from jax.random import PRNGKey
from numpy import ndarray

from mava.utils.debugging.environments.jax.simple_spread.core import (
    Action,
    Agent,
    EntityId,
    JaxWorld,
    step,
)
from mava.utils.debugging.multi_discrete import MultiDiscrete


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentJaxEnvBase(gym.Env, ABC):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        world: JaxWorld,
        reset_callback: Callable = None,
        reward_callback: Callable = None,
        observation_callback: Callable = None,
        info_callback: Callable = None,
        done_callback: Callable = None,
        shared_viewer: bool = True,
    ) -> None:

        self.dim_p = world.dim_p
        # I don't see where this was being used, but it would be static here so no point in having it
        # self.env_done = False

        # Generate IDs and convert agent list to dictionary format.
        self.agent_ids = []

        self.agent_ids = [
            EntityId(id=a_i, type=0) for a_i in range(len(world.agents.size))
        ]
        self.possible_agents = self.agent_ids
        self.num_agents = len(self.agent_ids)

        # set required vectorized gym env property
        self.n = len(world.agents.size)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback

        self.render_geoms: Union[List, None] = None
        self.render_geoms_xform: Union[List, None] = None

        # configure spaces
        self.action_spaces = {}
        self.observation_spaces = {}
        for a_i in range(self.n):
            agent_id = EntityId(id=a_i, type=0)
            agent = jax.tree_map(lambda x: x[a_i], world.agents)
            total_action_space = []
            # physical action space
            u_action_space = spaces.Discrete(self.dim_p * 2 + 1)

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
                obs_dim = len(observation_callback(agent, a_i, world))
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
        self, world: JaxWorld, action_n: Dict[str, Union[int, List[float]]]
    ) -> Tuple[
        JaxWorld,
        Tuple[
            Dict[EntityId, ndarray],
            Dict[EntityId, float],
            Dict[EntityId, bool],
            ndarray,
        ],
    ]:
        obs_n = {}
        reward_n = {}
        done_n = {}

        processed_agents = []
        # set action for each agent
        for agent_id in self.agent_ids:
            agent = jax.tree_map(lambda x: x[agent_id.id], world.agents)
            # agent = world.agents[agent_id.id]
            agent_action = self._process_action(action_n[agent_id], agent)
            processed_agents.append(agent.replace(action=agent_action))

        # stack pytrees
        processed_agents = jax.tree_map(lambda *x: jnp.stack(x), *processed_agents)
        # update world's agents with new actions
        world = world.replace(agents=processed_agents)
        # advance world state
        world = step(world)
        # record observation for each agent
        for a_i, agent_id in enumerate(self.agent_ids):
            agent = jax.tree_map(lambda x: x[agent_id.id], world.agents)
            obs_n[agent_id] = self._get_obs(world, a_i, agent)
            reward_n[agent_id] = self._get_reward(world, a_i, agent)
            done_n[agent_id] = self._get_done(world, agent)

        state_n = self._get_state(world)

        return world, (obs_n, reward_n, done_n, state_n)

    def reset(self, key: PRNGKey) -> Tuple[JaxWorld, Dict[str, Any], Dict[str, Any]]:
        # reset world
        if self.reset_callback is not None:
            world = self.reset_callback(key)
        else:
            raise ValueError("self.reset_callback is still None!")
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = {}
        for a_i, agent_id in enumerate(self.agent_ids):
            agent = jax.tree_map(lambda x: x[agent_id.id], world.agents)
            obs_n[agent_id] = self._get_obs(world, a_i, agent)
        state_n = self._get_state(world)
        return world, obs_n, {"s_t": state_n}

    # get info used for benchmarking
    def _get_info(self, world: JaxWorld, agent: Agent) -> Dict:
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, world)

    # get observation for a particular agent
    def _get_obs(self, world: JaxWorld, a_i: int, agent: Agent) -> np.ndarray:
        if self.observation_callback is None:
            return jnp.zeros(0)
        else:
            return self.observation_callback(agent, a_i, world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, world: JaxWorld, agent: Agent) -> bool:
        if self.done_callback is None:
            raise ValueError("No done callback specified.")
        return self.done_callback(agent, world)

    # get reward for a particular agent
    def _get_reward(self, world: JaxWorld, a_i: int, agent: Agent) -> float:
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, a_i, world)

    def _get_state(self, world: JaxWorld) -> np.ndarray:
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        # for entity in world.landmarks:  # world.entities:
        #     entity_pos.append(entity.state.p_pos)

        for i in range(len(world.landmarks.size)):
            entity = jax.tree_map(lambda x: x[i], world.landmarks)
            entity_pos.append(entity.state.p_pos)
        # entity colors

        agent_pos = []
        agent_vel = []
        for i in range(len(world.agents.size)):
            agent = jax.tree_map(lambda x: x[i], world.agents)
            agent_pos.append(agent.state.p_pos)
            agent_vel.append(agent.state.p_vel)

        # for i, agent in enumerate(world.agents):
        #     agent_pos.append(agent.state.p_pos)
        #     agent_vel.append(agent.state.p_vel)

        return jnp.array(
            jnp.concatenate(
                [jnp.array([world.current_step / 50])]
                + entity_pos
                + agent_pos
                + agent_vel
            ),
            dtype=np.float32,
        )

    # set env action for a particular agent
    @abstractmethod
    def _process_action(self, action: int, agent: Agent) -> Action:
        agent.action.u = jnp.zeros(self.dim_p)

        def on_movable(act: Action):
            sensitivity = jax.lax.cond(
                jnp.isnan(agent.accel), lambda: 5.0, lambda: agent.accel
            )

            return Action(
                u=jnp.array(
                    jax.lax.switch(
                        act - 1,
                        [
                            lambda x: [-1.0, 0.0],
                            lambda x: [1.0, 0.0],
                            lambda x: [0.0, -1.0],
                            lambda x: [0.0, 1.0],
                        ],
                        None,
                    )
                )
                * sensitivity
            )

        return jax.lax.cond(agent.movable, on_movable, lambda _: agent.action, action)

    # reset rendering assets
    def _reset_render(self) -> None:
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, world: JaxWorld, mode: str = "human") -> List[np.ndarray]:
        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't
                # import for headless machines)
                # from gym.envs.classic_control import rendering
                from mava.utils.debugging import rendering

                self.viewers[i] = rendering.Viewer(700, 700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import
            # for headless machines)
            # from gym.envs.classic_control import rendering
            from mava.utils.debugging import rendering

            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()

                if entity.color is not None:
                    r, g, b = entity.color
                    if isinstance(entity, Agent):
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
                pos = np.zeros(world.dim_p)
            else:
                agent_id = self.agent_ids[i]
                pos = world.agents[
                    int(agent_id.split("_")[1])
                ]  # self.agents[self.agent_ids[i]].state.p_pos
            self.viewers[i].set_bounds(
                pos[0] - cam_range,
                pos[0] + cam_range,
                pos[1] - cam_range,
                pos[1] + cam_range,
            )
            # update geometry positions
            if self.render_geoms_xform is not None:
                for e, entity in enumerate(world.entities):
                    self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            else:
                raise ValueError("self.render_geoms_xform is still None!")
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array=mode == "rgb_array"))

        return np.squeeze(results)

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self) -> List[np.ndarray]:
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
