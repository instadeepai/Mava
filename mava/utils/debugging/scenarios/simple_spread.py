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

from typing import List

import numpy as np

from mava.utils.debugging.core import Agent, Landmark, World
from mava.utils.debugging.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, num_agents: int) -> World:
        world = World()
        # set any world properties first
        num_landmarks = num_agents
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent %d" % i
            agent.collide = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world: World) -> None:
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        # Reset step counter
        world.current_step = 0

    def is_collision(self, agent1: Agent, agent2: Agent) -> bool:
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    @staticmethod
    def dist(pt1: np.array, pt2: np.array) -> float:
        return np.sqrt(np.sum(np.square(pt1 - pt2)))

    def reward(self, agent: Agent, a_i: int, world: World) -> float:
        # Agents are rewarded based on agent distance to its corresponding
        # landmark, penalized for collisions
        rew = 0.0
        # for i in range(len(world.agents)):
        other_agent = world.agents[a_i]
        landmark = world.landmarks[a_i]
        distance = self.dist(other_agent.state.p_pos, landmark.state.p_pos)

        # Cap distance
        if distance > 1.0:
            distance = 1.0
        rew += 1 - distance

        if agent.collide:
            for other in world.agents:
                if other is agent:
                    continue
                if self.is_collision(other, agent):
                    rew -= 1

        return rew

    def observation(self, agent: Agent, a_i: int, world: World) -> np.array:
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color: List[np.array] = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        other_pos = []
        other_landmarks = []
        for i, other in enumerate(world.agents):
            if other is agent:
                continue
            landmark = world.landmarks[i]
            other_landmarks.append(landmark.state.p_pos - landmark.state.p_pos)
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        return np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + [[world.current_step / 50]]
            + [entity_pos[a_i]]
            + other_pos
            + other_landmarks  # + comm
        )

    def done(self, agent: Agent, world: World) -> bool:
        if world.current_step < 50:
            return False
        else:
            return True
