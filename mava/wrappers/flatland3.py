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

"""Wraps a Flatland 3 environment to be used as a dm_env environment."""


import types as tp
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import dm_env
from flatland.envs.step_utils.states import TrainState
import numpy as np
import tensorflow as tf
from acme import specs
from acme.wrappers.gym_wrapper import _convert_to_spec

try:
    from flatland.envs.observations import GlobalObsForRailEnv, Node, TreeObsForRailEnv
    from flatland.envs.rail_env import RailEnv
    from flatland.utils.rendertools import AgentRenderVariant, RenderTool
except ModuleNotFoundError:
    pass
from gym.spaces import Discrete
from gym.spaces.box import Box

from mava.types import OLT, Observation
from mava.utils.sort_utils import sort_str_num
from mava.utils.wrapper_utils import (
    convert_dm_compatible_observations,
    convert_np_type,
    parameterized_restart,
)
from mava.wrappers.env_wrappers import ParallelEnvWrapper


class Flatland3EnvWrapper(ParallelEnvWrapper):
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

    def __init__(
        self,
        environment: RailEnv,
        include_agent_info: bool = True,
    ):
        """Wrap Flatland environment.

        NOTE: For now we will assume RailEnv uses the TreeObservation and we will
        normalize the observation.

        Args:
            environment: underlying RailEnv
            include_agent_info: whether or not to include agent info in extras.
        """
        self._environment = environment
        self._include_agent_info = include_agent_info

        self._agents = [get_agent_id(i) for i in range(self.num_agents)]
        self._possible_agents = self.agents[:]
        self._environment.remove_agents_at_target = False

        self._reset_next_step = True
        self._step_type = dm_env.StepType.FIRST
        

        self.num_actions = 5
        self.action_spaces = {
            agent: Discrete(self.num_actions) for agent in self.possible_agents
        }

        # observation space:
        # flatland defines no observation space for an agent. Here we try
        # to define the observation space. All agents are identical and would
        # have the same observation space.
        # Infer observation space based on returned observation
        obs, _ = self._environment.reset()
        obs = self._normalize_observations(obs)

        if self._include_agent_info:
            for k, o in obs.items():
                obs[k] = np.concatenate((np.ones((7,), dtype=np.float32), o), axis=-1)

        self.observation_spaces = {
            get_agent_id(i): infer_observation_space(o) for i, o in obs.items()
        }

        # Renderer
        self._env_renderer = RenderTool(self._environment)

    def _normalize_observations(self, obs_dict):
        """Normalise a dict of observations."""
        for agent in self._agents:
            agent_handle = get_agent_handle(agent)
            obs_dict[agent_handle] = normalize_observation(
                obs_dict[agent_handle], 
                tree_depth=self.obs_builder.max_depth,
            )

        return obs_dict

    @property
    def agents(self) -> List[str]:
        """Return list of active agents."""
        return self._agents

    @property
    def possible_agents(self) -> List[str]:
        """Return list of all possible agents."""
        return self._possible_agents

    def render(self, mode: str = "human") -> np.array:
        """Renders the environment."""
        if mode == "human":
            show = True
        else:
            show = False

        return self._env_renderer.render_env(
            show=show,
            show_observations=False,
            show_predictions=False,
            return_image=True,
        )

    def env_done(self) -> bool:
        """Checks if the environment is done."""
        return self._environment.dones["__all__"]

    def reset(self) -> dm_env.TimeStep:
        """Resets the episode."""
        # Reset the rendering sytem
        self._env_renderer.reset()

        self._reset_next_step = False
        obs, info = self._environment.reset() # NOTE (Claude) we assume TreeObs for now

        observations = self._create_observations(obs, self._environment.dones, info)

        rewards_spec = self.reward_spec()
        rewards = {
            agent: convert_np_type(rewards_spec[agent].dtype, 0) # Zero reward
            for agent in self.possible_agents
        }

        extras = {
            "state": {get_agent_id(handle): state for handle, state in info["state"].items()}
        }

        discount_spec = self.discount_spec()
        discounts = {
            agent: convert_np_type(discount_spec[agent].dtype, 1) # Discount = 1.0
            for agent in self.possible_agents
        }

        return parameterized_restart(rewards, discounts, observations), extras

    def step(self, actions: Dict[str, np.ndarray]) -> dm_env.TimeStep:
        """Steps the environment."""

        if self._reset_next_step:
            return self.reset()

        # Change action keys to ints for Flatland
        actions = {get_agent_handle(k): int(v) for k, v in actions.items()}

        # Step environment
        observations, rewards, dones, info = self._environment.step(actions)

        # Process observations
        observations = self._create_observations(observations, dones, info)

        extras = {
            "state": {get_agent_id(handle): state for handle, state in info["state"].items()}
        }

        # Process rewards
        rewards_spec = self.reward_spec()
        rewards = {
            get_agent_id(agent): convert_np_type(
                rewards_spec[get_agent_id(agent)].dtype, reward
            )
            for agent, reward in rewards.items()
        }

        if self.env_done():
            self._step_type = dm_env.StepType.LAST
            self._reset_next_step = True

            # Get discounts
            discounts_spec = self.discount_spec()
            discounts = {
                agent: convert_np_type(
                    discounts_spec[agent].dtype, 0 # Discount = 0.0
                )
                for agent in self.agents
            }
        else:
            self._step_type = dm_env.StepType.MID

            # Get discounts
            discounts_spec = self.discount_spec()
            discounts = {
                agent: convert_np_type(
                    discounts_spec[agent].dtype, 1 # Discount = 1.0
                )
                for agent in self.agents
            }

        return dm_env.TimeStep(
            observation=observations,
            reward=rewards,
            discount=discounts,
            step_type=self._step_type,
        ), extras

    def _create_observations(
        self,
        obs: Dict[int, np.ndarray],
        dones: Dict[int, bool],
        info: Dict
    ) -> Observation:
        """Convert observation to dm_env compatible."""
        # Normalize observations
        norm_obs = self._normalize_observations(obs)
        # Terminals
        dones = {get_agent_id(k): dones["__all__"] for k, v in dones.items()}

        observations = {}
        for agent in self._environment.agents:

            # Get legal actions mask
            if self._environment.action_required(agent):
                legal_actions = np.ones((5,), dtype=int)
            else:
                legal_actions = np.zeros((5,), dtype=int)
                legal_actions[0] = 1

            # Maybe include agent info
            if self._include_agent_info:
                one_hot_state = np.zeros((7,), dtype=np.float32)
                state = info["state"][agent.handle]
                one_hot_state[state] = 1
                agent_obs = np.concatenate((one_hot_state, norm_obs[agent.handle]), axis=-1)
            else:
                agent_obs = norm_obs[agent.handle]

            agent_id = get_agent_id(agent.handle)
            observations[agent_id] = {
                "observation": agent_obs,
                "action_mask": legal_actions
            }

        observations = convert_dm_compatible_observations(
            observations,
            dones,
            self.action_spaces,
            self.observation_spaces,
            self.env_done(),
            self.possible_agents,
        )
        
        return observations

    def observation_spec(self) -> Dict[str, OLT]:
        """Return observation spec."""
        observation_specs = {}
        for agent in self.agents:
            observation_specs[agent] = OLT(
                observation=_convert_to_spec(self.observation_spaces[agent]),
                legal_actions=np.ones((5,), dtype=int),
                terminal=specs.Array((1,), np.float32),
            )
        return observation_specs

    def action_spec(self) -> Dict[str, Union[specs.DiscreteArray, specs.BoundedArray]]:
        """Get action spec."""
        action_specs = {}
        action_spaces = self.action_spaces
        for agent in self.possible_agents:
            action_specs[agent] = _convert_to_spec(action_spaces[agent])
        return action_specs

    def reward_spec(self) -> Dict[str, specs.Array]:
        """Get the reward spec."""
        reward_specs = {}
        for agent in self.possible_agents:
            reward_specs[agent] = specs.Array((), np.float32)
        return reward_specs

    def discount_spec(self) -> Dict[str, specs.BoundedArray]:
        """Get the discount spec."""
        discount_specs = {}
        for agent in self.possible_agents:
            discount_specs[agent] = specs.BoundedArray(
                (), np.float32, minimum=0, maximum=1.0
            )
        return discount_specs

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        """Get the extras spec."""
        return {}

    def seed(self, seed: int = None) -> None:
        """Seed the environment."""
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


# Utility functions


def infer_observation_space(
    obs: Union[tuple, np.ndarray, dict]
) -> Union[Box, tuple, dict]:
    """Infer a gym Observation space from a sample observation from flatland"""
    if isinstance(obs, np.ndarray):
        return Box(
            -np.inf,
            np.inf,
            shape=obs.shape,
            dtype=obs.dtype,
        )
    elif isinstance(obs, tuple):
        return tuple(infer_observation_space(o) for o in obs)
    elif isinstance(obs, dict):
        return {key: infer_observation_space(value) for key, value in obs.items()}
    else:
        raise ValueError(
            f"Unexpected observation type: {type(obs)}. "
            f"Observation should be of either of this types "
            f"(np.ndarray, tuple, or dict)"
        )


def get_agent_id(handle: int) -> str:
    """Obtain the string that constitutes the agent id from an agent handle - an int"""
    return f"train_{handle}"


def get_agent_handle(id: str) -> int:
    """Obtain an agents handle given its id"""
    return int(id.split("_")[-1])


# The block of code below is obtained from the flatland starter-kit
# at https://gitlab.aicrowd.com/flatland/flatland-starter-kit/-/blob/master/
# utils/observation_utils.py
# this is done just to obtain the normalize_observation function that would
# serve as the default preprocessor for the Tree obs builder.


def max_lt(seq: Sequence, val: Any) -> Any:
    """Get max in sequence.

    Return greatest item in seq for which item < val applies.
    None is returned if seq was empty or all items in seq were >= val.
    """
    max = 0
    idx = len(seq) - 1
    while idx >= 0:
        if seq[idx] < val and seq[idx] >= 0 and seq[idx] > max:
            max = seq[idx]
        idx -= 1
    return max


def min_gt(seq: Sequence, val: Any) -> Any:
    """Gets min in a sequence.

    Return smallest item in seq for which item > val applies.
    None is returned if seq was empty or all items in seq were >= val.
    """
    min = np.inf
    idx = len(seq) - 1
    while idx >= 0:
        if seq[idx] >= val and seq[idx] < min:
            min = seq[idx]
        idx -= 1
    return min


def norm_obs_clip(
    obs: np.ndarray,
    clip_min: int = -1,
    clip_max: int = 1,
    fixed_radius: int = 0,
    normalize_to_range: bool = False,
) -> np.ndarray:
    """Normalize observation.

    This function returns the difference between min and max value of an observation
    :param obs: Observation that should be normalized
    :param clip_min: min value where observation will be clipped
    :param clip_max: max value where observation will be clipped
    :return: returnes normalized and clipped observatoin
    """
    if fixed_radius > 0:
        max_obs = fixed_radius
    else:
        max_obs = max(1, max_lt(obs, 1000)) + 1

    min_obs = 0  # min(max_obs, min_gt(obs, 0))
    if normalize_to_range:
        min_obs = min_gt(obs, 0)
    if min_obs > max_obs:
        min_obs = max_obs
    if max_obs == min_obs:
        return np.clip(np.array(obs) / max_obs, clip_min, clip_max)
    norm = np.abs(max_obs - min_obs)
    return np.clip((np.array(obs) - min_obs) / norm, clip_min, clip_max)


def _split_node_into_feature_groups(
    node: Node,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits node into features."""
    data = np.zeros(6)
    distance = np.zeros(1)
    agent_data = np.zeros(4)

    data[0] = node.dist_own_target_encountered
    data[1] = node.dist_other_target_encountered
    data[2] = node.dist_other_agent_encountered
    data[3] = node.dist_potential_conflict
    data[4] = node.dist_unusable_switch
    data[5] = node.dist_to_next_branch

    distance[0] = node.dist_min_to_target

    agent_data[0] = node.num_agents_same_direction
    agent_data[1] = node.num_agents_opposite_direction
    agent_data[2] = node.num_agents_malfunctioning
    agent_data[3] = node.speed_min_fractional

    return data, distance, agent_data


def _split_subtree_into_feature_groups(
    node: Node, current_tree_depth: int, max_tree_depth: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split subtree."""
    if node == -np.inf:
        remaining_depth = max_tree_depth - current_tree_depth
        # reference:
        # https://stackoverflow.com/questions/515214/total-number-of-nodes-in-a-tree-data-structure
        num_remaining_nodes = int((4 ** (remaining_depth + 1) - 1) / (4 - 1))
        return (
            [-np.inf] * num_remaining_nodes * 6,
            [-np.inf] * num_remaining_nodes,
            [-np.inf] * num_remaining_nodes * 4,
        )

    data, distance, agent_data = _split_node_into_feature_groups(node)

    if not node.childs:
        return data, distance, agent_data

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        sub_data, sub_distance, sub_agent_data = _split_subtree_into_feature_groups(
            node.childs[direction], current_tree_depth + 1, max_tree_depth
        )
        data = np.concatenate((data, sub_data))
        distance = np.concatenate((distance, sub_distance))
        agent_data = np.concatenate((agent_data, sub_agent_data))

    return data, distance, agent_data


def split_tree_into_feature_groups(
    tree: Node, max_tree_depth: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function splits the tree into three difference arrays."""
    data, distance, agent_data = _split_node_into_feature_groups(tree)

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        sub_data, sub_distance, sub_agent_data = _split_subtree_into_feature_groups(
            tree.childs[direction], 1, max_tree_depth
        )
        data = np.concatenate((data, sub_data))
        distance = np.concatenate((distance, sub_distance))
        agent_data = np.concatenate((agent_data, sub_agent_data))

    return data, distance, agent_data


def normalize_observation(
    observation: Node, tree_depth: int, observation_radius: int = 0
) -> np.ndarray:
    """This function normalizes the observation used by the RL algorithm."""
    if observation is None:
        return np.zeros(
            11 * sum(np.power(4, i) for i in range(tree_depth + 1)), dtype=np.float32
        )
    data, distance, agent_data = split_tree_into_feature_groups(observation, tree_depth)

    data = norm_obs_clip(data, fixed_radius=observation_radius)
    distance = norm_obs_clip(distance, normalize_to_range=True)
    agent_data = np.clip(agent_data, -1, 1)
    normalized_obs = np.array(
        np.concatenate((np.concatenate((data, distance)), agent_data)), dtype=np.float32
    )
    return normalized_obs
