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
from typing import Any, Callable, Dict, NamedTuple, Tuple, Union

import dm_env
import numpy as np
from acme import specs, types
from flatland.envs.observations import GlobalObsForRailEnv, Node, TreeObsForRailEnv
from flatland.envs.rail_env import RailEnv


class OLT(NamedTuple):
    """Container for (observation, legal_actions, terminal) tuples."""

    observation: types.Nest
    legal_actions: types.Nest
    terminal: types.Nest


class FlatlandEnvWrapper(dm_env.Environment):
    """Environment wrapper for Flatland environments.

    All environments would require an observation preprocessor, except for
    'GlobalObsForRailEnv'. This is because flatland gives users the
    flexibility of designing custom observation builders. 'TreeObsForRailEnv'
    would use the normalize_observation function from the flatland baselines
    if none is supplied.

    The supplied preprocessor should return either an array, tuple of arrays or
    a dictionary of arrays for an observation input.

    The obervation, for an agent, returned by this wrapper would consist of the
    agent observation and agent info. This is because flatland also provides
    informationn about the agents at each step. This information include;
    'action_required', 'malfunction', 'speed', and 'status', and it is appended
    to the observation, by this wrapper, as an array. action_required is a boolean,
    malfunction is an int denoting the number of steps for which the agent would
    remain motionless, speed is a float and status can be any of the below;

    READY_TO_DEPART = 0
    ACTIVE = 1
    DONE = 2
    DONE_REMOVED = 3
    """

    # Note: we don't inherit from base.EnvironmentWrapper because that class
    # assumes that the wrapped environment is a dm_env.Environment.
    def __init__(
        self,
        environment: RailEnv,
        preprocessor: Callable[
            [Any], Union[np.ndarray, Tuple[np.ndarray], Dict[str, np.ndarray]]
        ] = None,
    ):
        self._environment = environment
        self._reset_next_step = True
        self._step_type = dm_env.StepType.FIRST
        self.num_actions = 5

        # preprocessor must be for observation builders other than global obs
        # treeobs builders would use the default preprocessor if none is
        # supplied
        self.preprocessor: Callable[
            [Any], Union[np.ndarray, Tuple[np.ndarray], Dict[str, np.ndarray]]
        ] = self._obtain_preprocessor(preprocessor)

        # observation space:
        # flatland defines no observation space for an agent. Here we try
        # to define the observation space. All agents are identical and would
        # have the same observation space.
        # Infer observation space based on returned observation
        obs, _ = self._environment.reset()
        obs_spec = _infer_observation_spec(obs)
        agent_info_spec = _agent_info_spec()
        self.obs_spec = tuple((obs_spec, agent_info_spec))
        self.act_spec = specs.DiscreteArray(num_values=self.num_actions)

    def reset(self) -> dm_env.TimeStep:
        """Resets the episode."""
        self._reset_next_step = False
        self._discounts = {
            agent: np.dtype("float32").type(1.0)
            for agent in self._environment.number_of_agents
        }
        observe, info = self._environment.reset()
        observations = self._convert_observations(observe, info)
        return dm_env.restart(observations)

    def step(self, actions: Dict[str, np.ndarray]) -> dm_env.TimeStep:
        """Steps the environment."""
        if self._reset_next_step:
            return self.reset()

        actions_ = {int(k.split("_")[-1]): int(v) for k, v in actions.items()}
        observations, rewards, dones, infos = self._environment.step(actions_)
        rewards = {
            f"train_{agent}": np.dtype("float32").type(reward)
            for agent, reward in rewards.items()
        }

        observations = self._convert_observations(observations, infos)

        if self._step_type == dm_env.StepType.FIRST:
            step_type = self._step_type
            self._step_type = dm_env.StepType.MID
        else:
            step_type = (
                dm_env.StepType.LAST
                if self._environment.dones["__all__"]
                else dm_env.StepType.MID
            )
        self._reset_next_step = step_type == dm_env.StepType.LAST

        return dm_env.TimeStep(
            observation=observations,
            reward=rewards,
            discount=self._discounts,
            step_type=step_type,
        )

    # Convert Flatland observation so it's dm_env compatible. Also, the list
    # of legal actions must be converted to a legal actions mask.
    def _convert_observations(
        self, observes: Dict[int, np.ndarray], info: Dict[str, Dict[int, Any]]
    ) -> Dict[str, OLT]:
        observations: Dict[str, OLT] = {}
        for agent, observation in observes.items():
            agent_id = f"train_{agent}"
            legals = np.ones(self.num_actions, dtype=np.float32)
            agent_info = np.array([info[k][agent] for k in info.keys()])
            observation = self.preprocessor(observation)
            observations[agent_id] = OLT(
                observation=(observation, agent_info),
                legal_actions=legals,
                terminal=np.asarray([self._environment.dones[agent]], dtype=np.float32),
            )
        return observations

    def _obtain_preprocessor(
        self, preprocessor: Any
    ) -> Callable[[Any], Union[np.ndarray, Tuple[np.ndarray], Dict[str, np.ndarray]]]:
        """Obtains the actual preprocessor to be used based on the supplied
        preprocessor and the env's obs_builder object"""
        if not isinstance(self._environment.obs_builder, GlobalObsForRailEnv):
            _preprocessor = preprocessor
            if isinstance(self._environment.obs_builder, TreeObsForRailEnv):
                _preprocessor = (
                    normalize_observation if not preprocessor else preprocessor
                )
            assert _preprocessor is not None
        else:

            def _preprocessor(
                x: Tuple[np.ndarray, np.ndarray, np.ndarray]
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
                return x

        returned_preprocessor: Callable[
            [Any], Union[np.ndarray, Tuple[np.ndarray], Dict[str, np.ndarray]]
        ] = _preprocessor
        return returned_preprocessor

    def observation_spec(self) -> Dict[str, OLT]:
        observation_specs = {}
        for agent in range(self._environment.number_of_agents):
            agent_id = f"train_{agent}"
            observation_specs[agent_id] = OLT(
                observation=self.obs_spec,
                legal_actions=self.act_spec,
                terminal=specs.Array((1,), np.float32),
            )
        return observation_specs

    def action_spec(self) -> Dict[str, Union[specs.DiscreteArray, specs.BoundedArray]]:
        action_specs = {}
        for agent in self._environment.number_of_agents:
            action_specs[agent] = self.act_spec
        return action_specs

    def reward_spec(self) -> Dict[str, specs.Array]:
        reward_specs = {}
        for agent in self._environment.number_of_agents:
            reward_specs[agent] = specs.Array((), np.float32)
        return reward_specs

    def discount_spec(self) -> Dict[str, specs.BoundedArray]:
        discount_specs = {}
        for agent in self._environment.number_of_agents:
            discount_specs[agent] = specs.BoundedArray(
                (), np.float32, minimum=0, maximum=1.0
            )
        return discount_specs

    @property
    def environment(self) -> RailEnv:
        """Returns the wrapped environment."""
        return self._environment
    
    @property
    def num_agents(self) -> int:
        """Returns the number of trains/agents in the flatland environment"""
        return self._environment.number_of_agents

    @property
    def num_agents(self) -> int:
        """Returns the number of trains/agents in the flatland environment"""
        return int(self._environment.number_of_agents)

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        return getattr(self._environment, name)


# HELPER FUNCTIONS used in the flatland env wrapper above


def _infer_observation_spec(
    obs: Union[tuple, np.ndarray, dict]
) -> Union[specs.BoundedArray, tuple, dict]:
    """Observation spec from a sample observation"""
    if isinstance(obs, np.ndarray):
        return specs.BoundedArray(
            obs.shape,
            dtype=obs.dtype,
            minimum=-np.inf,
            maximum=np.inf,
        )
    elif isinstance(obs, tuple):
        return tuple(_infer_observation_spec(o) for o in obs)
    elif isinstance(obs, dict):
        return {key: _infer_observation_spec(value) for key, value in obs.items()}
    else:
        raise ValueError(
            f"Unexpected observation type: {type(obs)}. "
            f"Observation should be of either of this types "
            f"(np.ndarray, tuple, or dict)"
        )


def _agent_info_spec() -> specs.BoundedArray:
    """Create the spec for the agent_info part of the observation"""
    return specs.BoundedArray((4,), dtype=np.float32, minimum=0.0, maximum=10)


# The block of code below is obtained from the flatland starter-kit
# at https://gitlab.aicrowd.com/flatland/flatland-starter-kit/-/blob/master/
# utils/observation_utils.py
# this is done just to obtain the normalize_observation function that would
# serve as the default preprocessor for the Tree obs builder.


def max_lt(seq: Any, val: Any) -> Any:
    """
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


def min_gt(seq: Any, val: Any) -> Any:
    """
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
    """
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
    """
    This function splits the tree into three difference arrays of values
    """
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
    """
    This function normalizes the observation used by the RL algorithm
    """
    data, distance, agent_data = split_tree_into_feature_groups(observation, tree_depth)

    data = norm_obs_clip(data, fixed_radius=observation_radius)
    distance = norm_obs_clip(distance, normalize_to_range=True)
    agent_data = np.clip(agent_data, -1, 1)
    normalized_obs = np.concatenate((np.concatenate((data, distance)), agent_data))
    return normalized_obs
