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


# HELPER FUNCTIONS used in the flatland env wrapper

import types as tp
from typing import Any, Dict, Sequence, Tuple, Union

import dm_env
import numpy as np
from acme import specs
from flatland.envs.observations import Node, TreeObsForRailEnv
from flatland.envs.rail_env import RailEnv
from gym.spaces import Box


def _infer_observation_space(
    obs: Union[tuple, np.ndarray, dict]
) -> Union[Box, tuple, dict]:
    """Observation spec from a sample observation"""
    if isinstance(obs, np.ndarray):
        return Box(
            -np.inf,
            np.inf,
            shape=obs.shape,
            dtype=obs.dtype,
        )
    elif isinstance(obs, tuple):
        return tuple(_infer_observation_space(o) for o in obs)
    elif isinstance(obs, dict):
        return {key: _infer_observation_space(value) for key, value in obs.items()}
    else:
        raise ValueError(
            f"Unexpected observation type: {type(obs)}. "
            f"Observation should be of either of this types "
            f"(np.ndarray, tuple, or dict)"
        )


def _agent_info_spec() -> specs.BoundedArray:
    """Create the spec for the agent_info part of the observation"""
    return specs.BoundedArray((4,), dtype=np.float32, minimum=0.0, maximum=10)


def _get_agent_id(handle: int) -> str:
    return f"train_{handle}"


def _get_agent_handle(id: str) -> int:
    return int(id.split("_")[-1])


def _decorate_step_method(env: RailEnv) -> None:
    env.step_ = env.step  # type: ignore

    def _step(
        self: RailEnv, actions: Dict[str, Union[int, float, Any]]
    ) -> dm_env.TimeStep:
        actions_ = {_get_agent_handle(k): int(v) for k, v in actions.items()}
        return self.step_(actions_)

    env.step = tp.MethodType(_step, env)


# The block of code below is obtained from the flatland starter-kit
# at https://gitlab.aicrowd.com/flatland/flatland-starter-kit/-/blob/master/
# utils/observation_utils.py
# this is done just to obtain the normalize_observation function that would
# serve as the default preprocessor for the Tree obs builder.


def max_lt(seq: Sequence, val: Any) -> Any:
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


def min_gt(seq: Sequence, val: Any) -> Any:
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
