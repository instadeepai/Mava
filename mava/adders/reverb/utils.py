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

from typing import Iterable, Optional

import tensorflow as tf
import tree
from acme import specs, types
from acme.adders.reverb import utils as acme_utils
from acme.utils import tree_utils

from mava.adders.reverb import base


def final_step_like(
    step: base.Step, next_observations: types.NestedArray, next_extras: dict = None
) -> base.Step:
    """Return a list of steps with the final step zero-filled."""
    # Make zero-filled components so we can fill out the last step.
    zero_action, zero_reward, zero_discount = tree.map_structure(
        acme_utils.zeros_like, (step.actions, step.rewards, step.discounts)
    )

    return base.Step(
        observations=next_observations,
        actions=zero_action,
        rewards=zero_reward,
        discounts=zero_discount,
        start_of_episode=False,
        extras=next_extras
        if next_extras
        else tree.map_structure(acme_utils.zeros_like, step.extras),
    )


def trajectory_signature(
    environment_spec: specs.EnvironmentSpec,
    sequence_length: Optional[int] = None,
    extras_spec: types.NestedSpec = (),
) -> tf.TypeSpec:
    """This is a helper method for generating signatures for Reverb tables.

    Signatures are useful for validating data types and shapes, see Reverb's
    documentation for details on how they are used.

    Args:
        environment_spec: A `specs.EnvironmentSpec` whose fields are nested
        structures with leaf nodes that have `.shape` and `.dtype` attributes.
        This should come from the environment that will be used to generate
        the data inserted into the Reverb table.
        extras_spec: A nested structure with leaf nodes that have `.shape` and
        `.dtype` attributes. The structure (and shapes/dtypes) of this must
        be the same as the `extras` passed into `ReverbAdder.add`.
        sequence_length: An optional integer representing the expected length of
        sequences that will be added to replay.

    Returns:
        A `Trajectory` whose leaf nodes are `tf.TensorSpec` objects.
    """

    def add_time_dim(paths: Iterable[str], spec: tf.TensorSpec) -> None:
        return tf.TensorSpec(
            shape=(sequence_length, *spec.shape),
            dtype=spec.dtype,
            name="/".join(str(p) for p in paths),
        )

    agent_specs = environment_spec.get_agent_specs()
    agents = environment_spec.get_agent_ids()
    env_extras_spec = environment_spec.get_extra_specs()
    extras_spec.update(env_extras_spec)

    obs_specs = {}
    act_specs = {}
    reward_specs = {}
    step_discount_specs = {}
    for agent in agents:
        rewards_spec, step_discounts_spec = tree_utils.broadcast_structures(
            agent_specs[agent].rewards, agent_specs[agent].discounts
        )
        obs_specs[agent] = agent_specs[agent].observations
        act_specs[agent] = agent_specs[agent].actions
        reward_specs[agent] = rewards_spec
        step_discount_specs[agent] = step_discounts_spec

    # Add a time dimension to the specs
    (
        obs_specs,
        act_specs,
        reward_specs,
        step_discount_specs,
        soe_spec,
        extras_spec,
    ) = tree.map_structure_with_path(
        add_time_dim,
        (
            obs_specs,
            act_specs,
            reward_specs,
            step_discount_specs,
            specs.Array(shape=(), dtype=bool),
            extras_spec,
        ),
    )

    spec_step = base.Trajectory(
        observations=obs_specs,
        actions=act_specs,
        rewards=reward_specs,
        discounts=step_discount_specs,
        start_of_episode=soe_spec,
        extras=extras_spec,
    )

    return tree.map_structure_with_path(base.spec_like_to_tensor_spec, spec_step)
