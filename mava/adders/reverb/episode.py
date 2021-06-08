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

# Adapted from
# https://github.com/deepmind/acme/blob/master/acme/adders/reverb/episode.py

"""Episode adders.

This implements full episode adders, potentially with padding.
"""

from typing import Optional

import dm_env
import reverb
import tensorflow as tf
import tree
from acme import specs, types
from acme.adders.reverb import utils
from acme.utils import tree_utils

from mava.adders.reverb import base
from mava.adders.reverb import utils as mava_utils


class ParallelEpisodeAdder(base.ReverbParallelAdder):
    """An adder which adds sequences of fixed length."""

    def __init__(
        self,
        client: reverb.Client,
        max_sequence_length: int,
        delta_encoded: bool = False,
        chunk_length: Optional[int] = None,
        priority_fns: Optional[base.PriorityFnMapping] = None,
        pad_end_of_episode: bool = True,
        break_end_of_episode: bool = True,
        max_in_flight_items: Optional[int] = 25,
    ):
        """Makes a SequenceAdder instance.

        Args:
          client: See docstring for BaseAdder.
          max_sequence_length: The maximum length of an episode.
          period: The period with which we add sequences. If less than
            sequence_length, overlapping sequences are added. If equal to
            sequence_length, sequences are exactly non-overlapping.
          delta_encoded: If `True` (False by default) enables delta encoding, see
            `Client` for more information.
          chunk_length: Number of timesteps grouped together before delta encoding
            and compression. See `Client` for more information.
          priority_fns: See docstring for BaseAdder.
          pad_end_of_episode: If True (default) then upon end of episode the current
            sequence will be padded (with observations, actions, etc... whose values
            are 0) until its length is `sequence_length`. If False then the last
            sequence in the episode may have length less than `sequence_length`.
          break_end_of_episode: If 'False' (True by default) does not break
            sequences on env reset. In this case 'pad_end_of_episode' is not used.
          max_in_flight_items: The maximum number of items allowed to be "in flight"
            at the same time. See `reverb.Writer.writer` for more info.
        """

        super().__init__(
            client=client,
            buffer_size=max_sequence_length - 1,
            max_sequence_length=max_sequence_length,
            delta_encoded=delta_encoded,
            chunk_length=chunk_length,
            priority_fns=priority_fns,
            max_in_flight_items=max_in_flight_items,
        )

    def add(
        self,
        action: types.NestedArray,
        next_timestep: dm_env.TimeStep,
        extras: types.NestedArray = (),
    ) -> None:
        if len(self._buffer) == self._buffer.maxlen:
            # If the buffer is full that means we've buffered max_sequence_length-1
            # steps, one dangling observation, and are trying to add one more (which
            # will overflow the buffer).
            raise ValueError(
                "The number of observations within the same episode exceeds "
                "max_sequence_length"
            )

        super().add(action, next_timestep, extras)

    def _write(self) -> None:
        # Append the previous step and increment number of steps written.
        self._writer.append(self._buffer[-1])

    def _write_last(self) -> None:
        # Create a final step.
        # TODO (Dries): Should self._next_observation be used
        #  here? Should this function be used for sequential?
        final_step = mava_utils.final_step_like(
            self._buffer[0], self._next_observations, self._next_extras
        )

        # Append the final step.
        self._writer.append(final_step)

        # The length of the sequence we will be adding is the size of the buffer
        # plus one due to the final step.
        steps = list(self._buffer) + [final_step]
        num_steps = len(steps)

        # Calculate the priority for this episode.
        table_priorities = utils.calculate_priorities(self._priority_fns, steps)

        # Create a prioritized item for each table.
        for table_name, priority in table_priorities.items():
            self._writer.create_item(table_name, num_steps, priority)

    @classmethod
    def signature(
        cls,
        environment_spec: specs.EnvironmentSpec,
        extras_specs: tf.TypeSpec,
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

        Returns:
          A `Step` whose leaf nodes are `tf.TensorSpec` objects.
        """
        agent_specs = environment_spec.get_agent_specs()
        agents = environment_spec.get_agent_ids()
        extras_specs.update(environment_spec.get_extra_specs())
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

        spec_step = base.Step(
            observations=obs_specs,
            actions=act_specs,
            rewards=reward_specs,
            discounts=step_discount_specs,
            start_of_episode=specs.Array(shape=(), dtype=bool),
            extras=extras_specs,
        )
        return tree.map_structure_with_path(base.spec_like_to_tensor_spec, spec_step)
