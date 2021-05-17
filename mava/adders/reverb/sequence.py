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
# https://github.com/deepmind/acme/blob/master/acme/adders/reverb/sequence.py

"""Sequence adders.
This implements adders which add sequences or partial trajectories.
"""

from typing import NamedTuple, Optional

import reverb
import tensorflow as tf
import tree
from acme import specs, types
from acme.adders.reverb import utils
from acme.utils import tree_utils

from mava.adders.reverb import base
from mava.adders.reverb import utils as mava_utils


class StateSpecs(NamedTuple):
    """Container for (observation, legal_actions, terminal) tuples."""

    hidden: types.Nest
    cell: types.Nest


class ParallelSequenceAdder(base.ReverbParallelAdder):
    """An adder which adds sequences of fixed length."""

    def __init__(
        self,
        client: reverb.Client,
        sequence_length: int,
        period: int,
        delta_encoded: bool = False,
        chunk_length: Optional[int] = None,
        priority_fns: Optional[base.PriorityFnMapping] = None,
        pad_end_of_episode: bool = True,
        break_end_of_episode: bool = True,
        max_in_flight_items: Optional[int] = 25,
        use_next_extras: bool = True,
    ):
        """Makes a SequenceAdder instance.
        Args:
          client: See docstring for BaseAdder.
          sequence_length: The fixed length of sequences we wish to add.
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

        if delta_encoded:
            NotImplementedError(
                "Note (dries): Delta encoding has not been verified to "
                "work in Mava yet. If you want to use delta encoding"
                " first verify that it is working and then contribute "
                "the update with a working example of delta encoding"
                " to the Mava repo."
            )

        super().__init__(
            client=client,
            buffer_size=sequence_length,
            max_sequence_length=sequence_length,
            delta_encoded=delta_encoded,
            chunk_length=chunk_length,
            priority_fns=priority_fns,
            max_in_flight_items=max_in_flight_items,
            use_next_extras=use_next_extras,
        )

        if pad_end_of_episode and not break_end_of_episode:
            raise ValueError(
                "Can't set pad_end_of_episode=True and break_end_of_episode=False at"
                " the same time, since those behaviors are incompatible."
            )
        self._period = period
        self._step = 0
        self._pad_end_of_episode = pad_end_of_episode
        self._break_end_of_episode = break_end_of_episode

    def reset(self) -> None:
        # If we do not break on end of episode, we should not reset the _step
        # counter, neither clear the buffer/writer.
        if self._break_end_of_episode:
            self._step = 0
            super().reset()

    def _write(self) -> None:
        # Append the previous step and increment number of steps written.
        self._writer.append(self._buffer[-1])
        self._step += 1
        self._maybe_add_priorities()

    def _write_last(self) -> None:
        # Create a final step.
        # TODO (Dries): Should self._next_observation be used
        #  here? Should this function be used for sequential?

        final_step = mava_utils.final_step_like(
            self._buffer[0],
            self._next_observations,
            self._next_extras if self._use_next_extras else None,
        )

        # Append the final step.
        self._buffer.append(final_step)
        self._writer.append(final_step)
        self._step += 1

        if not self._break_end_of_episode:
            # Write priorities for the sequence.
            self._maybe_add_priorities()

            # base.py has a check that on add_first self._next_observation should be
            # None, thus we need to clear it at the end of each episode.
            self._next_observations = None
            self._next_extras = None
            return None

        # Determine the delta to the next time we would write a sequence.
        first_write = self._step <= self._max_sequence_length
        if first_write:
            delta = self._max_sequence_length - self._step
        else:
            delta = (
                self._period - (self._step - self._max_sequence_length)
            ) % self._period

        # Bump up to the position where we will write a sequence.
        self._step += delta

        if self._pad_end_of_episode:
            zero_step = tree.map_structure(utils.zeros_like, final_step)

            # Pad with zeros to get a full sequence.
            for _ in range(delta):
                self._buffer.append(zero_step)
                self._writer.append(zero_step)
        elif not first_write:
            # Pop items from the buffer to get a truncated sequence.
            # Note: this is consistent with the padding loop above, since adding zero
            # steps pops the left-most elements. Here we just pop without padding.
            for _ in range(delta):
                self._buffer.popleft()

        # Write priorities for the sequence.
        self._maybe_add_priorities()

    def _maybe_add_priorities(self) -> None:
        if not (
            # Write the first time we hit the max sequence length...
            self._step == self._max_sequence_length
            or
            # ... or every `period`th time after hitting max length.
            (
                self._step > self._max_sequence_length
                and (self._step - self._max_sequence_length) % self._period == 0
            )
        ):
            return

        # Compute priorities for the buffer.
        steps = list(self._buffer)
        num_steps = len(steps)
        table_priorities = utils.calculate_priorities(self._priority_fns, steps)

        # Create a prioritized item for each table.
        for table_name, priority in table_priorities.items():
            self._writer.create_item(table_name, num_steps, priority)

    @classmethod
    def signature(
        cls,
        environment_spec: specs.EnvironmentSpec,
        extras_spec: tf.TypeSpec = {},
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

        spec_step = base.Step(
            observations=obs_specs,
            actions=act_specs,
            rewards=reward_specs,
            discounts=step_discount_specs,
            start_of_episode=specs.Array(shape=(), dtype=bool),
            extras=extras_spec,
        )
        return tree.map_structure_with_path(base.spec_like_to_tensor_spec, spec_step)
