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

import operator
from typing import Optional

import reverb
import tensorflow as tf
import tree
from acme import specs
from acme.adders.reverb import utils as acme_utils
from acme.adders.reverb.sequence import SequenceAdder
from acme.types import NestedSpec

from mava.adders.reverb import base
from mava.adders.reverb.base import ReverbParallelAdder
from mava.adders.reverb.utils import trajectory_signature

# TODO Clean this up, when using newer versions of acme.
try:
    from acme.adders.reverb.sequence import EndBehavior
except ImportError:
    from acme.adders.reverb.sequence import EndOfEpisodeBehavior as EndBehavior


class ParallelSequenceAdder(SequenceAdder, ReverbParallelAdder):
    """An adder which adds sequences of fixed length."""

    def __init__(
        self,
        client: reverb.Client,
        sequence_length: int,
        period: int,
        *,
        delta_encoded: bool = False,
        priority_fns: Optional[base.PriorityFnMapping] = None,
        max_in_flight_items: int = 2,
        end_of_episode_behavior: Optional[EndBehavior] = EndBehavior.ZERO_PAD,
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
          priority_fns: See docstring for BaseAdder.
          max_in_flight_items: The maximum number of items allowed to be "in flight"
            at the same time. See `block_until_num_items` in
            `reverb.TrajectoryWriter.flush` for more info.
          end_of_episode_behavior:  Determines how sequences at the end of the
            episode are handled (default `EndOfEpisodeBehavior.ZERO_PAD`). See
            the docstring for `EndOfEpisodeBehavior` for more information.
          chunk_length: Deprecated and unused.
          pad_end_of_episode: If True (default) then upon end of episode the current
            sequence will be padded (with observations, actions, etc... whose values
            are 0) until its length is `sequence_length`. If False then the last
            sequence in the episode may have length less than `sequence_length`.
          break_end_of_episode: If 'False' (True by default) does not break
            sequences on env reset. In this case 'pad_end_of_episode' is not used.
          use_next_extras: If true extras will be processed the same way observations
          are processed. If false extras will be processed as actions are processed.
        """
        ReverbParallelAdder.__init__(
            self,
            client=client,
            # We need an additional space in the buffer for the partial step the
            # base.ReverbAdder will add with the next observation.
            max_sequence_length=sequence_length + 1,
            delta_encoded=delta_encoded,
            priority_fns=priority_fns,
            max_in_flight_items=max_in_flight_items,
            use_next_extras=use_next_extras,
        )

        self._period = period
        self._sequence_length = sequence_length
        self._end_of_episode_behavior = end_of_episode_behavior

    def _maybe_create_item(
        self, sequence_length: int, *, end_of_episode: bool = False, force: bool = False
    ) -> None:

        # Check conditions under which a new item is created.
        first_write = self._writer.episode_steps == sequence_length
        # NOTE(acme): the following line assumes that the only way sequence_length
        # is less than self._sequence_length, is if the episode is shorter than
        # self._sequence_length.
        period_reached = self._writer.episode_steps > self._sequence_length and (
            (self._writer.episode_steps - self._sequence_length) % self._period == 0
        )

        if not first_write and not period_reached and not force:
            return

        if not end_of_episode:
            get_traj = operator.itemgetter(slice(-sequence_length - 1, -1))
        else:
            get_traj = operator.itemgetter(slice(-sequence_length, None))

        history = self._writer.history
        trajectory = base.Trajectory(**tree.map_structure(get_traj, history))

        # Compute priorities for the buffer.
        table_priorities = acme_utils.calculate_priorities(
            self._priority_fns, trajectory
        )

        # Create a prioritized item for each table.
        for table_name, priority in table_priorities.items():
            self._writer.create_item(table_name, priority, trajectory)
            self._writer.flush(self._max_in_flight_items)

    @classmethod
    def signature(
        cls,
        environment_spec: specs.EnvironmentSpec,
        sequence_length: Optional[int] = None,
        extras_spec: NestedSpec = (),
    ) -> tf.TypeSpec:
        """Returns adder signature.

        Args:
            environment_spec (specs.EnvironmentSpec): Spec of MA environment.
            sequence_length (Optional[int], optional): Length of sequence.
                Defaults to None.
            extras_spec (NestedSpec, optional): Spec for extra data. Defaults to ().

        Returns:
            tf.TypeSpec: Signature for sequence adder.
        """
        return trajectory_signature(
            environment_spec=environment_spec,
            sequence_length=sequence_length,
            extras_spec=extras_spec,
        )
