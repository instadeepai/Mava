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
from acme.adders.reverb import utils as acme_utils
from acme.adders.reverb.episode import EpisodeAdder, _PaddingFn

from mava.adders.reverb import base
from mava.adders.reverb import utils as mava_utils
from mava.adders.reverb.base import ReverbParallelAdder


class ParallelEpisodeAdder(EpisodeAdder, ReverbParallelAdder):
    """An adder which adds sequences of fixed length."""

    def __init__(
        self,
        client: reverb.Client,
        max_sequence_length: int,
        delta_encoded: bool = False,
        priority_fns: Optional[base.PriorityFnMapping] = None,
        max_in_flight_items: int = 1,
        padding_fn: Optional[_PaddingFn] = None,
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

        ReverbParallelAdder.__init__(
            self,
            client=client,
            max_sequence_length=max_sequence_length,
            delta_encoded=delta_encoded,
            priority_fns=priority_fns,
            max_in_flight_items=max_in_flight_items,
        )
        self._padding_fn = padding_fn

    def add(
        self,
        action: types.NestedArray,
        next_timestep: dm_env.TimeStep,
        next_extras: types.NestedArray = (),
    ) -> None:
        if self._writer.episode_steps >= self._max_sequence_length - 1:
            raise ValueError(
                "The number of observations within the same episode will exceed "
                "max_sequence_length with the addition of this transition."
            )

        super().add(action, next_timestep, next_extras)

    def _write_last(self) -> None:
        if (
            self._padding_fn is not None
            and self._writer.episode_steps < self._max_sequence_length
        ):
            history = self._writer.history
            padding_step = dict(
                observation=history["observation"],
                action=history["action"],
                reward=history["reward"],
                discount=history["discount"],
                extras=history.get("extras", ()),
            )
            # Get shapes and dtypes from the last element.
            padding_step = tree.map_structure(
                lambda col: self._padding_fn(col[-1].shape, col[-1].dtype), padding_step
            )
            padding_step["start_of_episode"] = False
            while self._writer.episode_steps < self._max_sequence_length:
                self._writer.append(padding_step)

        trajectory = tree.map_structure(lambda x: x[:], self._writer.history)

        # Pack the history into a base.Step structure and get numpy converted
        # variant for priotiy computation.
        trajectory = base.Trajectory(**trajectory)

        # Calculate the priority for this episode.
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
        extras_spec: types.NestedSpec = (),
    ) -> tf.TypeSpec:
        return mava_utils.trajectory_signature(
            environment_spec=environment_spec,
            sequence_length=sequence_length,
            extras_spec=extras_spec,
        )
