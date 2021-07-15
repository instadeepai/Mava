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
from typing import Iterable, Optional

import numpy as np

# import enum
import reverb
import tensorflow as tf
import tree
from acme import specs
from acme.adders.reverb import utils
from acme.types import NestedSpec

# TODO Clean this up, when using newer versions of acme.
try:
    from acme.adders.reverb.sequence import EndBehavior
except ImportError:
    from acme.adders.reverb.sequence import EndOfEpisodeBehavior as EndBehavior
from acme.utils import tree_utils

from mava.adders.reverb import base

# from mava.adders.reverb import utils as mava_utils


class ParallelSequenceAdder(base.ReverbParallelAdder):
    """An adder which adds sequences of fixed length."""

    def __init__(
        self,
        client: reverb.Client,
        sequence_length: int,
        period: int,
        *,
        delta_encoded: bool = False,
        priority_fns: Optional[base.PriorityFnMapping] = None,
        max_in_flight_items: Optional[int] = 2,
        end_of_episode_behavior: Optional[  # type: ignore
            EndBehavior
        ] = EndBehavior.ZERO_PAD,
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
        super().__init__(
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

    def reset(self, timeout_ms: Optional[int] = None) -> None:
        """Resets the adder's buffer."""
        # If we do not write on end of episode, we should not reset the writer.
        if self._end_of_episode_behavior is EndBehavior.CONTINUE:
            return

        super().reset()

    def _write(self) -> None:
        self._maybe_create_item(self._sequence_length)

    def _write_last(self) -> None:
        # Maybe determine the delta to the next time we would write a sequence.
        if self._end_of_episode_behavior in (
            EndBehavior.TRUNCATE,
            EndBehavior.ZERO_PAD,
        ):
            delta = self._sequence_length - self._writer.episode_steps
            if delta < 0:
                delta = (self._period + delta) % self._period

        # Handle various end-of-episode cases.
        if self._end_of_episode_behavior is EndBehavior.CONTINUE:
            self._maybe_create_item(self._sequence_length, end_of_episode=True)

        elif self._end_of_episode_behavior is EndBehavior.WRITE:
            # Drop episodes that are too short.
            if self._writer.episode_steps < self._sequence_length:
                return
            self._maybe_create_item(
                self._sequence_length, end_of_episode=True, force=True
            )

        elif self._end_of_episode_behavior is EndBehavior.TRUNCATE:
            self._maybe_create_item(
                self._sequence_length - delta, end_of_episode=True, force=True
            )

        elif self._end_of_episode_behavior is EndBehavior.ZERO_PAD:
            zero_step = tree.map_structure(
                lambda x: np.zeros_like(x[-2].numpy()), self._writer.history
            )
            for _ in range(delta):
                self._writer.append(zero_step)

            self._maybe_create_item(
                self._sequence_length, end_of_episode=True, force=True
            )
        else:
            raise ValueError(
                f"Unhandled end of episode behavior: {self._end_of_episode_behavior}."
                " This should never happen, please contact Mava dev team."
            )

    def _maybe_create_item(
        self, sequence_length: int, *, end_of_episode: bool = False, force: bool = False
    ) -> None:

        # Check conditions under which a new item is created.
        first_write = self._writer.episode_steps == sequence_length
        # NOTE(bshahr): the following line assumes that the only way sequence_length
        # is less than self._sequence_length, is if the episode is shorter than
        # self._sequence_length.
        period_reached = self._writer.episode_steps > self._sequence_length and (
            (self._writer.episode_steps - self._sequence_length) % self._period == 0
        )

        if not first_write and not period_reached and not force:
            return

        # TODO(b/183945808): will need to change to adhere to the new protocol.
        if not end_of_episode:
            get_traj = operator.itemgetter(slice(-sequence_length - 1, -1))
        else:
            get_traj = operator.itemgetter(slice(-sequence_length, None))

        history = self._writer.history
        trajectory = base.Trajectory(**tree.map_structure(get_traj, history))

        # Compute priorities for the buffer.
        table_priorities = utils.calculate_priorities(self._priority_fns, trajectory)

        # Create a prioritized item for each table.
        for table_name, priority in table_priorities.items():
            self._writer.create_item(table_name, priority, trajectory)
            self._writer.flush(self._max_in_flight_items)

    # TODO(bshahr): make this into a standalone method. Class methods should be
    # used as alternative constructors or when modifying some global state,
    # neither of which is done here.
    @classmethod
    def signature(  # type: ignore
        cls,
        environment_spec: specs.EnvironmentSpec,
        sequence_length: Optional[int] = None,
        extras_spec: NestedSpec = (),
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

        spec_step = base.Step(
            observations=obs_specs,
            actions=act_specs,
            rewards=reward_specs,
            discounts=step_discount_specs,
            start_of_episode=soe_spec,
            extras=extras_spec,
        )

        return tree.map_structure_with_path(base.spec_like_to_tensor_spec, spec_step)
