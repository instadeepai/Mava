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

"""Adders that use Reverb (github.com/deepmind/reverb) as a backend."""

import abc
import collections
from typing import (
    Callable,
    Deque,
    Dict,
    Iterable,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

from absl import logging
import dm_env
import numpy as np
import reverb
import tensorflow as tf
import tree
from acme import specs as acme_specs
from acme import types

from mava import specs as mava_specs
from mava.adders import base

DEFAULT_PRIORITY_TABLE = "priority_table"


class Step(NamedTuple):
    """Step class used internally for reverb adders."""

    observations: Dict[str, types.NestedArray]
    actions: Dict[str, types.NestedArray]
    rewards: Dict[str, types.NestedArray]
    discounts: Dict[str, types.NestedArray]
    start_of_episode: Union[bool, acme_specs.Array, tf.Tensor, Tuple[()]]
    extras: Dict[str, types.NestedArray]


class PriorityFnInput(NamedTuple):
    """The input to a priority function consisting of stacked steps."""

    observations: Dict[str, types.NestedArray]
    actions: Dict[str, types.NestedArray]
    rewards: Dict[str, types.NestedArray]
    discounts: Dict[str, types.NestedArray]
    start_of_episode: types.NestedArray
    extras: Dict[str, types.NestedArray]


# Define the type of a priority function and the mapping from table to function.
PriorityFn = Callable[["PriorityFnInput"], float]
PriorityFnMapping = Mapping[str, Optional[PriorityFn]]


def spec_like_to_tensor_spec(
    paths: Iterable[str], spec: acme_specs.Array
) -> tf.TypeSpec:
    return tf.TensorSpec.from_spec(spec, name="/".join(str(p) for p in paths))


class ReverbParallelAdder(base.ParallelAdder):
    """Base class for Reverb adders."""

    def __init__(
        self,
        client: reverb.Client,
        # buffer_size: int,
        max_sequence_length: int,
        delta_encoded: bool = False,
        # chunk_length: Optional[int] = None,
        priority_fns: Optional[PriorityFnMapping] = None,
        max_in_flight_items: Optional[int] = 25,
    ):
        """Initialize a ReverbAdder instance.
        Args:
          client: A client to the Reverb backend.
          buffer_size: Number of steps to retain in memory.
          max_sequence_length: The maximum length of sequences (corresponding to the
            number of observations) that can be added to replay.
          delta_encoded: If `True` (False by default) enables delta encoding, see
            `Client` for more information.
          chunk_length: Number of timesteps grouped together before delta encoding
            and compression. See `Client` for more information.
          priority_fns: A mapping from table names to priority functions; if
            omitted, all transitions/steps/sequences are given uniform priorities
            (1.0) and placed in DEFAULT_PRIORITY_TABLE.
          max_in_flight_items: The maximum number of items allowed to be "in flight"
            at the same time. See `reverb.Writer.writer` for more info.
        """
        if priority_fns:
            priority_fns = dict(priority_fns)
        else:
            priority_fns = {DEFAULT_PRIORITY_TABLE: None}

        self._client = client
        self._priority_fns = priority_fns
        self._max_sequence_length = max_sequence_length
        self._delta_encoded = delta_encoded
        # self._chunk_length = chunk_length
        self._max_in_flight_items = max_in_flight_items
        self._add_first_called = False
        self._use_next_extras = True

        # This is exposed as the _writer property in such a way that it will create
        # a new writer automatically whenever the internal __writer is None. Users
        # should ONLY ever interact with self._writer.
        self.__writer = None
        # Every time a new writer is created, it must fetch the signature from the
        # Reverb server. If this is set too low it can crash the adders in a
        # distributed setup where the replay may take a while to spin up.
        self._get_signature_timeout_ms = 300_000

        def __del__(self):
            if self.__writer is not None:
                timeout_ms = 10_000
                # Try flush all appended data before closing to avoid loss of experience.
                try:
                    self.__writer.flush(self._max_in_flight_items, timeout_ms=timeout_ms)
                except reverb.DeadlineExceededError as e:
                    logging.error(
                        'Timeout (%d ms) exceeded when flushing the writer before '
                        'deleting it. Caught Reverb exception: %s', timeout_ms, str(e))
                self.__writer.close()

        # The state of the adder is captured by a buffer of `buffer_size` steps
        # (generally SAR tuples) and one additional dangling observation.
        # self._buffer: Deque = collections.deque(maxlen=buffer_size)
        # self._next_extras: Union[None, Dict[str, types.NestedArray]] = None
        # self._next_observations = None
        # self._start_of_episode = False

    def __del__(self) -> None:
        if self.__writer is not None:
            # Explicitly close the writer with no retry on server unavailable.
            # This is to avoid hang on closing if the server has already terminated.
            self.__writer.close(retry_on_unavailable=False)

    @property
    def _writer(self) -> reverb.Writer:
        if self.__writer is None:
            self.__writer = self._client.writer(
                self._max_sequence_length,
                delta_encoded=self._delta_encoded,
                chunk_length=self._chunk_length,
                max_in_flight_items=self._max_in_flight_items,
            )
        return self.__writer

    def add_priority_table(self, table_name: str,
                           priority_fn: Optional[PriorityFn]):
        if table_name in self._priority_fns:
            raise ValueError(
                f'A priority function already exists for {table_name}. '
                f'Existing tables: {", ".join(self._priority_fns.keys())}.'
            )
        self._priority_fns[table_name] = priority_fn

    def reset(self, timeout_ms: Optional[int] = None):
        """Resets the adder's buffer."""
        if self.__writer:
            # Flush all appended data and clear the buffers.
            self.__writer.end_episode(clear_buffers=True, timeout_ms=timeout_ms)
        self._add_first_called = False

    def add_first(
        self, timestep: dm_env.TimeStep, extras: Dict[str, types.NestedArray] = {}
    ) -> None:
        """Record the first observation of a trajectory."""
        if not timestep.first():
            raise ValueError('adder.add_first with an initial timestep (i.e. one for '
                            'which timestep.first() is True')

        # Record the next observation but leave the history buffer row open by
        # passing `partial_step=True`.
        self._writer.append(dict(observation=timestep.observation,
                                extras=extras,
                                start_of_episode=timestep.first()),
                            partial_step=True)
        self._add_first_called = True

    def add(
        self,
        actions: Dict[str, types.NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """Record an action and the following timestep."""
        if self._next_observations is None:
            raise ValueError("adder.add_first must be called before adder.add.")

        # Add the timestep to the buffer.
        current_step = dict(
            # Observations was passed at the previous add call.
            actions=actions,
            reward=next_timestep.reward,
            discount=next_timestep.discount,
            # Start of episode indicator was passed at the previous add call.
        )
        self._writer.append(current_step)

        # Record the next observation and write.
        self._writer.append(
            dict(
                observation=next_timestep.observation,
                **({'extras': next_extras} if next_extras else {}),
                start_of_episode=next_timestep.first()),
            partial_step=True)
        self._write()

        if next_timestep.last():
            # Complete the row by appending zeros to remaining open fields.
            # TODO(b/183945808): remove this when fields are no longer expected to be
            # of equal length on the learner side.
            dummy_step = tree.map_structure(np.zeros_like, current_step)
            self._writer.append(dummy_step)
            self._write_last()
            self.reset()

    @abc.abstractmethod
    def signature(
        cls,
        environment_spec: mava_specs.MAEnvironmentSpec,
        extras_spec: tf.TypeSpec,
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
        spec_step = Step(
        observation=environment_spec.observations,
        action=environment_spec.actions,
        reward=environment_spec.rewards,
        discount=environment_spec.discounts,
        start_of_episode=acme_specs.Array(shape=(), dtype=bool),
        extras=extras_spec)
        return tree.map_structure_with_path(spec_like_to_tensor_spec, spec_step)

    @abc.abstractmethod
    def _write(self) -> None:
        """Write data to replay from the buffer."""

    @abc.abstractmethod
    def _write_last(self) -> None:
        """Write data to replay from the buffer."""
