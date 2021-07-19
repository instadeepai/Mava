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
        buffer_size: int,
        max_sequence_length: int,
        delta_encoded: bool = False,
        chunk_length: Optional[int] = None,
        priority_fns: Optional[PriorityFnMapping] = None,
        max_in_flight_items: Optional[int] = 25,
        use_next_extras: bool = True,
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
            priority_fns = {DEFAULT_PRIORITY_TABLE: lambda x: 1.0}

        self._client = client
        self._priority_fns = priority_fns
        self._max_sequence_length = max_sequence_length
        self._delta_encoded = delta_encoded
        self._chunk_length = chunk_length
        self._max_in_flight_items = max_in_flight_items
        self._use_next_extras = use_next_extras

        # This is exposed as the _writer property in such a way that it will create
        # a new writer automatically whenever the internal __writer is None. Users
        # should ONLY ever interact with self._writer.
        self.__writer = None

        # The state of the adder is captured by a buffer of `buffer_size` steps
        # (generally SAR tuples) and one additional dangling observation.
        self._buffer: Deque = collections.deque(maxlen=buffer_size)
        self._next_extras: Union[None, Dict[str, types.NestedArray]] = None
        self._next_observations = None
        self._start_of_episode = False

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

    def add_priority_table(
        self, table_name: str, priority_fn: Optional[PriorityFn]
    ) -> None:
        if table_name in self._priority_fns:
            raise ValueError(
                "A priority function already exists for {}.".format(table_name)
            )
        self._priority_fns[table_name] = priority_fn

    def reset(self) -> None:
        """Resets the adder's buffer."""
        if self.__writer:
            self._writer.close()
            self.__writer = None
        self._buffer.clear()
        self._next_observations = None

    def add_first(
        self, timestep: dm_env.TimeStep, extras: Dict[str, types.NestedArray] = {}
    ) -> None:
        """Record the first observation of a trajectory."""
        if not timestep.first():
            raise ValueError(
                "adder.add_first with an initial timestep (i.e. one for "
                "which timestep.first() is True"
            )

        if self._next_observations is not None:
            raise ValueError(
                "adder.reset must be called before adder.add_first "
                "(called automatically if `next_timestep.last()` is "
                "true when `add` is called)."
            )

        # Record the next observation.
        self._next_observations = timestep.observation
        self._start_of_episode = True

        if self._use_next_extras:
            self._next_extras = extras

    def add(
        self,
        actions: Dict[str, types.NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """Record an action and the following timestep."""
        if self._next_observations is None:
            raise ValueError("adder.add_first must be called before adder.add.")

        discount = next_timestep.discount
        if next_timestep.last():
            # Terminal timesteps created by dm_env.termination() will have a scalar
            # discount of 0.0. This may not match the array shape / nested structure
            # of the previous timesteps' discounts. The below will match
            # next_timestep.discount's shape/structure to that of
            # self._buffer[-1].discount.
            if self._buffer and not tree.is_nested(next_timestep.discount):
                discount = tree.map_structure(
                    lambda d: np.broadcast_to(next_timestep.discount, np.shape(d)),
                    self._buffer[-1].discount,
                )

        self._buffer.append(
            Step(
                observations=self._next_observations,
                actions=actions,
                rewards=next_timestep.reward,
                discounts=discount,
                start_of_episode=self._start_of_episode,
                extras=self._next_extras if self._use_next_extras else next_extras,
            )
        )

        # Write the last "dangling" observation.
        if next_timestep.last():
            self._start_of_episode = False
            self._write()
            self._write_last()
            self.reset()
        else:
            # Record the next observation and write.
            # Possibly store next_extras
            if self._use_next_extras:
                self._next_extras = next_extras
            self._next_observations = next_timestep.observation
            self._start_of_episode = False
            self._write()

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
          core_state_spec: A nested structure with leaf nodes that have `.shape` and
            `.dtype` attributes. The structure (and shapes/dtypes) of this must
            be the same as the `core_state` passed into `ReverbAdder.add`.
        Returns:
          A `Step` whose leaf nodes are `tf.TensorSpec` objects.
        """

    @abc.abstractmethod
    def _write(self) -> None:
        """Write data to replay from the buffer."""

    @abc.abstractmethod
    def _write_last(self) -> None:
        """Write data to replay from the buffer."""
