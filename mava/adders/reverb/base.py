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

from typing import Callable, Dict, Iterable, Mapping, NamedTuple, Optional, Tuple, Union

import dm_env
import numpy as np
import reverb
import tensorflow as tf
import tree
from acme import specs as acme_specs
from acme import types
from acme.adders.reverb.base import ReverbAdder

DEFAULT_PRIORITY_TABLE = "priority_table"


class Step(NamedTuple):
    """Step class used internally for reverb adders."""

    observations: Dict[str, types.NestedArray]
    actions: Dict[str, types.NestedArray]
    rewards: Dict[str, types.NestedArray]
    discounts: Dict[str, types.NestedArray]
    start_of_episode: Union[bool, acme_specs.Array, tf.Tensor, Tuple[()]]
    extras: Dict[str, types.NestedArray]


Trajectory = Step


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
    """Convert spec like object to tensorspec.

    Args:
        paths (Iterable[str]): Spec like path.
        spec (acme_specs.Array): Spec to use.

    Returns:
        tf.TypeSpec: Returned tensorspec.
    """
    return tf.TensorSpec.from_spec(spec, name="/".join(str(p) for p in paths))


class ReverbParallelAdder(ReverbAdder):
    """Base reverb class."""

    def __init__(
        self,
        client: reverb.Client,
        max_sequence_length: int,
        max_in_flight_items: int,
        delta_encoded: bool = False,
        priority_fns: Optional[PriorityFnMapping] = None,
        get_signature_timeout_ms: int = 300_000,
        use_next_extras: bool = True,
    ):
        """Reverb Base Adder.

        Args:
            client (reverb.Client): Client to access reverb.
            max_sequence_length (int): The number of observations that can be added to
                replay.
            max_in_flight_items (int): The maximum number of "in flight" items
                at the same time. See `block_until_num_items` in
                `reverb.TrajectoryWriter.flush` for more info.
            delta_encoded (bool, optional): Enables delta encoding, see `Client` for
                more information. Defaults to False.
            priority_fns (Optional[PriorityFnMapping], optional): A mapping from
                table names to priority functions. Defaults to None.
            get_signature_timeout_ms (int, optional): Timeout while fetching
                signature. Defaults to 300_000.
            use_next_extras (bool, optional): Whether to use extras or not. Defaults to
                True.
        """
        super().__init__(
            client=client,
            max_sequence_length=max_sequence_length,
            max_in_flight_items=max_in_flight_items,
            delta_encoded=delta_encoded,
            priority_fns=priority_fns,
            get_signature_timeout_ms=get_signature_timeout_ms,
        )
        self._use_next_extras = use_next_extras

    def add_first(
        self, timestep: dm_env.TimeStep, extras: Dict[str, types.NestedArray] = {}
    ) -> None:
        """Record the first observation of a trajectory."""
        if not timestep.first():
            raise ValueError(
                "adder.add_first with an initial timestep (i.e. one for "
                "which timestep.first() is True"
            )

        # Record the next observation but leave the history buffer row open by
        # passing `partial_step=True`.
        add_dict = dict(
            observations=timestep.observation,
            start_of_episode=timestep.first(),
        )

        if self._use_next_extras:
            add_dict["extras"] = extras

        self._writer.append(
            add_dict,
            partial_step=True,
        )
        self._add_first_called = True

    def add(
        self,
        actions: Dict[str, types.NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """Record an action and the following timestep."""
        if not self._add_first_called:
            raise ValueError("adder.add_first must be called before adder.add.")

        # Add the timestep to the buffer.
        current_step = dict(
            # Observations was passed at the previous add call.
            actions=actions,
            rewards=next_timestep.reward,
            discounts=next_timestep.discount,
            # Start of episode indicator was passed at the previous add call.
        )

        if not self._use_next_extras:
            current_step["extras"] = next_extras

        self._writer.append(current_step)

        # Record the next observation and write.
        next_step = dict(
            observations=next_timestep.observation,
            start_of_episode=next_timestep.first(),
        )

        if self._use_next_extras:
            next_step["extras"] = next_extras

        self._writer.append(
            next_step,
            partial_step=True,
        )
        self._write()

        if next_timestep.last():
            # Complete the row by appending zeros to remaining open fields.
            # TODO(acme): remove this when fields are no longer expected to be
            # of equal length on the learner side.
            dummy_step = tree.map_structure(np.zeros_like, current_step)
            self._writer.append(dummy_step)
            self._write_last()
            self.reset()
