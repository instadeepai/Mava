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

import copy
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
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
from acme.adders.reverb.base import ReverbAdder

from mava import types as mava_types
from mava.utils.sort_utils import sort_str_num

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


def get_trans_net_agents(
    trajectory: Union[Trajectory, mava_types.Transition], entry_net_keys: Dict[str, str]
) -> Tuple[List, Dict[str, List]]:
    agents = sort_str_num(trajectory.actions.keys())
    unique_nets = sort_str_num(set(entry_net_keys.values()))
    trans_nets_agent: Dict[str, List] = {key: [] for key in unique_nets}
    for agent in agents:
        trans_nets_agent[entry_net_keys[agent]].append(agent)
    return agents, trans_nets_agent


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
            max_in_flight_items (int): [description]
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
            # TODO(Kale-ab) Re-add this when using newer version of acme.
            # get_signature_timeout_ms=get_signature_timeout_ms
        )
        self._use_next_extras = use_next_extras

    def write_experience_to_tables(  # noqa
        self,
        trajectory: Union[Trajectory, mava_types.Transition],
        table_priorities: Dict[str, Any],
    ) -> None:
        # Get a dictionary of the transition nets and agents.
        if self._table_network_config:
            entry_extras = trajectory.extras["network_int_keys"]
            entry_net_keys = {}
            agents = sort_str_num(trajectory.actions.keys())
            for agent in agents:
                arr = entry_extras[agent].numpy()
                if type(trajectory) == Step:
                    entry_net_keys[agent] = self._int_to_nets[arr[0]]
                else:
                    entry_net_keys[agent] = self._int_to_nets[arr]

            # Get the unique agents and mapping from net_keys to agents.
            agents, trans_nets_agent = get_trans_net_agents(
                trajectory=trajectory, entry_net_keys=entry_net_keys
            )

        # Check if experience was used
        created_item = False

        # If all the network entries of a trainer is in the data then add it to that
        # trainer's data table.
        for table, priority in table_priorities.items():
            if self._table_network_config is None:
                created_item = True
                self._writer.create_item(
                    table=table, priority=priority, trajectory=trajectory
                )
            else:
                # Check if all the networks are in trans_nets_agent.
                trans_dict_copy = copy.deepcopy(trans_nets_agent)

                # While the networks are in the data keep creating tables
                # Each training example can therefore create multiple items
                is_in_entry = True
                while is_in_entry:
                    item_agents = []
                    for net_key in self._table_network_config[table]:
                        if (
                            net_key in trans_dict_copy
                            and len(trans_dict_copy[net_key]) > 0
                        ):
                            item_agents.append(trans_dict_copy[net_key].pop())
                        else:
                            is_in_entry = False
                            break

                    if is_in_entry:
                        created_item = True

                        # Create new empty transition
                        if type(trajectory) == Step:
                            soe = trajectory.start_of_episode  # type: ignore
                            new_trans = Step(  # type: ignore
                                {},
                                {},
                                {},
                                {},
                                start_of_episode=soe,
                                extras={},
                            )
                        else:
                            new_trans = mava_types.Transition(  # type: ignore
                                {}, {}, {}, {}, {}, {}, {}
                            )

                        for key in trajectory.extras.keys():
                            new_trans.extras[key] = {}

                            if type(trajectory) == mava_types.Transition:
                                new_trans.next_extras[key] = {}  # type: ignore

                        for a_i in range(len(item_agents)):
                            cur_agent = item_agents[a_i]
                            want_agent = agents[a_i]
                            new_trans.observations[
                                want_agent
                            ] = trajectory.observations[cur_agent]
                            new_trans.actions[want_agent] = trajectory.actions[
                                cur_agent
                            ]
                            new_trans.rewards[want_agent] = trajectory.rewards[
                                cur_agent
                            ]
                            new_trans.discounts[want_agent] = trajectory.discounts[
                                cur_agent
                            ]

                            if type(trajectory) == mava_types.Transition:
                                new_trans.next_observations[  # type: ignore
                                    want_agent
                                ] = trajectory.next_observations[  # type: ignore
                                    cur_agent
                                ]  # type: ignore
                            # Convert extras
                            for key in trajectory.extras.keys():
                                new_trans.extras[key][want_agent] = trajectory.extras[
                                    key
                                ][cur_agent]
                                if type(trajectory) == mava_types.Transition:
                                    new_trans.next_extras[key][  # type: ignore
                                        want_agent
                                    ] = trajectory.next_extras[  # type: ignore
                                        key
                                    ][  # type: ignore
                                        cur_agent
                                    ]  # type: ignore

                        self._writer.create_item(
                            table=table, priority=priority, trajectory=new_trans
                        )
        self._writer.flush(self._max_in_flight_items)
        if not created_item:
            raise EOFError(
                "This experience was not used by any trainer: ",
                trajectory.actions.keys(),
            )

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
