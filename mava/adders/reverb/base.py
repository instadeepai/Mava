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


def get_trajectory_net_agents(
    trajectory: Union[Trajectory, mava_types.Transition],
    trajectory_net_keys: Dict[str, str],
) -> Tuple[List, Dict[str, List]]:
    """Returns a dictionary that maps network_keys to a list of agents using that specific
    network.

    Args:
        trajectory: Episode experience recorded by
        the adders.
        trajectory_net_keys: The network_keys used by each agent in the trajectory.
    Returns:
        agents: A sorted list of all the agent_keys.
        agents_per_network: A dictionary that maps network_keys to
        a list of agents using that specific network.
    """
    agents = sort_str_num(trajectory.actions.keys())
    unique_nets = sort_str_num(set(trajectory_net_keys.values()))
    agents_per_network: Dict[str, List] = {key: [] for key in unique_nets}
    for agent in agents:
        agents_per_network[trajectory_net_keys[agent]].append(agent)
    return agents, agents_per_network


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

    def write_experience_to_tables(  # noqa
        self,
        trajectory: Union[Trajectory, mava_types.Transition],
        table_priorities: Dict[str, Any],
    ) -> None:
        """Write an episode experience (trajectory) to the reverb tables. Each
        table represents experience used by each of the trainers. Therefore
        this function dynamically determines to which table(s) to write
        parts of the trajectory based on what networks where used by
        the agents in the episode run.
        Args:
            trajectory: Trajectory to be
                written to the reverb tables.
            table_priorities: A dictionary that maps table names to priorities.
        """

        if self._table_network_config:
            # This if statement activates if the table_network_config
            # is specified. If it is not the write_experience_to_tables
            # function defaults back to just writing the entire
            # trajectory to one default table.

            # Get the networks use by each agent by
            # converting the network_int_keys to strings.
            traj_extras = trajectory.extras["network_int_keys"]
            trajectory_net_keys = {}
            agents = sort_str_num(trajectory.actions.keys())
            for agent in agents:
                arr = traj_extras[agent].numpy()
                if type(trajectory) == Step:
                    # Sequential adder case.
                    trajectory_net_keys[agent] = self._net_ids_to_keys[arr[0]]
                else:
                    # Transition adder case.
                    trajectory_net_keys[agent] = self._net_ids_to_keys[arr]

            # Get a list of the agents and mapping from net_keys to all
            # agents using that network.
            agents, trajectory_nets_agent = get_trajectory_net_agents(
                trajectory=trajectory, trajectory_net_keys=trajectory_net_keys
            )

            # Flag to check if all experience was used
            created_item = False

            # In this loop we go through every table one by one.
            # We check for each table if the experience contains
            # the correct networks for that trainer. If it does
            # we add to the table a subset of the trajectory's agents
            # which only contains the networks combination that the
            # trainer is interested in. Note that a table might
            # find multiple copies of the correct network combination
            # and therefore might write more than once to a table
            # for a given experience. The table might also not
            # write at all for a given trajectory. The created_item
            # flag checks that at least one table used some of the
            # experience in the trajectory.
            for table, priority in table_priorities.items():
                # Copy the original trajectory_nets_agent as we are going to pop
                # form it for each table. Therefore each table starts with
                # a fresh copy of all the agents and removes agents as it
                # pushes the experience to its table.
                trajectory_dict_copy = copy.deepcopy(trajectory_nets_agent)

                # While the networks are in the data keep creating tables
                # Each training example can therefore create multiple items
                is_in_entry = True
                while is_in_entry:
                    # Go through all the networks in the table specification.
                    # Now check if every network in this table specification is used
                    # atleast once by the remaining agents in the trajectory.
                    # Pop the agents from the trajectory, that uses the required
                    # networks and add them to item_agents. If all the
                    # networks was found item_agents will be written
                    # to the table. So basically we try to find a group of
                    # agents that matches the network specification of the table.
                    # We do this until the table cannot find a match in the remaining
                    # agents and therefore exists the and gives another table a chance
                    # to find a matches.
                    item_agents = []
                    for net_key in self._table_network_config[table]:
                        if (
                            net_key in trajectory_dict_copy
                            and len(trajectory_dict_copy[net_key]) > 0
                        ):
                            item_agents.append(trajectory_dict_copy[net_key].pop())
                        else:
                            is_in_entry = False
                            break

                    if is_in_entry:
                        # Write the subset of the trajectory experience to
                        # the table. The below code creates a new Step/Transition
                        # with only the agents with the correct network combination
                        # in order. This new Step/Transition is then written to
                        # the table.
                        created_item = True

                        # Create new empty transition
                        if type(trajectory) == Step:
                            # Create a new sequence trajectory
                            soe = trajectory.start_of_episode  # type: ignore
                            new_trajectory = Step(  # type: ignore
                                {},
                                {},
                                {},
                                {},
                                start_of_episode=soe,
                                extras={},
                            )
                        else:
                            # Create a new transition trajectory
                            new_trajectory = mava_types.Transition(  # type: ignore
                                {}, {}, {}, {}, {}, {}, {}
                            )

                        # Initialise empty extras
                        for key in trajectory.extras.keys():
                            new_trajectory.extras[key] = {}

                            if type(trajectory) == mava_types.Transition:
                                new_trajectory.next_extras[key] = {}  # type: ignore

                        # Go through each of the agents in item_agents and add them
                        # to the new trajectory in the correct spot based on their
                        # networks.
                        for a_i in range(len(item_agents)):
                            # Write the agent to the new trajectory.
                            cur_agent = item_agents[a_i]
                            want_agent = agents[a_i]
                            new_trajectory.observations[
                                want_agent
                            ] = trajectory.observations[cur_agent]
                            new_trajectory.actions[want_agent] = trajectory.actions[
                                cur_agent
                            ]
                            new_trajectory.rewards[want_agent] = trajectory.rewards[
                                cur_agent
                            ]
                            new_trajectory.discounts[want_agent] = trajectory.discounts[
                                cur_agent
                            ]

                            if type(trajectory) == mava_types.Transition:
                                new_trajectory.next_observations[  # type: ignore
                                    want_agent
                                ] = trajectory.next_observations[  # type: ignore
                                    cur_agent
                                ]  # type: ignore

                            # Write this agent to the extras of the new trajectory.
                            for key in trajectory.extras.keys():
                                if (
                                    type(trajectory.extras[key]) is dict
                                    and cur_agent in trajectory.extras[key]
                                ):
                                    new_trajectory.extras[key][
                                        want_agent
                                    ] = trajectory.extras[key][cur_agent]
                                else:
                                    # TODO: (dries) Only actually need to do this once
                                    # and not per agent. Maybe fix this in the future.
                                    new_trajectory.extras[key] = trajectory.extras[key]
                                if type(trajectory) == mava_types.Transition:
                                    ext = trajectory.next_extras[key]  # type: ignore
                                    if (
                                        type(ext) is dict  # type: ignore
                                        and cur_agent in ext  # type: ignore
                                    ):
                                        new_trajectory.next_extras[key][  # type: ignore
                                            want_agent
                                        ] = trajectory.next_extras[  # type: ignore
                                            key
                                        ][  # type: ignore
                                            cur_agent
                                        ]  # type: ignore
                                    else:
                                        # TODO: (dries) Only actually need to
                                        # do this once and not per agent. Maybe
                                        # fix this in the future.
                                        new_trajectory.next_extras[  # type: ignore
                                            key
                                        ] = trajectory.next_extras[  # type: ignore
                                            key
                                        ]  # type: ignore

                        # Write the new_trajectory to the table.
                        self._writer.create_item(
                            table=table, priority=priority, trajectory=new_trajectory
                        )
            if not created_item:
                raise EOFError(
                    "This experience was not used by any trainer: ",
                    trajectory.actions.keys(),
                )
        else:
            # Default setting (deprecate this) with only one table. In this setting
            # we write the entire trajectory to that table.
            for table_name, priority in table_priorities.items():
                self._writer.create_item(
                    table=table_name, priority=priority, trajectory=trajectory
                )

        # Flush the writer.
        self._writer.flush(self._max_in_flight_items)

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
