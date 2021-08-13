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

import copy
import time
from typing import Any, Dict, Sequence, Union

import numpy as np
import tensorflow as tf

from mava.systems.tf.mad4pg.execution import MAD4PGRecurrentExecutor
from mava.systems.tf.mad4pg.system import MAD4PG
from mava.systems.tf.maddpg.builder import MADDPGBuilder
from mava.systems.tf.maddpg.execution import MADDPGRecurrentExecutor
from mava.systems.tf.maddpg.system import MADDPG
from mava.systems.tf.variable_sources import VariableSource as MavaVariableSource
from mava.utils.sort_utils import sample_new_agent_keys, sort_str_num

"""Best response training interface for multi-agent RL systems."""
supported_br_systems = [MADDPG, MAD4PG]
supported_br_executors = [MADDPGRecurrentExecutor, MAD4PGRecurrentExecutor]


# BR variable source
class BRVariableSource(MavaVariableSource):
    def __init__(
        self,
        variables: Dict[str, Any],
        checkpoint: bool,
        checkpoint_subpath: str,
        unique_net_keys: Sequence[str],
    ) -> None:
        """Initialise the variable source
        Args:
            variables (Dict[str, Any]): a dictionary with
            variables which should be stored in it.
            checkpoint (bool): Indicates whether checkpointing should be performed.
            checkpoint_subpath (str): checkpoint path
        Returns:
            None
        """
        # TODO: Fix these values
        self._br_update_interval = 10
        self._br_update_rate = 0.001
        self._checkpoint_interval = 10 * 60

        self._last_checkpoint_time = time.time()
        self._last_br_update_time = time.time()
        self._unique_net_keys = unique_net_keys
        super().__init__(
            variables=variables,
            checkpoint=checkpoint,
            checkpoint_subpath=checkpoint_subpath,
        )

    def run(self) -> None:
        """Run the variable source. This function allows for
        checkpointing and other centralised computations to
        be performed by the variable source.
                Args:
                    None
                Returns:
                    None
        """
        # Checkpoints every 5 minutes
        while True:
            # Wait 10 seconds before checking again
            time.sleep(10)

            # Check if system should checkpoint (+1 second to make sure the
            # checkpointer does the save)
            if (
                self._system_checkpointer
                and self._last_checkpoint_time + self._checkpoint_interval + 1
                < time.time()
            ):
                self._last_checkpoint_time = time.time()
                self._system_checkpointer.save()
                tf.print("Saved latest variables.")

            # Update the best response networks
            if self._last_br_update_time + self._br_update_interval < time.time():
                tf.print("Update br agent.")
                self._last_br_update_time = time.time()

                network_type_keys = ["policies", "observations"]
                tau = self._br_update_rate
                for network_type in network_type_keys:
                    add_nets = []
                    for var_key in self.variables.keys():
                        if (
                            network_type in var_key
                            and "br_network" not in var_key
                            and "target" not in var_key
                        ):
                            add_nets.append(var_key)

                    # Loop through tuple
                    br_key = f"br_network_{network_type}"
                    for var_i in range(len(self.variables[br_key])):
                        var_sum = tf.zeros(self.variables[br_key][var_i].shape)
                        for var_key in add_nets:
                            var_sum += self.variables[var_key][var_i]
                        self.variables[br_key][var_i].assign(
                            self.variables[br_key][var_i] * (1.0 - tau)
                            + tau * var_sum / len(add_nets)
                        )


def BestResponseWrapper(  # noqa
    system: Union[MADDPG, MAD4PG],  # noqa
) -> Union[MADDPG, MAD4PG]:
    """Initializes the broadcaster communicator.
    Args:
        system: The system that should be wrapped.
    Returns:
        system: The wrapped system.
    """
    if type(system) not in supported_br_systems:
        raise NotImplementedError(
            f"Currently only the {supported_br_systems} systems have "
            f"the correct hooks to support this wrapper. Not {type(system)}."
        )

    if system._builder._executor_fn not in supported_br_executors:
        raise NotImplementedError(
            f"Currently only the {supported_br_executors} executors have "
            f"the correct hooks to support this wrapper. "
            f"Not {system._builder._executor_fn}."
        )

    # Assuming that all the networks have the
    # same specifications
    system._net_spec_keys["br_network"] = list(system._net_spec_keys.values())[0]

    # Wrap the executor with the BR hooks
    class BRExecutor(system._builder._executor_fn):  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)

        # BR executor
        def sample_new_keys(self) -> None:
            """Sample new keys for the network ints."""
            agent_list = list(self._agent_net_keys.keys())  # type: ignore
            agent_slots = sort_str_num(agent_list)

            assert len(agent_slots) % 2 == 0
            half_ind = int(len(agent_slots) / 2)

            # Sample random networks for half of the agents
            self._network_int_keys_extras, self._agent_net_keys = sample_new_agent_keys(
                agent_slots[:half_ind],
                self._executor_samples,
                self._net_to_ints,
            )

            # Set the other half of the agents to use the best response network
            agent_slots = copy.copy(agent_slots[half_ind:])
            net_key = "br_network"
            self._agent_net_keys.update({agent: net_key for agent in agent_slots})
            self._network_int_keys_extras.update(
                {
                    agent: np.array(self._net_to_ints[net_key], dtype=np.int32)
                    for agent in agent_slots
                }
            )

    system._builder._executor_fn = BRExecutor

    # Wrap the system builder with the BR hooks
    class BRBuilder(type(system._builder)):  # type: ignore
        def __init__(
            self,
            builder: MADDPGBuilder,
        ):
            """Initialise the system.
            Args:
                builder: The builder to wrap.
            """

            self.__dict__ = builder.__dict__

            # Update the builder variables
            self._config.net_to_ints["br_network"] = len(self._config.unique_net_keys)
            self._config.unique_net_keys.append("br_network")

        def variable_server_fn(
            self,
            *args: Any,
            **kwargs: Any,
        ) -> BRVariableSource:
            """Create a variable server.
            Args:
                *args: Variable arguments.
                **kwargs: Variable keyword arguments.
            Returns:
                a BRVariableSource.
            """
            return BRVariableSource(*args, **kwargs)

    system._builder = BRBuilder(system._builder)
    return system
