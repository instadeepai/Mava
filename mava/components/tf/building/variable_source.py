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

"""Commonly used adder signature components for system builders"""
from typing import Dict, Optional, Union, List
from mava.components.tf.variables import server

import tensorflow as tf
from acme.tf import utils as tf2_utils

from mava.systems.building import SystemBuilder
from mava.components.building import VariableSource as BaseVariableSource
from mava.systems.variables import VariableClient, VariableServer
from mava.components import variables
from mava.components.tf import variables as tf_variables
from mava.utils.decorators import execution, evaluation


class TFVariableSource(BaseVariableSource):
    def _create_tf_counter_variables(
        self, variables: Dict[str, tf.Variable]
    ) -> Dict[str, tf.Variable]:
        """Create counter variables.
        Args:
            variables (Dict[str, snt.Variable]): dictionary with variable_source
            variables in.
        Returns:
            variables (Dict[str, snt.Variable]): dictionary with variable_source
            variables in.
        """
        variables["trainer_steps"] = tf.Variable(0, dtype=tf.int32)
        variables["trainer_walltime"] = tf.Variable(0, dtype=tf.float32)
        variables["evaluator_steps"] = tf.Variable(0, dtype=tf.int32)
        variables["evaluator_episodes"] = tf.Variable(0, dtype=tf.int32)
        variables["executor_episodes"] = tf.Variable(0, dtype=tf.int32)
        variables["executor_steps"] = tf.Variable(0, dtype=tf.int32)
        return variables


class TFVariableServer(TFVariableSource):
    def __init__(
        self,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
        checkpoint_minute_interval: int = 10,
        termination_condition: Optional[Dict[str, int]] = None,
    ) -> None:
        """[summary]

        Args:
            checkpoint (bool, optional): [description]. Defaults to True.
            checkpoint_subpath (str, optional): [description]. Defaults to "~/mava/".
            checkpoint_minute_interval (int, optional): [description]. Defaults to 10.
        """

        self.checkpoint = checkpoint
        self.checkpoint_subpath = checkpoint_subpath
        self.checkpoint_minute_interval = checkpoint_minute_interval
        self.termination_condition = termination_condition

    def on_building_variable_server_start(self, builder: SystemBuilder) -> None:
        # Create the system
        builder.networks = builder.system()

    def on_building_variable_server_make_variable_server(
        self, builder: SystemBuilder
    ) -> None:
        # Create variables
        vars = {}
        # Network variables
        for net_type_key in builder.networks.keys():
            for net_key in builder.networks[net_type_key].keys():
                # Ensure obs and target networks are sonnet modules
                vars[f"{net_key}_{net_type_key}"] = tf2_utils.to_sonnet_module(
                    builder.networks[net_type_key][net_key]
                ).variables

        vars = self._create_tf_counter_variables(vars)

        # Create variable server components
        server = tf_variables.TFServer(vars)
        terminator = variables.Terminator(self.termination_condition)
        server_components = [server, terminator]
        if self.checkpointer:
            checkpointer = tf_variables.TFCheckpointer(
                self.checkpoint_subpath,
                self.checkpoint_minute_interval,
            )
            server_components.append(checkpointer)

        # Create variable server
        variable_server = VariableServer(server_components)

        builder.variable_server = variable_server


class TFExecutorVariableClient(TFVariableSource):
    def __init__(
        self,
        executor_variable_update_period: int = 1000,
    ) -> None:
        """[summary]

        Args:
            executor_variable_update_period (int, optional): [description]. Defaults to 1000.
        """

        self.executor_variable_update_period = executor_variable_update_period

    def _make_variable_client(self, builder: SystemBuilder):
        # Create policy variables
        vars = {}
        get_keys = []
        for net_type_key in ["observations", "policies"]:
            for net_key in builder.networks[net_type_key].keys():
                var_key = f"{net_key}_{net_type_key}"
                vars[var_key] = builder.networks[net_type_key][net_key].variables
                get_keys.append(var_key)
        vars = self._create_tf_counter_variables(vars)

        count_names = [
            "trainer_steps",
            "trainer_walltime",
            "evaluator_steps",
            "evaluator_episodes",
            "executor_episodes",
            "executor_steps",
        ]
        get_keys.extend(count_names)
        counts = {name: vars[name] for name in count_names}

        variable_client = None
        if builder.variable_server:
            # Create variable client components
            client = tf_variables.TFClient(
                client=builder.variable_server,
                variables=vars,
                get_keys=get_keys,
                update_period=self.executor_variable_update_period,
            )
            client_components = [client]

            # Create variable client
            variable_client = VariableClient(client_components)

            # Make sure not to use a random policy after checkpoint restoration by
            # assigning variables before running the environment loop.
            variable_client.get_and_wait()
        return variable_client, counts

    @execution
    def on_building_executor_variable_client(self, builder: SystemBuilder) -> None:

        variable_client, counts = self._make_variable_client(builder)
        builder.executor_variable_client = variable_client
        builder.executor_counts = counts

    @evaluation
    def on_building_evaluator_variable_client(self, builder: SystemBuilder) -> None:

        variable_client, counts = self._make_variable_client(builder)
        builder.evaluator_variable_client = variable_client
        builder.evaluator_counts = counts


class TFTrainerVariableClient(VariableSource):
    def __init__(self) -> None:
        """[summary]

        Args:
            trainer_networks (Union[ Dict[str, List], enums.Trainer ], optional):
                [description]. Defaults to enums.Trainer.single_trainer.
        """

    def on_building_trainer_variable_client(self, builder: SystemBuilder) -> None:
        # Create variable client
        variables = {}
        set_keys = []
        get_keys = []
        # TODO (dries): Only add the networks this trainer is working with.
        # Not all of them.
        for net_type_key in builder.networks.keys():
            for net_key in builder.networks[net_type_key].keys():
                variables[f"{net_key}_{net_type_key}"] = builder.networks[net_type_key][
                    net_key
                ].variables
                if net_key in set(builder.trainer_networks):
                    set_keys.append(f"{net_key}_{net_type_key}")
                else:
                    get_keys.append(f"{net_key}_{net_type_key}")

        variables = self._create_tf_counter_variables(variables)

        count_names = [
            "trainer_steps",
            "trainer_walltime",
            "evaluator_steps",
            "evaluator_episodes",
            "executor_episodes",
            "executor_steps",
        ]
        get_keys.extend(count_names)
        builder.trainer_counts = {name: variables[name] for name in count_names}

        variable_client = variable_utils.VariableClient(
            client=builder.variable_server,
            variables=variables,
            get_keys=get_keys,
            set_keys=set_keys,
        )

        # Get all the initial variables
        variable_client.get_all_and_wait()

        builder.trainer_variable_client = variable_client