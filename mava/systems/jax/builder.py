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

# TODO (Arnu): remove once we figure out the no attribute error.
# TODO (Arnu): reintroduce proper return types, e.g. data/parameter server once those
# have been created.
# type: ignore

"""Jax-based Mava system builder implementation."""

from typing import Any, List

from mava.core_jax import SystemBuilder


class Builder(SystemBuilder):
    def __init__(
        self,
        components: List[Any],
    ) -> None:
        """System building init

        Args:
            components: system callback components
        """

        self.callbacks = components

        self.on_building_init_start()

        self.on_building_init()

        self.on_building_init_end()

    def data_server(self) -> List[Any]:
        """Data server to store and serve transition data from and to system.

        Returns:
            System data server
        """

        # start of make replay tables
        self.on_building_tables_start()

        # make adder signature
        self.on_building_tables_adder_signature()

        # make rate limiter
        self.on_building_tables_rate_limiter()

        # make tables
        self.on_building_tables_make_tables()

        # end of make replay tables
        self.on_building_tables_end()

        return self.system_data_server

    def parameter_server(self) -> Any:
        """Parameter server to store and serve system network parameters.

        Args:
            extra_nodes : additional nodes to add to a launchpad program build
        Returns:
            System parameter server
        """

        # start of make parameter server
        self.on_building_parameter_server_start()

        # make parameter server
        self.on_building_parameter_server_make_parameter_server()

        # end of make parameter server
        self.on_building_parameter_server_end()

        return self.system_parameter_server

    def executor(
        self, executor_id: str, data_server_client: Any, parameter_server_client: Any
    ) -> Any:
        """Executor, a collection of agents in an environment to gather experience.

        Args:
            executor_id : id to identify the executor process for logging purposes
            data_server_client : data server client for pushing transition data
            parameter_server_client : parameter server client for pulling parameters
        Returns:
            System executor
        """

        self._executor_id = executor_id
        self._data_server_client = data_server_client
        self._parameter_server_client = parameter_server_client
        self._evaluator = False

        if self._executor_id == "evaluator":
            self._evaluator = True
        else:
            # ADDER
            # start of make adder
            self.on_building_adder_start()

            # make adder signature
            self.on_building_adder_set_priority()

            # make rate limiter
            self.on_building_adder_make_adder()

            # end of make adder
            self.on_building_adder_end()

        # EXECUTOR
        # start of making the executor
        self.on_building_executor_start()

        # make executor logger
        self.on_building_executor_logger()

        # make executor parameter client
        self.on_building_executor_parameter_client()

        # make executor
        self.on_building_executor_make_executor()

        # make copy of environment
        self.on_building_executor_environment()

        # make train loop
        self.on_building_executor_train_loop()

        # end of making the executor
        self.on_building_executor_end()

        return self.system_executor

    def trainer(
        self, trainer_id: str, data_server_client: Any, parameter_server_client: Any
    ) -> Any:
        """Trainer, a system process for updating agent specific network parameters.

        Args:
            trainer_id : id to identify the trainer process for logging purposes
            data_server_client : data server client for pulling transition data
            parameter_server_client : parameter server client for pushing parameters
        Returns:
            System trainer
        """

        self._trainer_id = trainer_id
        self._data_server_client = data_server_client
        self._table_name = f"table_{trainer_id}"
        self._parameter_server_client = parameter_server_client

        # DATASET
        # start of make dataset iterator
        self.on_building_dataset_start()

        # make dataset
        self.on_building_dataset_make_dataset()

        # end of make dataset iterator
        self.on_building_dataset_end()

        # TRAINER
        # start of making the trainer
        self.on_building_trainer_start()

        # make trainer logger
        self.on_building_trainer_logger()

        # make dataset
        self.on_building_trainer_dataset()

        # make trainer parameter client
        self.on_building_trainer_parameter_client()

        # make trainer
        self.on_building_trainer_make_trainer()

        # end of making the trainer
        self.on_building_trainer_end()

        return self.system_trainer

    def build(self) -> None:
        """Construct program nodes."""
        # start of system building
        self.on_building_start()

        # build program nodes
        self.on_building_program_nodes()

        # end of system building
        self.on_building_end()

    def launch(self) -> None:
        """Run the graph program."""
        # start of system launch
        self.on_building_launch_start()

        # launch system
        self.on_building_launch_distributor()

        # end of system launch
        self.on_building_launch_end()
