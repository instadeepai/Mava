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

"""Abstract mixin class used to call system component hooks."""

from abc import ABC
from typing import List


class CallbackHookMixin(ABC):

    ######################
    # system builder hooks
    ######################

    callbacks: List

    # INITIALISATION
    def on_building_init_start(self) -> None:
        """Start of builder initialisation."""
        for callback in self.callbacks:
            callback.on_building_init_start(self)

    def on_building_init(self) -> None:
        """Builder initialisation."""
        for callback in self.callbacks:
            callback.on_building_init(self)

    def on_building_init_end(self) -> None:
        """End of builder initialisation."""
        for callback in self.callbacks:
            callback.on_building_init_end(self)

    # DATA SERVER
    def on_building_data_server_start(self) -> None:
        """Start of data server table building."""
        for callback in self.callbacks:
            callback.on_building_data_server_start(self)

    def on_building_data_server_adder_signature(self) -> None:
        """Building of table adder signature."""
        for callback in self.callbacks:
            callback.on_building_data_server_adder_signature(self)

    def on_building_data_server_rate_limiter(self) -> None:
        """Building of table rate limiter."""
        for callback in self.callbacks:
            callback.on_building_data_server_rate_limiter(self)

    def on_building_data_server(self) -> None:
        """Building system data server tables."""
        for callback in self.callbacks:
            callback.on_building_data_server(self)

    def on_building_data_server_end(self) -> None:
        """End of data server table building."""
        for callback in self.callbacks:
            callback.on_building_data_server_end(self)

    # PARAMETER SERVER
    def on_building_parameter_server_start(self) -> None:
        """Start of building parameter server."""
        for callback in self.callbacks:
            callback.on_building_parameter_server_start(self)

    def on_building_parameter_server(self) -> None:
        """Building system parameter server."""
        for callback in self.callbacks:
            callback.on_building_parameter_server(self)

    def on_building_parameter_server_end(self) -> None:
        """End of building parameter server."""
        for callback in self.callbacks:
            callback.on_building_parameter_server_end(self)

    # EXECUTOR
    def on_building_executor_start(self) -> None:
        """Start of building executor."""
        for callback in self.callbacks:
            callback.on_building_executor_start(self)

    def on_building_executor_adder_priority(self) -> None:
        """Building adder priority function."""
        for callback in self.callbacks:
            callback.on_building_executor_adder_priority(self)

    def on_building_executor_adder(self) -> None:
        """Building executor adder."""
        for callback in self.callbacks:
            callback.on_building_executor_adder(self)

    def on_building_executor_logger(self) -> None:
        """Building executor logger."""
        for callback in self.callbacks:
            callback.on_building_executor_logger(self)

    def on_building_executor_parameter_client(self) -> None:
        """Building executor parameter server client."""
        for callback in self.callbacks:
            callback.on_building_executor_parameter_client(self)

    def on_building_executor(self) -> None:
        """Building system executor."""
        for callback in self.callbacks:
            callback.on_building_executor(self)

    def on_building_executor_environment(self) -> None:
        """Building executor environment copy."""
        for callback in self.callbacks:
            callback.on_building_executor_environment(self)

    def on_building_executor_environment_loop(self) -> None:
        """Building executor system-environment loop."""
        for callback in self.callbacks:
            callback.on_building_executor_environment_loop(self)

    def on_building_executor_end(self) -> None:
        """End of building executor."""
        for callback in self.callbacks:
            callback.on_building_executor_end(self)

    # TRAINER
    def on_building_trainer_start(self) -> None:
        """Start of building trainer."""
        for callback in self.callbacks:
            callback.on_building_trainer_start(self)

    def on_building_trainer_logger(self) -> None:
        """Building trainer logger."""
        for callback in self.callbacks:
            callback.on_building_trainer_logger(self)

    def on_building_trainer_dataset(self) -> None:
        """Building trainer dataset."""
        for callback in self.callbacks:
            callback.on_building_trainer_dataset(self)

    def on_building_trainer_parameter_client(self) -> None:
        """Building trainer parameter server client."""
        for callback in self.callbacks:
            callback.on_building_trainer_parameter_client(self)

    def on_building_trainer(self) -> None:
        """Building trainer."""
        for callback in self.callbacks:
            callback.on_building_trainer(self)

    def on_building_trainer_end(self) -> None:
        """End of building trainer."""
        for callback in self.callbacks:
            callback.on_building_trainer_end(self)

    # BUILD
    def on_building_start(self) -> None:
        """Start of system graph program build."""
        for callback in self.callbacks:
            callback.on_building_start(self)

    def on_building_program_nodes(self) -> None:
        """Building system graph program nodes."""
        for callback in self.callbacks:
            callback.on_building_program_nodes(self)

    def on_building_end(self) -> None:
        """End of system graph program build."""
        for callback in self.callbacks:
            callback.on_building_end(self)

    # LAUNCH
    def on_building_launch_start(self) -> None:
        """Start of system launch."""
        for callback in self.callbacks:
            callback.on_building_launch_start(self)

    def on_building_launch(self) -> None:
        """System launch."""
        for callback in self.callbacks:
            callback.on_building_launch(self)

    def on_building_launch_end(self) -> None:
        """End of system launch."""
        for callback in self.callbacks:
            callback.on_building_launch_end(self)

    ###############################
    # system parameter server hooks
    ###############################

    def on_parameter_server_init_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_server_init_start(self)

    def on_parameter_server_init(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_server_init(self)

    def on_parameter_server_init_checkpointer(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_server_init_checkpointer(self)

    def on_parameter_server_init_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_server_init_end(self)

    def on_parameter_server_get_parameters_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_server_get_parameters_start(self)

    def on_parameter_server_get_parameters(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_server_get_parameters(self)

    def on_parameter_server_get_parameters_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_server_get_parameters_end(self)

    def on_parameter_server_set_parameters_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_server_set_parameters_start(self)

    def on_parameter_server_set_parameters(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_server_set_parameters(self)

    def on_parameter_server_set_parameters_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_server_set_parameters_end(self)

    def on_parameter_server_add_to_parameters_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_server_add_to_parameters_start(self)

    def on_parameter_server_add_to_parameters(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_server_add_to_parameters(self)

    def on_parameter_server_add_to_parameters_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_server_add_to_parameters_end(self)

    def on_parameter_server_run_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_server_run_start(self)

    def on_parameter_server_run_loop_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_server_run_loop_start(self)

    def on_parameter_server_run_loop_checkpoint(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_server_run_loop_checkpoint(self)

    def on_parameter_server_run_loop(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_server_run_loop(self)

    def on_parameter_server_run_loop_termination(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_server_run_loop_termination(self)

    def on_parameter_server_run_loop_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_server_run_loop_end(self)

    ###############################
    # system parameter client hooks
    ###############################

    def on_parameter_client_init_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_client_init_start(self)

    def on_parameter_client_init(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_client_init(self)

    def on_parameter_client_init_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_client_init_end(self)

    def on_parameter_client_get_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_client_get_start(self)

    def on_parameter_client_get(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_client_get(self)

    def on_parameter_client_get_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_client_get_end(self)

    def on_parameter_client_set_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_client_set_start(self)

    def on_parameter_client_set(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_client_set(self)

    def on_parameter_client_set_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_client_set_end(self)

    def on_parameter_client_set_and_get_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_client_set_and_get_start(self)

    def on_parameter_client_set_and_get(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_client_set_and_get(self)

    def on_parameter_client_set_and_get_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_client_set_and_get_end(self)

    def on_parameter_client_add_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_client_add_start(self)

    def on_parameter_client_add(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_client_add(self)

    def on_parameter_client_add_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_client_add_end(self)

    def on_parameter_client_add_and_wait_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_client_add_and_wait_start(self)

    def on_parameter_client_add_and_wait(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_client_add_and_wait(self)

    def on_parameter_client_add_and_wait_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_client_add_and_wait_end(self)

    def on_parameter_client_get_and_wait_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_client_get_and_wait_start(self)

    def on_parameter_client_get_and_wait(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_client_get_and_wait(self)

    def on_parameter_client_get_and_wait_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_client_get_and_wait_end(self)

    def on_parameter_client_get_all_and_wait_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_client_get_all_and_wait_start(self)

    def on_parameter_client_get_all_and_wait(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_client_get_all_and_wait(self)

    def on_parameter_client_get_all_and_wait_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_client_get_all_and_wait_end(self)

    def on_parameter_client_set_and_wait_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_client_set_and_wait_start(self)

    def on_parameter_client_set_and_wait(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_client_set_and_wait(self)

    def on_parameter_client_set_and_wait_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_client_set_and_wait_end(self)
