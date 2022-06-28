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

"""Abstract base class used to build new system components."""

from abc import ABC

from mava.core_jax import (
    SystemBuilder,
    SystemExecutor,
    SystemParameterServer,
    SystemTrainer,
)


class Callback(ABC):
    """Abstract base class used to build new components. \
        Subclass this class and override any of the relevant hooks \
        to create a new system component."""

    ######################
    # system builder hooks
    ######################

    # BUILDER INITIAlISATION
    def on_building_init_start(self, builder: SystemBuilder) -> None:
        """Start of builder initialisation."""
        pass

    def on_building_init(self, builder: SystemBuilder) -> None:
        """Builder initialisation."""
        pass

    def on_building_init_end(self, builder: SystemBuilder) -> None:
        """End of builder initialisation."""
        pass

    # DATA SERVER
    def on_building_data_server_start(self, builder: SystemBuilder) -> None:
        """Start of data server table building."""
        pass

    def on_building_data_server_adder_signature(self, builder: SystemBuilder) -> None:
        """Building of table adder signature."""
        pass

    def on_building_data_server_rate_limiter(self, builder: SystemBuilder) -> None:
        """Building of table rate limiter."""
        pass

    def on_building_data_server(self, builder: SystemBuilder) -> None:
        """Building system data server tables."""
        pass

    def on_building_data_server_end(self, builder: SystemBuilder) -> None:
        """End of data server table building."""
        pass

    # PARAMETER SERVER
    def on_building_parameter_server_start(self, builder: SystemBuilder) -> None:
        """Start of building parameter server."""
        pass

    def on_building_parameter_server(self, builder: SystemBuilder) -> None:
        """Building system parameter server."""
        pass

    def on_building_parameter_server_end(self, builder: SystemBuilder) -> None:
        """End of building parameter server."""
        pass

    # EXECUTOR
    def on_building_executor_start(self, builder: SystemBuilder) -> None:
        """Start of building executor."""
        pass

    def on_building_executor_adder_priority(self, builder: SystemBuilder) -> None:
        """Building adder priority function."""
        pass

    def on_building_executor_adder(self, builder: SystemBuilder) -> None:
        """Building executor adder."""
        pass

    def on_building_executor_logger(self, builder: SystemBuilder) -> None:
        """Building executor logger."""
        pass

    def on_building_executor_parameter_client(self, builder: SystemBuilder) -> None:
        """Building executor parameter server client."""
        pass

    def on_building_executor(self, builder: SystemBuilder) -> None:
        """Building system executor."""
        pass

    def on_building_executor_environment(self, builder: SystemBuilder) -> None:
        """Building executor environment copy."""
        pass

    def on_building_executor_environment_loop(self, builder: SystemBuilder) -> None:
        """Building executor system-environment loop."""
        pass

    def on_building_executor_end(self, builder: SystemBuilder) -> None:
        """End of building executor."""
        pass

    # TRAINER
    def on_building_trainer_start(self, builder: SystemBuilder) -> None:
        """Start of building trainer."""
        pass

    def on_building_trainer_logger(self, builder: SystemBuilder) -> None:
        """Building trainer logger."""
        pass

    def on_building_trainer_dataset(self, builder: SystemBuilder) -> None:
        """Building trainer dataset."""
        pass

    def on_building_trainer_parameter_client(self, builder: SystemBuilder) -> None:
        """Building trainer parameter server client."""
        pass

    def on_building_trainer(self, builder: SystemBuilder) -> None:
        """Building trainer."""
        pass

    def on_building_trainer_end(self, builder: SystemBuilder) -> None:
        """End of building trainer."""
        pass

    # BUILD
    def on_building_start(self, builder: SystemBuilder) -> None:
        """Start of system graph program build."""
        pass

    def on_building_program_nodes(self, builder: SystemBuilder) -> None:
        """Building system graph program nodes."""
        pass

    def on_building_end(self, builder: SystemBuilder) -> None:
        """End of system graph program build."""
        pass

    # LAUNCH
    def on_building_launch_start(self, builder: SystemBuilder) -> None:
        """Start of system launch."""
        pass

    def on_building_launch(self, builder: SystemBuilder) -> None:
        """System launch."""
        pass

    def on_building_launch_end(self, builder: SystemBuilder) -> None:
        """End of system launch."""
        pass

    #######################
    # system executor hooks
    #######################

    def on_execution_init_start(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_init(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_init_end(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_select_action_start(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_select_action_preprocess(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_select_action_compute(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_select_action_sample(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_select_action_end(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_observe_first_start(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_observe_first(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_observe_first_end(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_observe_start(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_observe(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_observe_end(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_select_actions_start(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_select_actions(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_select_actions_end(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_update_start(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_update(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_update_end(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    ######################
    # system trainer hooks
    ######################

    def on_training_init_start(self, trainer: SystemTrainer) -> None:
        """[summary]"""
        pass

    def on_training_init(self, trainer: SystemTrainer) -> None:
        """[summary]"""
        pass

    def on_training_init_end(self, trainer: SystemTrainer) -> None:
        """[summary]"""
        pass

    def on_training_utility_fns(self, trainer: SystemTrainer) -> None:
        """[summary]"""
        pass

    def on_training_loss_fns(self, trainer: SystemTrainer) -> None:
        """[summary]"""
        pass

    def on_training_step_fn(self, trainer: SystemTrainer) -> None:
        """[summary]"""
        pass

    def on_training_step_start(self, trainer: SystemTrainer) -> None:
        """[summary]"""
        pass

    def on_training_step(self, trainer: SystemTrainer) -> None:
        """[summary]"""
        pass

    def on_training_step_end(self, trainer: SystemTrainer) -> None:
        """[summary]"""
        pass

    ###############################
    # system parameter server hooks
    ###############################

    def on_parameter_server_init_start(self, server: SystemParameterServer) -> None:
        """[summary]"""
        pass

    def on_parameter_server_init(self, server: SystemParameterServer) -> None:
        """[summary]"""
        pass

    def on_parameter_server_init_checkpointer(
        self, server: SystemParameterServer
    ) -> None:
        """[summary]"""
        pass

    def on_parameter_server_init_end(self, server: SystemParameterServer) -> None:
        """[summary]"""
        pass

    def on_parameter_server_get_parameters_start(
        self, server: SystemParameterServer
    ) -> None:
        """[summary]"""
        pass

    def on_parameter_server_get_parameters(self, server: SystemParameterServer) -> None:
        """[summary]"""
        pass

    def on_parameter_server_get_parameters_end(
        self, server: SystemParameterServer
    ) -> None:
        """[summary]"""
        pass

    def on_parameter_server_set_parameters_start(
        self, server: SystemParameterServer
    ) -> None:
        """[summary]"""
        pass

    def on_parameter_server_set_parameters(self, server: SystemParameterServer) -> None:
        """[summary]"""
        pass

    def on_parameter_server_set_parameters_end(
        self, server: SystemParameterServer
    ) -> None:
        """[summary]"""
        pass

    def on_parameter_server_add_to_parameters_start(
        self, server: SystemParameterServer
    ) -> None:
        """[summary]"""
        pass

    def on_parameter_server_add_to_parameters(
        self, server: SystemParameterServer
    ) -> None:
        """[summary]"""
        pass

    def on_parameter_server_add_to_parameters_end(
        self, server: SystemParameterServer
    ) -> None:
        """[summary]"""
        pass

    def on_parameter_server_run_start(self, server: SystemParameterServer) -> None:
        """[summary]"""
        pass

    def on_parameter_server_run_loop_start(self, server: SystemParameterServer) -> None:
        """[summary]"""
        pass

    def on_parameter_server_run_loop_checkpoint(
        self, server: SystemParameterServer
    ) -> None:
        """[summary]"""
        pass

    def on_parameter_server_run_loop(self, server: SystemParameterServer) -> None:
        """[summary]"""
        pass

    def on_parameter_server_run_loop_termination(
        self, server: SystemParameterServer
    ) -> None:
        """[summary]"""
        pass

    def on_parameter_server_run_loop_end(self, server: SystemParameterServer) -> None:
        """[summary]"""
        pass
