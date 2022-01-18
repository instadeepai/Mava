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

"""
Abstract base class used to build new callbacks.
"""

import abc

from mava.core import SystemBuilder, SystemExecutor, SystemTrainer


class Callback(abc.ABC):
    """
    Abstract base class used to build new callbacks.
    Subclass this class and override any of the relevant hooks
    """

    ######################
    # system builder hooks
    ######################

    # initialisation
    def on_building_init_start(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_init(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_init_end(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    # tables
    def on_building_tables_start(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_tables_adder_signature(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_tables_rate_limiter(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_tables_make_tables(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_tables_end(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    # dataset
    def on_building_dataset_start(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_dataset_make_dataset(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_dataset_end(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    # adder
    def on_building_adder_start(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_adder_set_priority(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_adder_make_adder(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_adder_end(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    # system
    def on_building_system_start(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_system_networks(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_system_architecture(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_system_make_system(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_system_end(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    # variable server
    def on_building_variable_server_start(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_variable_server_make_variable_server(
        self, builder: SystemBuilder
    ) -> None:
        """[summary]"""
        pass

    def on_building_variable_server_end(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    # executor
    def on_building_executor_start(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_executor_logger(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_executor_variable_client(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_executor_make_executor(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_executor_environment(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_executor_train_loop(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_executor_end(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    # evaluator
    def on_building_evaluator_start(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_evaluator_logger(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_evaluator_variable_client(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_evaluator_make_evaluator(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_evaluator_environment(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_evaluator_eval_loop(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_evaluator_end(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    # trainer
    def on_building_trainer_start(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_trainer_logger(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_trainer_dataset(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_trainer_variable_client(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_trainer_make_trainer(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_trainer_end(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    # distributor
    def on_building_program_nodes(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_launch_distributor(self, builder: SystemBuilder) -> None:
        """[summary]"""
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

    def on_execution_policy_start(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_policy_preprocess(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_policy_compute(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_policy_sample_action(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_policy_end(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_select_action_start(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_select_action(self, executor: SystemExecutor) -> None:
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

    def on_training_init_observation_networks(self, trainer: SystemTrainer) -> None:
        """[summary]"""
        pass

    def on_training_init_target_observation_networks(
        self, trainer: SystemTrainer
    ) -> None:
        """[summary]"""
        pass

    def on_training_init_policy_networks(self, trainer: SystemTrainer) -> None:
        """[summary]"""
        pass

    def on_training_init_target_policy_networks(self, trainer: SystemTrainer) -> None:
        """[summary]"""
        pass

    def on_training_init_critic_networks(self, trainer: SystemTrainer) -> None:
        """[summary]"""
        pass

    def on_training_init_target_critic_networks(self, trainer: SystemTrainer) -> None:
        """[summary]"""
        pass

    def on_training_init_parameters(self, trainer: SystemTrainer) -> None:
        """[summary]"""
        pass

    def on_training_init(self, trainer: SystemTrainer) -> None:
        """[summary]"""
        pass

    def on_training_init_end(self, trainer: SystemTrainer) -> None:
        """[summary]"""
        pass

    def on_training_update_target_networks_start(self, trainer: SystemTrainer) -> None:
        """[summary]"""
        pass

    def on_training_update_target_observation_networks(
        self, trainer: SystemTrainer
    ) -> None:
        """[summary]"""
        pass

    def on_training_update_target_policy_networks(self, trainer: SystemTrainer) -> None:
        """[summary]"""
        pass

    def on_training_update_target_critic_networks(self, trainer: SystemTrainer) -> None:
        """[summary]"""
        pass

    def on_training_update_target_networks_end(self, trainer: SystemTrainer) -> None:
        """[summary]"""
        pass

    def on_training_transform_observations_start(self, trainer: SystemTrainer) -> None:
        """[summary]"""
        pass

    def on_training_transform_observations(self, trainer: SystemTrainer) -> None:
        """[summary]"""
        pass

    def on_training_transform_target_observations(self, trainer: SystemTrainer) -> None:
        """[summary]"""
        pass

    def on_training_transform_observations_end(self, trainer: SystemTrainer) -> None:
        """[summary]"""
        pass

    def on_training_get_feed_start(self, trainer: SystemTrainer) -> None:
        """[summary]"""
        pass

    def on_training_get_feed(self, trainer: SystemTrainer) -> None:
        """[summary]"""
        pass

    def on_training_get_feed_end(self, trainer: SystemTrainer) -> None:
        """[summary]"""
        pass
