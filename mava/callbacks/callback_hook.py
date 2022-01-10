from abc import ABC


class SystemCallbackHookMixin(ABC):

    ######################
    # system builder hooks
    ######################

    # initialisation
    def on_building_init_start(self) -> None:
        """Called when the builder initialisation begins."""
        for callback in self.callbacks:
            callback.on_building_init_start(self, self.builder)

    def on_building_init(self) -> None:
        """Called when the builder initialisation begins."""
        for callback in self.callbacks:
            callback.on_building_init(self, self.builder)

    def on_building_init_end(self) -> None:
        """Called when the builder initialisation ends."""
        for callback in self.callbacks:
            callback.on_building_init_end(self, self.builder)

    # tables
    def on_building_tables_start(self) -> None:
        """[description]"""
        for callback in self.callbacks:
            callback.on_building_tables_start(self, self.builder)

    def on_building_tables_adder_signature(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_tables_adder_signature(self, self.builder)

    def on_building_tables_rate_limiter(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_tables_rate_limiter(self, self.builder)

    def on_building_create_tables(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_tables_create_tables(self, self.builder)

    def on_building_tables_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_table_end(self, self.builder)

    # dataset
    def on_building_dataset_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_dataset_start(self, self.builder)

    def on_building_dataset_create_dataset(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_dataset_create_dataset(self, self.builder)

    def on_building_dataset_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_dataset_end(self, self.builder)

    # adder
    def on_building_adder_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_adder_start(self, self.builder)

    def on_building_adder_set_priority(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_adder_set_priority(self, self.builder)

    def on_building_adder_create_adder(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_adder_create_adder(self, self.builder)

    # system
    def on_building_system_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_system_start(self, self.builder)

    def on_building_system_networks(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_system_networks(self, self.builder)

    def on_building_system_architecture(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_system_architecture(self, self.builder)

    def on_building_system_create_system(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_system_create_system(self, self.builder)

    def on_building_system_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_system_end(self, self.builder)

    # variable server
    def on_building_variable_server_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_make_variable_server_start(self, self.builder)

    def on_building_variable_server_create_variable_server(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_variable_server(self, self.builder)

    def on_building_make_variable_server_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_make_variable_server_end(self, self.builder)

    # executor
    def on_building_executor_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_executor_start(self, self.builder)

    def on_building_executor_logger(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_executor_logger(self, self.builder)

    def on_building_executor_variable_client(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_executor_variable_client(self, self.builder)

    def on_building_executor_create_executor(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_executor_create_executor(self, self.builder)

    def on_building_executor_environment(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_executor_environment(self, self.builder)

    def on_building_executor_train_loop(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_executor_train_loop(self, self.builder)

    def on_building_executor_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_executor_end(self, self.builder)

    # evaluator
    def on_building_evaluator_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_evaluator_start(self, self.builder)

    def on_building_evaluator_logger(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_evaluator_logger(self, self.builder)

    def on_building_evaluator_variable_client(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_evaluator_variable_client(self, self.builder)

    def on_building_evaluator_create_evaluator(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_evaluator_create_evaluator(self, self.builder)

    def on_building_evaluator_environment(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_evaluator_environment(self, self.builder)

    def on_building_evaluator_train_loop(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_evaluator_train_loop(self, self.builder)

    def on_building_evaluator_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_evaluator_end(self, self.builder)

    # trainer
    def on_building_trainer_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_trainer_start(self, self.builder)

    def on_building_trainer_logger(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_trainer_logger(self, self.builder)

    def on_building_trainer_variable_client(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_trainer_variable_client(self, self.builder)

    def on_building_trainer_create_trainer(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_trainer_create_trainer(self, self.builder)

    def on_building_make_trainer_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_make_trainer_end(self, self.builder)

    ########################
    # system execution hooks
    ########################

    def on_execution_init_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_init_start(self, self.executor)

    def on_execution_init(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_init(self, self.executor)

    def on_execution_init_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_init_end(self, self.executor)

    def on_execution_policy_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_policy_start(self, self.executor)

    def on_execution_policy_preprocess(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_policy_preprocess(self, self.executor)

    def on_execution_policy_compute(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_policy_compute(self, self.executor)

    def on_execution_policy_sample_action(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_policy_sample_action(self, self.executor)

    def on_execution_policy_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_policy_end(self, self.executor)

    def on_execution_select_action_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_select_action_start(self, self.executor)

    def on_execution_select_action(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_select_action(self, self.executor)

    def on_execution_select_action_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_select_action_end(self, self.executor)

    def on_execution_observe_first_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_observe_first_start(self, self.executor)

    def on_execution_observe_first(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_observe_first(self, self.executor)

    def on_execution_observe_first_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_observe_first_end(self, self.executor)

    def on_execution_observe_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_observe_start(self, self.executor)

    def on_execution_observe(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_observe(self, self.executor)

    def on_execution_observe_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_observe_end(self, self.executor)

    def on_execution_select_actions_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_select_actions_start(self, self.executor)

    def on_execution_select_actions(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_select_actions(self, self.executor)

    def on_execution_select_actions_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_select_actions_end(self, self.executor)

    def on_execution_update_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_update_start(self, self.executor)

    def on_execution_update(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_update(self, self.executor)

    def on_execution_update_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_update_end(self, self.executor)

    ######################
    # system trainer hooks
    ######################

    def on_training_init_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_init_start(self, self.trainer)

    def on_training_init_observation_networks(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_init_observation_networks(self, self.trainer)

    def on_training_init_target_observation_networks(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_init_target_observation_networks(self, self.trainer)

    def on_training_init_policy_networks(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_init_policy_networks(self, self.trainer)

    def on_training_init_target_policy_networks(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_init_target_policy_networks(self, self.trainer)

    def on_training_init_critic_networks(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_init_critic_networks(self, self.trainer)

    def on_training_init_target_critic_networks(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_init_target_critic_networks(self, self.trainer)

    def on_training_init_parameters(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_init_parameters(self, self.trainer)

    def on_training_init(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_init(self, self.trainer)

    def on_training_init_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_init_parameters(self, self.trainer)

    # updating target networks
    def on_training_update_target_networks_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_update_target_networks_start(self, self.trainer)

    def on_training_update_target_observation_networks(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_update_target_observation_networks(self, self.trainer)

    def on_training_update_target_policy_networks(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_update_target_policy_networks(self, self.trainer)

    def on_training_update_target_critic_networks(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_update_target_critic_networks(self, self.trainer)

    def on_training_update_target_networks_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_update_target_networks_end(self, self.trainer)

    # transform observations
    def on_training_transform_observations_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_transform_observations_start(self, self.trainer)

    def on_training_transform_observations(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_transform_observations(self, self.trainer)

    def on_training_transform_target_observations(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_transform_target_observations(self, self.trainer)

    def on_training_transform_observations_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_transform_observations_end(self, self.trainer)

    # get feed
    def on_training_get_feed_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_get_feed_start(self, self.trainer)

    def on_training_get_feed(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_get_feed(self, self.trainer)

    def on_training_get_feed_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_get_feed_end(self, self.trainer)
