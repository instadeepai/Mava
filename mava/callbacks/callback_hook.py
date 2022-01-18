from abc import ABC


class CallbackHookMixin(ABC):

    ######################
    # system builder hooks
    ######################

    # initialisation
    def on_building_init_start(self) -> None:
        """Called when the builder initialisation begins."""
        for callback in self.callbacks:
            callback.on_building_init_start(self)

    def on_building_init(self) -> None:
        """Called when the builder initialisation begins."""
        for callback in self.callbacks:
            callback.on_building_init(self)

    def on_building_init_end(self) -> None:
        """Called when the builder initialisation ends."""
        for callback in self.callbacks:
            callback.on_building_init_end(self)

    # tables
    def on_building_tables_start(self) -> None:
        """[description]"""
        for callback in self.callbacks:
            callback.on_building_tables_start(self)

    def on_building_tables_adder_signature(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_tables_adder_signature(self)

    def on_building_tables_rate_limiter(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_tables_rate_limiter(self)

    def on_building_tables_make_tables(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_tables_make_tables(self)

    def on_building_tables_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_tables_end(self)

    # dataset
    def on_building_dataset_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_dataset_start(self)

    def on_building_dataset_make_dataset(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_dataset_make_dataset(self)

    def on_building_dataset_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_dataset_end(self)

    # adder
    def on_building_adder_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_adder_start(self)

    def on_building_adder_set_priority(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_adder_set_priority(self)

    def on_building_adder_make_adder(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_adder_make_adder(self)

    def on_building_adder_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_adder_end(self)

    # system
    def on_building_system_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_system_start(self)

    def on_building_system_networks(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_system_networks(self)

    def on_building_system_architecture(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_system_architecture(self)

    def on_building_system_make_system(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_system_make_system(self)

    def on_building_system_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_system_end(self)

    # variable server
    def on_building_variable_server_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_variable_server_start(self)

    def on_building_variable_server_make_variable_server(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_variable_server_make_variable_server(self)

    def on_building_variable_server_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_variable_server_end(self)

    # executor
    def on_building_executor_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_executor_start(self)

    def on_building_executor_logger(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_executor_logger(self)

    def on_building_executor_variable_client(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_executor_variable_client(self)

    def on_building_executor_make_executor(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_executor_make_executor(self)

    def on_building_executor_environment(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_executor_environment(self)

    def on_building_executor_train_loop(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_executor_train_loop(self)

    def on_building_executor_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_executor_end(self)

    # evaluator
    def on_building_evaluator_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_evaluator_start(self)

    def on_building_evaluator_logger(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_evaluator_logger(self)

    def on_building_evaluator_variable_client(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_evaluator_variable_client(self)

    def on_building_evaluator_make_evaluator(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_evaluator_make_evaluator(self)

    def on_building_evaluator_environment(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_evaluator_environment(self)

    def on_building_evaluator_eval_loop(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_evaluator_eval_loop(self)

    def on_building_evaluator_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_evaluator_end(self)

    # trainer
    def on_building_trainer_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_trainer_start(self)

    def on_building_trainer_logger(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_trainer_logger(self)

    def on_building_trainer_dataset(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_trainer_dataset(self)

    def on_building_trainer_variable_client(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_trainer_variable_client(self)

    def on_building_trainer_make_trainer(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_trainer_make_trainer(self)

    def on_building_trainer_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_trainer_end(self)

    # distributor
    def on_building_program_nodes(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_program_nodes(self)

    def on_building_launch_distributor(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_launch_distributor(self)

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
