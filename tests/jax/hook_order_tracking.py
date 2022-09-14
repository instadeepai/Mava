from typing import List


class HookOrderTracking:
    """Defines hook implementations which append their names to a list.

    This is useful when testing that hooks are called in the correct order.
    All hooks to be overridden are found in mava.callbacks.base.Callback.

    When using this class, inherit from it first in the multiple inheritance.
    E.g. TestParameterServer(HashableHooks, ParameterServer).
    See tests/jax/core_system_components/parameter_server_test.py for an example.
    """

    def reset_hook_list(self) -> None:
        """Reset called hook list to be empty"""
        self.hook_list: List[str] = []

    ######################
    # system builder hooks
    ######################

    # BUILDER INITIAlISATION
    def on_building_init_start(self) -> None:
        """Start of builder initialisation."""
        self.hook_list.append("on_building_init_start")

    def on_building_init(self) -> None:
        """Builder initialisation."""
        self.hook_list.append("on_building_init")

    def on_building_init_end(self) -> None:
        """End of builder initialisation."""
        self.hook_list.append("on_building_init_end")

    # DATA SERVER
    def on_building_data_server_start(self) -> None:
        """Start of data server table building."""
        self.hook_list.append("on_building_data_server_start")

    def on_building_data_server_adder_signature(self) -> None:
        """Building of table adder signature."""
        self.hook_list.append("on_building_data_server_adder_signature")

    def on_building_data_server_rate_limiter(self) -> None:
        """Building of table rate limiter."""
        self.hook_list.append("on_building_data_server_rate_limiter")

    def on_building_data_server(self) -> None:
        """Building system data server tables."""
        self.hook_list.append("on_building_data_server")

    def on_building_data_server_end(self) -> None:
        """End of data server table building."""
        self.hook_list.append("on_building_data_server_end")

    # PARAMETER SERVER
    def on_building_parameter_server_start(self) -> None:
        """Start of building parameter server."""
        self.hook_list.append("on_building_parameter_server_start")

    def on_building_parameter_server(self) -> None:
        """Building system parameter server."""
        self.hook_list.append("on_building_parameter_server")

    def on_building_parameter_server_end(self) -> None:
        """End of building parameter server."""
        self.hook_list.append("on_building_parameter_server_end")

    # EXECUTOR
    def on_building_executor_start(self) -> None:
        """Start of building executor."""
        self.hook_list.append("on_building_executor_start")

    def on_building_executor_adder_priority(self) -> None:
        """Building adder priority function."""
        self.hook_list.append("on_building_executor_adder_priority")

    def on_building_executor_adder(self) -> None:
        """Building executor adder."""
        self.hook_list.append("on_building_executor_adder")

    def on_building_executor_logger(self) -> None:
        """Building executor logger."""
        self.hook_list.append("on_building_executor_logger")

    def on_building_executor_parameter_client(self) -> None:
        """Building executor parameter server client."""
        self.hook_list.append("on_building_executor_parameter_client")

    def on_building_executor(self) -> None:
        """Building system executor."""
        self.hook_list.append("on_building_executor")

    def on_building_executor_environment(self) -> None:
        """Building executor environment copy."""
        self.hook_list.append("on_building_executor_environment")

    def on_building_executor_environment_loop(self) -> None:
        """Building executor system-environment loop."""
        self.hook_list.append("on_building_executor_environment_loop")

    def on_building_executor_end(self) -> None:
        """End of building executor."""
        self.hook_list.append("on_building_executor_end")

    # TRAINER
    def on_building_trainer_start(self) -> None:
        """Start of building trainer."""
        self.hook_list.append("on_building_trainer_start")

    def on_building_trainer_logger(self) -> None:
        """Building trainer logger."""
        self.hook_list.append("on_building_trainer_logger")

    def on_building_trainer_dataset(self) -> None:
        """Building trainer dataset."""
        self.hook_list.append("on_building_trainer_dataset")

    def on_building_trainer_parameter_client(self) -> None:
        """Building trainer parameter server client."""
        self.hook_list.append("on_building_trainer_parameter_client")

    def on_building_trainer(self) -> None:
        """Building trainer."""
        self.hook_list.append("on_building_trainer")

    def on_building_trainer_end(self) -> None:
        """End of building trainer."""
        self.hook_list.append("on_building_trainer_end")

    # BUILD
    def on_building_start(self) -> None:
        """Start of system graph program build."""
        self.hook_list.append("on_building_start")

    def on_building_program_nodes(self) -> None:
        """Building system graph program nodes."""
        self.hook_list.append("on_building_program_nodes")

    def on_building_end(self) -> None:
        """End of system graph program build."""
        self.hook_list.append("on_building_end")

    # LAUNCH
    def on_building_launch_start(self) -> None:
        """Start of system launch."""
        self.hook_list.append("on_building_launch_start")

    def on_building_launch(self) -> None:
        """System launch."""
        self.hook_list.append("on_building_launch")

    def on_building_launch_end(self) -> None:
        """End of system launch."""
        self.hook_list.append("on_building_launch_end")

    #######################
    # system executor hooks
    #######################

    def on_execution_init_start(self) -> None:
        """[summary]"""
        self.hook_list.append("on_execution_init_start")

    def on_execution_init(self) -> None:
        """[summary]"""
        self.hook_list.append("on_execution_init")

    def on_execution_init_end(self) -> None:
        """[summary]"""
        self.hook_list.append("on_execution_init_end")

    def on_execution_select_action_start(self) -> None:
        """[summary]"""
        self.hook_list.append("on_execution_select_action_start")

    def on_execution_select_action_preprocess(self) -> None:
        """[summary]"""
        self.hook_list.append("on_execution_select_action_preprocess")

    def on_execution_select_action_sample(self) -> None:
        """[summary]"""
        self.hook_list.append("on_execution_select_action_sample")

    def on_execution_select_action_end(self) -> None:
        """[summary]"""
        self.hook_list.append("on_execution_select_action_end")

    def on_execution_observe_first_start(self) -> None:
        """[summary]"""
        self.hook_list.append("on_execution_observe_first_start")

    def on_execution_observe_first(self) -> None:
        """[summary]"""
        self.hook_list.append("on_execution_observe_first")

    def on_execution_observe_first_end(self) -> None:
        """[summary]"""
        self.hook_list.append("on_execution_observe_first_end")

    def on_execution_observe_start(self) -> None:
        """[summary]"""
        self.hook_list.append("on_execution_observe_start")

    def on_execution_observe(self) -> None:
        """[summary]"""
        self.hook_list.append("on_execution_observe")

    def on_execution_observe_end(self) -> None:
        """[summary]"""
        self.hook_list.append("on_execution_observe_end")

    def on_execution_select_actions_start(self) -> None:
        """[summary]"""
        self.hook_list.append("on_execution_select_actions_start")

    def on_execution_select_actions(self) -> None:
        """[summary]"""
        self.hook_list.append("on_execution_select_actions")

    def on_execution_select_actions_end(self) -> None:
        """[summary]"""
        self.hook_list.append("on_execution_select_actions_end")

    def on_execution_update_start(self) -> None:
        """[summary]"""
        self.hook_list.append("on_execution_update_start")

    def on_execution_update(self) -> None:
        """[summary]"""
        self.hook_list.append("on_execution_update")

    def on_execution_update_end(self) -> None:
        """[summary]"""
        self.hook_list.append("on_execution_update_end")

    ######################
    # system trainer hooks
    ######################

    def on_training_init_start(self) -> None:
        """[summary]"""
        self.hook_list.append("on_training_init_start")

    def on_training_init(self) -> None:
        """[summary]"""
        self.hook_list.append("on_training_init")

    def on_training_init_end(self) -> None:
        """[summary]"""
        self.hook_list.append("on_training_init_end")

    def on_training_utility_fns(self) -> None:
        """[summary]"""
        self.hook_list.append("on_training_utility_fns")

    def on_training_loss_fns(self) -> None:
        """[summary]"""
        self.hook_list.append("on_training_loss_fns")

    def on_training_step_fn(self) -> None:
        """[summary]"""
        self.hook_list.append("on_training_step_fn")

    def on_training_step_start(self) -> None:
        """[summary]"""
        self.hook_list.append("on_training_step_start")

    def on_training_step(self) -> None:
        """[summary]"""
        self.hook_list.append("on_training_step")

    def on_training_step_end(self) -> None:
        """[summary]"""
        self.hook_list.append("on_training_step_end")

    ###############################
    # system parameter server hooks
    ###############################

    def on_parameter_server_init_start(self) -> None:
        """[summary]"""
        self.hook_list.append("on_parameter_server_init_start")

    def on_parameter_server_init(self) -> None:
        """[summary]"""
        self.hook_list.append("on_parameter_server_init")

    def on_parameter_server_init_checkpointer(self) -> None:
        """[summary]"""
        self.hook_list.append("on_parameter_server_init_checkpointer")

    def on_parameter_server_init_end(self) -> None:
        """[summary]"""
        self.hook_list.append("on_parameter_server_init_end")

    def on_parameter_server_get_parameters_start(self) -> None:
        """[summary]"""
        self.hook_list.append("on_parameter_server_get_parameters_start")

    def on_parameter_server_get_parameters(self) -> None:
        """[summary]"""
        self.hook_list.append("on_parameter_server_get_parameters")

    def on_parameter_server_get_parameters_end(self) -> None:
        """[summary]"""
        self.hook_list.append("on_parameter_server_get_parameters_end")

    def on_parameter_server_set_parameters_start(self) -> None:
        """[summary]"""
        self.hook_list.append("on_parameter_server_set_parameters_start")

    def on_parameter_server_set_parameters(self) -> None:
        """[summary]"""
        self.hook_list.append("on_parameter_server_set_parameters")

    def on_parameter_server_set_parameters_end(self) -> None:
        """[summary]"""
        self.hook_list.append("on_parameter_server_set_parameters_end")

    def on_parameter_server_add_to_parameters_start(self) -> None:
        """[summary]"""
        self.hook_list.append("on_parameter_server_add_to_parameters_start")

    def on_parameter_server_add_to_parameters(self) -> None:
        """[summary]"""
        self.hook_list.append("on_parameter_server_add_to_parameters")

    def on_parameter_server_add_to_parameters_end(self) -> None:
        """[summary]"""
        self.hook_list.append("on_parameter_server_add_to_parameters_end")

    def on_parameter_server_run_start(self) -> None:
        """[summary]"""
        self.hook_list.append("on_parameter_server_run_start")

    def on_parameter_server_run_loop_start(self) -> None:
        """[summary]"""
        self.hook_list.append("on_parameter_server_run_loop_start")

    def on_parameter_server_run_loop_checkpoint(self) -> None:
        """[summary]"""
        self.hook_list.append("on_parameter_server_run_loop_checkpoint")

    def on_parameter_server_run_loop(self) -> None:
        """[summary]"""
        self.hook_list.append("on_parameter_server_run_loop")

    def on_parameter_server_run_loop_termination(self) -> None:
        """[summary]"""
        self.hook_list.append("on_parameter_server_run_loop_termination")

    def on_parameter_server_run_loop_end(self) -> None:
        """[summary]"""
        self.hook_list.append("on_parameter_server_run_loop_end")
