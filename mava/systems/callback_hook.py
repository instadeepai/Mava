from abc import ABC
from copy import deepcopy
from typing import Any, Dict, List, Optional, Type, Union


class SystemCallbackHookMixin(ABC):

    ######################
    # system builder hooks
    ######################

    def on_building_init_start(self) -> None:
        """Called when the builder initialisation begins."""
        for callback in self.callbacks:
            callback.on_building_init_start(self, self.builder)

    def on_building_init_end(self) -> None:
        """Called when the builder initialisation ends."""
        for callback in self.callbacks:
            callback.on_building_init_end(self, self.builder)

    def on_building_make_replay_table_start(self) -> None:
        """[description]"""
        for callback in self.callbacks:
            callback.on_building_make_replay_table_start(self, self.builder)

    def on_building_adder_signature(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_adder_signature(self, self.builder)

    def on_building_rate_limiter(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_rate_limiter(self, self.builder)

    def on_building_make_tables(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_make_tables(self, self.builder)

    def on_building_make_replay_table_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_make_replay_table_end(self, self.builder)

    def on_building_make_dataset_iterator_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_make_dataset_iterator_start(self, self.builder)

    def on_building_dataset(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_dataset(self, self.builder)

    def on_building_make_dataset_iterator_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_make_dataset_iterator_end(self, self.builder)

    def on_building_make_adder_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_make_adder_start(self, self.builder)

    def on_building_adder_priority(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_adder_priority(self, self.builder)

    def on_building_make_adder(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_make_adder(self, self.builder)

    def on_building_make_tables(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_make_tables(self, self.builder)

    def on_building_make_replay_table_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_make_replay_table_start(self, self.builder)

    def on_building_make_variable_server_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_make_variable_server_start(self, self.builder)

    def on_building_variable_server(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_variable_server(self, self.builder)

    def on_building_make_variable_server_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_make_variable_server_end(self, self.builder)

    def on_building_make_executor_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_make_executor_start(self, self.builder)

    def on_building_variable_client(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_variable_client(self, self.builder)

    def on_building_executor(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_executor(self, self.builder)

    def on_building_make_executor_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_make_executor_end(self, self.builder)

    def on_building_make_trainer_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_make_trainer_start(self, self.builder)

    def on_building_variable_client(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_variable_client(self, self.builder)

    def on_building_trainer(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_trainer(self, self.builder)

    def on_building_trainer_statistics(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_trainer_statistics(self, self.builder)

    def on_building_make_trainer_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_building_make_trainer_end(self, self.builder)