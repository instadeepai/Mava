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

from mava.core_jax import SystemBuilder


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
