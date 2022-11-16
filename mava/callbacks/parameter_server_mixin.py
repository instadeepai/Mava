# python3
# Copyright 2022 InstaDeep Ltd. All rights reserved.
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


class ParameterServerHookMixin(ABC):

    callbacks: List

    ###############################
    # system parameter server hooks
    ###############################

    # INIT
    def on_parameter_server_init_start(self) -> None:
        """Start of parameter server initialisation."""
        for callback in self.callbacks:
            callback.on_parameter_server_init_start(self)

    def on_parameter_server_init(self) -> None:
        """Parameter server initialisation."""
        for callback in self.callbacks:
            callback.on_parameter_server_init(self)

    def on_parameter_server_init_checkpointer(self) -> None:
        """Create checkpointer during parameter server initialisation."""
        for callback in self.callbacks:
            callback.on_parameter_server_init_checkpointer(self)

    def on_parameter_server_init_end(self) -> None:
        """End of parameter server initialisation."""
        for callback in self.callbacks:
            callback.on_parameter_server_init_end(self)

    # GET PARAMETERS
    def on_parameter_server_get_parameters_start(self) -> None:
        """Start of getting parameters from parameter server."""
        for callback in self.callbacks:
            callback.on_parameter_server_get_parameters_start(self)

    def on_parameter_server_get_parameters(self) -> None:
        """Get parameters from parameter server."""
        for callback in self.callbacks:
            callback.on_parameter_server_get_parameters(self)

    def on_parameter_server_get_parameters_end(self) -> None:
        """End of getting parameters from parameter server."""
        for callback in self.callbacks:
            callback.on_parameter_server_get_parameters_end(self)

    # SET PARAMETERS
    def on_parameter_server_set_parameters_start(self) -> None:
        """Start of setting parameters in parameter server."""
        for callback in self.callbacks:
            callback.on_parameter_server_set_parameters_start(self)

    def on_parameter_server_set_parameters(self) -> None:
        """Set parameters in parameter server."""
        for callback in self.callbacks:
            callback.on_parameter_server_set_parameters(self)

    def on_parameter_server_set_parameters_end(self) -> None:
        """End of setting parameters in parameter server."""
        for callback in self.callbacks:
            callback.on_parameter_server_set_parameters_end(self)

    # ADD TO PARAMETERS
    def on_parameter_server_add_to_parameters_start(self) -> None:
        """Start of adding to parameters in parameter server."""
        for callback in self.callbacks:
            callback.on_parameter_server_add_to_parameters_start(self)

    def on_parameter_server_add_to_parameters(self) -> None:
        """Add to parameters in parameter server."""
        for callback in self.callbacks:
            callback.on_parameter_server_add_to_parameters(self)

    def on_parameter_server_add_to_parameters_end(self) -> None:
        """End of adding to parameters in parameter server."""
        for callback in self.callbacks:
            callback.on_parameter_server_add_to_parameters_end(self)

    # RUN
    def on_parameter_server_run_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_parameter_server_run_start(self)

    # STEP
    def on_parameter_server_run_loop_start(self) -> None:
        """Start of parameter server run loop."""
        for callback in self.callbacks:
            callback.on_parameter_server_run_loop_start(self)

    def on_parameter_server_run_loop_checkpoint(self) -> None:
        """Checkpoint during parameter server run loop."""
        for callback in self.callbacks:
            callback.on_parameter_server_run_loop_checkpoint(self)

    def on_parameter_server_run_loop(self) -> None:
        """Parameter server run loop."""
        for callback in self.callbacks:
            callback.on_parameter_server_run_loop(self)

    def on_parameter_server_run_loop_termination(self) -> None:
        """Check for termination during parameter server run loop."""
        for callback in self.callbacks:
            callback.on_parameter_server_run_loop_termination(self)

    def on_parameter_server_run_loop_end(self) -> None:
        """End of parameter server run loop."""
        for callback in self.callbacks:
            callback.on_parameter_server_run_loop_end(self)
