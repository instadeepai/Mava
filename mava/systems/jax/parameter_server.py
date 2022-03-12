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

"""Jax systems parameter server."""


from typing import Dict, List, Sequence, Union

import jax.numpy as jnp

from mava.callbacks import Callback, CallbackHookMixin
from mava.core_jax import SystemParameterServer
from mava.utils.training_utils import non_blocking_sleep


class ParameterServer(SystemParameterServer, CallbackHookMixin):
    def __init__(
        self,
        components: List[Callback],
    ) -> None:
        """Initialise the parameter server."""
        super().__init__()

        self.callbacks = components

        self.on_parameter_server_init_start()

        self.on_parameter_server_init()

        self.on_parameter_server_init_checkpointer()

        self.on_parameter_server_init_end()

    def get_parameters(
        self, names: Union[str, Sequence[str]]
    ) -> Dict[str, Dict[str, jnp.ndarray]]:
        """Get parameters from the parameter server.

        Args:
            names : Names of the parameters to get
        Returns:
            The parameters that were requested
        """
        self._names = names

        self.on_parameter_server_get_parameters_start()

        self.on_parameter_server_get_parameters()

        self.on_parameter_server_get_parameters_end()

        return self.attr.parameters

    def set_parameters(
        self, names: Sequence[str], vars: Dict[str, jnp.ndarray]
    ) -> None:
        """Set parameters in the parameter server.

        Args:
            names : Names of the parameters to set
            vars : The values to set the parameters to
        """

        self.on_parameter_server_set_parameters_start()

        self.on_parameter_server_set_parameters()

        self.on_parameter_server_set_parameters_end()

    def add_to_parameters(
        self, names: Sequence[str], vars: Dict[str, jnp.ndarray]
    ) -> None:
        """Add to the parameters in the parameter server.

        Args:
            names : Names of the parameters to add to
            vars : The values to add to the parameters to
        """

        self.on_parameter_server_add_to_parameters_start()

        self.on_parameter_server_add_to_parameters()

        self.on_parameter_server_add_to_parameters_end()

    def run(self) -> None:
        """Run the parameter server. This function allows for checkpointing and other \
            centralised computations to be performed by the parameter server."""

        self.on_parameter_server_run_start()

        while True:
            # Wait 10 seconds before checking again
            non_blocking_sleep(10)

            self.on_parameter_server_run_loop_start()

            self.on_parameter_server_run_loop_checkpoint()

            self.on_parameter_server_run_loop()

            self.on_parameter_server_run_loop_termination()

            self.on_parameter_server_run_loop_end()
