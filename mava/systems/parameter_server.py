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

"""Jax systems parameter server."""


from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Union

from mava.callbacks import Callback, ParameterServerHookMixin
from mava.core_jax import SystemParameterServer
from mava.utils.training_utils import non_blocking_sleep


class ParameterServer(SystemParameterServer, ParameterServerHookMixin):
    def __init__(
        self,
        store: SimpleNamespace,
        components: List[Callback],
    ) -> None:
        """Initialise the parameter server.

        Args:
            store: builder store.
            components: components in the system.
        """
        super().__init__()

        self.store = store
        self.callbacks = components

        self.on_parameter_server_init_start()

        self.on_parameter_server_init()

        self.on_parameter_server_init_checkpointer()

        self.on_parameter_server_init_end()

    def get_parameters(self, names: Union[str, Sequence[str]]) -> Any:
        """Get parameters from the parameter server.

        Args:
            names: names of the parameters to get.

        Returns:
            The parameters that were requested.
        """
        self.store._param_names = names

        self.on_parameter_server_get_parameters_start()

        self.on_parameter_server_get_parameters()

        self.on_parameter_server_get_parameters_end()

        return self.store.get_parameters

    def set_parameters(self, set_params: Dict[str, Any]) -> None:
        """Set parameters in the parameter server.

        Args:
            set_params: dictionary {parameter name: new value}.

        Returns:
            None.
        """
        self.store._set_params = set_params

        self.on_parameter_server_set_parameters_start()

        self.on_parameter_server_set_parameters()

        self.on_parameter_server_set_parameters_end()

    def add_to_parameters(self, add_to_params: Dict[str, Any]) -> None:
        """Add to the parameters in the parameter server.

        Args:
            add_to_params: dictionary {parameter name: value to add}.

        Returns:
            None.
        """
        self.store._add_to_params = add_to_params

        self.on_parameter_server_add_to_parameters_start()

        self.on_parameter_server_add_to_parameters()

        self.on_parameter_server_add_to_parameters_end()

    def step(self) -> None:
        """Single step of the parameter server.

        Calls to hooks which define the main operations of the parameter server.
        Also calls to termination condition and checkpointing hooks.

        Returns:
            None.
        """
        # Wait {non_blocking_sleep_seconds} seconds before checking again
        non_blocking_sleep(self.store.global_config.non_blocking_sleep_seconds)

        self.on_parameter_server_run_loop_start()

        self.on_parameter_server_run_loop_checkpoint()

        self.on_parameter_server_run_loop()

        self.on_parameter_server_run_loop_termination()

        self.on_parameter_server_run_loop_end()

    def run(self) -> None:
        """Run the parameter server, stepping in an infinite loop.

        This function allows for checkpointing and other
        centralised computations to be performed by the parameter server.

        Returns:
            None.
        """

        self.on_parameter_server_run_start()

        while True:
            self.step()
