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

import time
import warnings
from typing import List, Type, Union

from acme.jax import savers as acme_savers
from chex import dataclass

from mava.callbacks import Callback
from mava.components import Component
from mava.components.normalisation.base_normalisation import BaseNormalisation
from mava.components.updating.parameter_server import ParameterServer
from mava.core_jax import SystemParameterServer
from mava.utils.checkpointing_utils import update_to_best_net
from mava.wrappers import SaveableWrapper

"""Checkpointer component for Mava systems."""


@dataclass
class CheckpointerConfig:
    checkpoint_minute_interval: float = 5
    restore_best_net: Union[str, None] = None


class Checkpointer(Component):
    def __init__(
        self,
        config: CheckpointerConfig = CheckpointerConfig(),
    ):
        """Component for checkpointing system variables."""
        self.config = config

    def on_parameter_server_init(self, server: SystemParameterServer) -> None:
        """Create the system checkpointer.

        Args:
            server: SystemParameterServer.

        Returns:
            None.
        """
        saveable_parameters = SaveableWrapper(server.store.parameters)
        old_trainer_steps = server.store.parameters["trainer_steps"].copy()
        server.store.system_checkpointer = acme_savers.Checkpointer(
            object_to_save=saveable_parameters,  # must be type saveable
            directory=server.store.experiment_path,
            add_uid=False,
            time_delta_minutes=0,
        )

        # Check if the checkpointer restored the network parameters
        # and if the user wants the network with the best performance.
        if (old_trainer_steps != server.store.parameters["trainer_steps"]) and (
            self.config.restore_best_net is not None
        ):
            if server.has(BaseNormalisation) and (
                server.store.global_config.normalise_observations
                or server.store.global_config.normalise_target_values
            ):
                warnings.warn(
                    """Best checkpointing does not save normalisation parameters,
                    thus checkpoint loading may not work"""
                )
            update_to_best_net(server, self.config.restore_best_net)

        server.store.last_checkpoint_time = time.time()
        server.store.checkpoint_minute_interval = self.config.checkpoint_minute_interval

    def on_parameter_server_run_loop_checkpoint(
        self, server: SystemParameterServer
    ) -> None:
        """Intermittently checkpoint the server parameters.

        Args:
            server: SystemParameterServer.

        Returns:
            None.
        """
        if (
            time.time() - server.store.last_checkpoint_time
            > self.config.checkpoint_minute_interval * 60 + 1
        ):
            server.store.system_checkpointer.save()
            server.store.last_checkpoint_time = time.time()

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "checkpointer"

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        ParameterServer required to set up server.store.parameters
        and server.store.experiment_path.

        Returns:
            List of required component classes.
        """
        return [ParameterServer]
