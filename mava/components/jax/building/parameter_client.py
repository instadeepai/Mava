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

"""Parameter client for system builders"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from mava.components.jax import Component
from mava.core_jax import SystemBuilder
from mava.systems.jax import ParameterClient


class BaseParameterClient(Component):
    def _set_up_count_parameters(
        self, params: Dict[str, Any]
    ) -> Tuple[List[str], Dict[str, Any]]:
        add_params = {
            "trainer_steps": np.array(0, dtype=np.int32),
            "trainer_walltime": np.array(0, dtype=np.float32),
            "evaluator_steps": np.array(0, dtype=np.int32),
            "evaluator_episodes": np.array(0, dtype=np.int32),
            "executor_episodes": np.array(0, dtype=np.int32),
            "executor_steps": np.array(0, dtype=np.int32),
        }
        params.update(add_params)
        return list(add_params.keys()), params


@dataclass
class ExecutorParameterClientConfig:
    executor_parameter_update_period: int = 1000


class ExecutorParameterClient(BaseParameterClient):
    def __init__(
        self,
        config: ExecutorParameterClientConfig = ExecutorParameterClientConfig(),
    ) -> None:
        """Parameter client

        Args:
            config : parameter client config
        """

        self.config = config

    def on_building_executor_parameter_client(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        # Create policy parameters
        params = {}
        get_keys = []
        net_type_key = "networks"
        for agent_net_key in builder.store.networks[net_type_key].keys():
            param_key = f"{net_type_key}-{agent_net_key}"
            params[param_key] = builder.store.networks[net_type_key][
                agent_net_key
            ].params
            get_keys.append(param_key)

        count_names, params = self._set_up_count_parameters(params=params)

        get_keys.extend(count_names)

        builder.store.executor_counts = {name: params[name] for name in count_names}

        set_keys = get_keys.copy()

        # Executors should only be able to update relevant params.
        if builder.store.is_evaluator is True:
            set_keys = [x for x in set_keys if x.startswith("evaluator")]
        else:
            set_keys = [x for x in set_keys if x.startswith("executor")]

        parameter_client = None
        if builder.store.parameter_server_client:
            # Create parameter client
            parameter_client = ParameterClient(
                client=builder.store.parameter_server_client,
                parameters=params,
                get_keys=get_keys,
                set_keys=set_keys,
                update_period=self.config.executor_parameter_update_period,
            )

            # Make sure not to use a random policy after checkpoint restoration by
            # assigning parameters before running the environment loop.
            parameter_client.get_and_wait()

        builder.store.executor_parameter_client = parameter_client

    @staticmethod
    def name() -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "executor_parameter_client"

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return ExecutorParameterClientConfig


@dataclass
class TrainerParameterClientConfig:
    pass


class TrainerParameterClient(BaseParameterClient):
    def __init__(
        self,
        config: TrainerParameterClientConfig = TrainerParameterClientConfig(),
    ) -> None:
        """Parameter client

        Args:
            config : parameter client config
        """

        self.config = config

    def on_building_trainer_parameter_client(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        # Create parameter client
        params = {}
        set_keys = []
        get_keys = []
        # TODO (dries): Only add the networks this trainer is working with.
        # Not all of them.
        trainer_networks = builder.store.trainer_networks[builder.store.trainer_id]
        for net_type_key in builder.store.networks.keys():
            for net_key in builder.store.networks[net_type_key].keys():
                params[f"{net_type_key}-{net_key}"] = builder.store.networks[
                    net_type_key
                ][net_key].params
                if net_key in set(trainer_networks):
                    set_keys.append(f"{net_type_key}-{net_key}")
                else:
                    get_keys.append(f"{net_type_key}-{net_key}")

        # Add the optimizers to the variable server.
        # TODO (dries): Adjust this if using policy and critic optimizers.
        # TODO (dries): Add this back if we want the optimizer_state to
        # be store in the variable source. However some code might
        # need to be moved around as the builder currently does not
        # have access to the opt_states yet.
        # params["optimizer_state"] = trainer.store.opt_states

        count_names, params = self._set_up_count_parameters(params=params)

        get_keys.extend(count_names)
        builder.store.trainer_counts = {name: params[name] for name in count_names}

        # Create parameter client
        parameter_client = None
        if builder.store.parameter_server_client:
            parameter_client = ParameterClient(
                client=builder.store.parameter_server_client,
                parameters=params,
                get_keys=get_keys,
                set_keys=set_keys,
            )

            # Get all the initial parameters
            parameter_client.get_all_and_wait()

        builder.store.trainer_parameter_client = parameter_client

    @staticmethod
    def name() -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "trainer_parameter_client"
