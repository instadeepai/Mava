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

"""Execution components for system builders"""

from typing import Dict, Type, Any

from acme.specs import EnvironmentSpec

from mava import core
from mava.callbacks import Callback
from mava.systems.building import SystemBuilder


class Executor(Callback):
    def __init__(
        self,
        config: Dict[str, Any],
    ):
        """[summary]

        Args:
            config (Dict[str, Any]): [description]
            executor_fn (Type[core.Executor]): [description]
        """
        self.config = config

    def on_building_executor_start(self, builder: SystemBuilder) -> None:
        """[summary]"""
        builder._system_networks = self.create_system()

    def on_building_executor_logger(self, builder: SystemBuilder) -> None:
        """[summary]"""
        # Create executor logger
        executor_logger_config = {}
        logger_config = self.config["system"]["logger_config"]
        if logger_config and "executor" in logger_config:
            executor_logger_config = logger_config["executor"]
        exec_logger = builder._logger_factory(  # type: ignore
            f"executor_{builder._executor_id}", **executor_logger_config
        )
        builder._exec_logger = exec_logger

    def on_building_executor(self, builder: SystemBuilder) -> None:
        """[summary]"""

        # Create the executor.
        executor = self._builder.make_executor(
            networks=networks,
            policy_networks=behaviour_policy_networks,
            adder=self._builder.make_adder(replay),
            variable_source=variable_source,
        )

        # Create policy variables
        variables = {}
        get_keys = []
        for net_type_key in ["observations", "policies"]:
            for net_key in networks[net_type_key].keys():
                var_key = f"{net_key}_{net_type_key}"
                variables[var_key] = networks[net_type_key][net_key].variables
                get_keys.append(var_key)
        variables = self.create_counter_variables(variables)

        count_names = [
            "trainer_steps",
            "trainer_walltime",
            "evaluator_steps",
            "evaluator_episodes",
            "executor_episodes",
            "executor_steps",
        ]
        get_keys.extend(count_names)
        counts = {name: variables[name] for name in count_names}

        variable_client = None
        if variable_source:
            # Get new policy variables
            variable_client = variable_utils.VariableClient(
                client=variable_source,
                variables=variables,
                get_keys=get_keys,
                update_period=self._config.executor_variable_update_period,
            )

            # Make sure not to use a random policy after checkpoint restoration by
            # assigning variables before running the environment loop.
            variable_client.get_and_wait()

        builder.executor = self._executor_fn(
            policy_networks=policy_networks,
            counts=counts,
            net_keys_to_ids=self._config.net_keys_to_ids,
            agent_specs=self._config.environment_spec.get_agent_specs(),
            agent_net_keys=self._config.agent_net_keys,
            network_sampling_setup=self._config.network_sampling_setup,
            variable_client=variable_client,
            adder=adder,
        )

    def on_building_executor_train_loop(self, builder: SystemBuilder) -> None:
        """[summary]"""
        # TODO (Arnu): figure out why factory function are giving type errors
        # Create the environment.
        environment = self._environment_factory(evaluation=False)  # type: ignore

        # Create the loop to connect environment and executor.
        train_loop = self._train_loop_fn(
            environment,
            executor,
            logger=exec_logger,
            **self._train_loop_fn_kwargs,
        )

        train_loop = DetailedPerAgentStatistics(train_loop)

    def on_building_executor_end(self, builder: SystemBuilder) -> None:
        """[summary]"""
