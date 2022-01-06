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

from typing import Dict, Any, List

from mava.core import SystemBuilder
from mava.systems.executing import Executor
from mava.systems.tf import variable_utils
from mava.callbacks import Callback
from mava.wrappers import DetailedPerAgentStatistics


class Executor(Callback):
    def __init__(self, config: Dict[str, Any], components: List[Callback]):
        """[summary]

        Args:
            executor_fn (Type[core.Executor]): [description]
        """
        self.config = config
        self.components = components

    def on_building_executor_logger(self, builder: SystemBuilder) -> None:
        """[summary]"""
        # Create executor logger
        executor_logger_config = {}
        if builder._logger_config and "executor" in builder._logger_config:
            executor_logger_config = builder._logger_config["executor"]
        exec_logger = builder._logger_factory(  # type: ignore
            f"executor_{builder._executor_id}", **executor_logger_config
        )
        builder.executor_logger = exec_logger

    def on_building_executor(self, builder: SystemBuilder) -> None:
        """[summary]"""

        # create networks
        networks = builder.create_system()

        # create adder
        adder = builder.adder(builder._replay_client)

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
        if builder._variable_source:
            # Get new policy variables
            variable_client = variable_utils.VariableClient(
                client=builder._variable_source,
                variables=variables,
                get_keys=get_keys,
                update_period=builder._config.executor_variable_update_period,
            )

            # Make sure not to use a random policy after checkpoint restoration by
            # assigning variables before running the environment loop.
            variable_client.get_and_wait()

        self.config.update(
            {
                "networks": networks,
                "adder": adder,
                "variable_client": variable_client,
                "counts": counts,
                "logger": builder.executor_logger,
            }
        )

        builder.executor = Executor(self.config, self.components)

    def on_building_executor_train_loop(self, builder: SystemBuilder) -> None:
        """[summary]"""
        environment = builder._environment_factory(evaluation=False)  # type: ignore

        # Create the loop to connect environment and executor.
        builder.train_loop = builder._train_loop_fn(
            environment,
            builder._executor,
            logger=builder._exec_logger,
            **builder._train_loop_fn_kwargs,
        )

    def on_building_evaluator_end(self, builder: SystemBuilder) -> None:
        """[summary]"""

        builder.train_loop = DetailedPerAgentStatistics(builder.train_loop)


class Evaluator(Executor):
    def on_building_evaluator_logger(self, builder: SystemBuilder) -> None:
        """[summary]"""
        # Create eval logger.
        evaluator_logger_config = {}
        if builder._logger_config and "evaluator" in builder._logger_config:
            evaluator_logger_config = builder._logger_config["evaluator"]
        eval_logger = builder._logger_factory(  # type: ignore
            "evaluator", **evaluator_logger_config
        )
        builder._eval_logger = eval_logger

    def on_building_evaluator(self, builder: SystemBuilder) -> None:
        """[summary]"""

        # create networks
        networks = builder.create_system()

        # create adder
        adder = builder.adder(builder._replay_client)

        # Create policy variables
        variables = {}
        get_keys = []
        for net_type_key in ["observations", "policies"]:
            for net_key in builder._system_networks[net_type_key].keys():
                var_key = f"{net_key}_{net_type_key}"
                variables[var_key] = builder._system_networks[net_type_key][
                    net_key
                ].variables
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
        if builder._variable_source:
            # Get new policy variables
            variable_client = variable_utils.VariableClient(
                client=builder._variable_source,
                variables=variables,
                get_keys=get_keys,
                update_period=builder._config.executor_variable_update_period,
            )

            # Make sure not to use a random policy after checkpoint restoration by
            # assigning variables before running the environment loop.
            variable_client.get_and_wait()

        self.config.update(
            {
                "networks": networks,
                "adder": adder,
                "variable_client": variable_client,
                "counts": counts,
                "logger": builder.executor_logger,
            }
        )

        builder.evaluator = builder.executor(
            policy_networks=builder._system_networks,
            counts=counts,
            net_keys_to_ids=builder._config.net_keys_to_ids,
            agent_specs=builder._config.environment_spec.get_agent_specs(),
            agent_net_keys=builder._config.agent_net_keys,
            network_sampling_setup=builder._config.network_sampling_setup,
            variable_client=variable_client,
            adder=adder,
        )

    def on_building_evaluator_eval_loop(self, builder: SystemBuilder) -> None:
        """[summary]"""
        environment = builder._environment_factory(evaluation=False)  # type: ignore

        # Create the loop to connect environment and executor.
        builder.eval_loop = builder._eval_loop_fn(
            environment,
            builder._evaluator,
            logger=builder._eval_logger,
            **builder._eval_loop_fn_kwargs,
        )

    def on_building_evaluator_end(self, builder: SystemBuilder) -> None:
        """[summary]"""

        builder.eval_loop = DetailedPerAgentStatistics(builder.eval_loop)