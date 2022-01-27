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

from typing import List

from mava.core import SystemBuilder
from mava.systems import Executor as Execute
from mava.callbacks import Callback
from mava.utils.decorators import execution, evaluation
from mava.components.execution import ExecutorSetup


class Executor(Callback):
    def __init__(
        self, components: List[Callback], evaluation=False, evaluation_interval=None
    ):
        """[summary]

        Args:
            config (Dict[str, Any]): [description]
            components (List[Callback]): [description]
            evaluator (bool, optional): [description]. Defaults to False.
        """
        self.components = components
        self.evaluation = evaluation
        self.evaluation_interval = evaluation_interval

    @execution
    def on_building_executor_make_executor(self, builder: SystemBuilder) -> None:
        """[summary]"""
        # create networks
        networks = builder.system()

        # create adder
        adder = builder.adder(builder._replay_client)

        # executor components
        executor_setup = ExecutorSetup(
            policy_networks=networks,
            agent_specs=builder.agent_specs,
            agent_net_keys=builder.agent_net_keys,
            network_sampling_setup=builder.network_sampling_setup,
            net_keys_to_ids=builder.net_keys_to_ids,
            adder=adder,
            counts=builder.executor_counts,
            variable_client=builder.executor_variable_client,
        )
        self.components.append(executor_setup)

        # create executor
        builder.executor = Execute(self.components)

    @evaluation
    def on_building_evaluator_make_evaluator(self, builder: SystemBuilder) -> None:
        """[summary]"""

        # create networks
        networks = builder.system()

        # evaluator components
        evaluator_setup = ExecutorSetup(
            policy_networks=networks,
            agent_specs=builder.agent_specs,
            agent_net_keys=builder.agent_net_keys,
            network_sampling_setup=builder.network_sampling_setup,
            net_keys_to_ids=builder.net_keys_to_ids,
            counts=builder.executor_counts,
            variable_client=builder.executor_variable_client,
            evaluator=True,
            interval=self.evaluator_interval,
        )
        self.components.append(evaluator_setup)

        # create evaluator
        builder.evaluator = Execute(self.components)
