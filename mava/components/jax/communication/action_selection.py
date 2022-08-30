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

"""Components for GDN communication during action selection."""
import abc
from dataclasses import dataclass
from typing import List, Type

from mava.callbacks import Callback
from mava.components.jax import Component
from mava.components.jax.communication.observing import (
    GdnGraphConstructor,
    TempGraphsTuple,
)
from mava.core_jax import SystemExecutor
from mava.types import OLT


@dataclass
class ExecutorGdnConfig:
    pass


class ExecutorGdn(Component):
    @abc.abstractmethod
    def __init__(
        self,
        config: ExecutorGdnConfig = ExecutorGdnConfig(),
    ):
        """Component modifies executor observations using a GNN."""
        self.config = config

    @abc.abstractmethod
    def on_execution_select_actions_start(self, executor: SystemExecutor) -> None:
        """Modify and store observations using a GNN.

        Args:
            executor: SystemExecutor.

        Returns:
            None.
        """

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "executor_gdn"

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        None required.

        Returns:
            List of required component classes.
        """
        return [GdnGraphConstructor]


class FeedforwardExecutorGdn(ExecutorGdn):
    def __init__(
        self,
        config: ExecutorGdnConfig = ExecutorGdnConfig(),
    ):
        """Component modifies executor observations using a feedforward GNN."""
        self.config = config

    def on_execution_select_actions_start(self, executor: SystemExecutor) -> None:
        """Modify and store observations using a feedforward GNN.

        Args:
            executor: SystemExecutor.

        Returns:
            None.
        """
        # TODO(Matthew): replace with actual GNN
        def gnn(input_graph: TempGraphsTuple) -> TempGraphsTuple:
            input_graph.node_features += 1
            return input_graph

        output_graphs_tuple = gnn(executor.store.communication_graphs_tuple)
        new_agent_obs = output_graphs_tuple.node_features

        for agent in executor.store.observations.keys():
            agent_num = int(agent[6:])
            executor.store.observations[agent] = OLT(
                observation=new_agent_obs[agent_num],
                legal_actions=executor.store.observations[agent].legal_actions,
                terminal=executor.store.observations[agent].terminal,
            )
