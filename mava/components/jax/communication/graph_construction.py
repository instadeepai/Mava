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
import abc
from dataclasses import dataclass
from typing import Dict, List, Type

import numpy as np

from mava.callbacks import Callback
from mava.components.jax import Component
from mava.core_jax import SystemExecutor
from mava.types import OLT

"""Components to construct GraphsTuples from the environment observations."""


# TODO(Matthew): delete once using actual GraphsTuples
class TempGraphsTuple:
    def __init__(self) -> None:
        """Create temp graph."""
        self.node_features = np.zeros(1)
        self.structure = np.zeros(1)


@dataclass
class GdnGraphConstructorConfig:
    pass


class GdnGraphConstructor(Component):
    @abc.abstractmethod
    def __init__(self, config: GdnGraphConstructorConfig = GdnGraphConstructorConfig()):
        """Component defines hooks to override for building a GDN graph.

        Args:
            config: GdnGraphConstructorConfig.
        """
        self.config = config

    @abc.abstractmethod
    def on_execution_observe_first(self, executor: SystemExecutor) -> None:
        """Create GDN graph at start of episode.

        Method must create executor.store.communication_graphs_tuple.

        Args:
            executor: SystemExecutor.

        Returns:
            None.
        """
        pass

    def on_execution_observe(self, executor: SystemExecutor) -> None:
        """Create GDN graph during an episode.

        Method must create executor.store.communication_graphs_tuple.

        Args:
            executor: SystemExecutor.

        Returns:
            None.
        """
        pass

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "gdn_graph_constructor"

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        None required.

        Returns:
            List of required component classes.
        """
        return []


class GdnGraphFromEnvironment(GdnGraphConstructor):
    def __init__(self, config: GdnGraphConstructorConfig = GdnGraphConstructorConfig()):
        """Component builds a GDN graph from the environment.

        Args:
            config: GdnGraphConstructorConfig.
        """
        self.config = config

    def on_execution_observe_first(self, executor: SystemExecutor) -> None:
        """Create GDN graph from the environment at start of episode.

        Args:
            executor: SystemExecutor.

        Returns:
            None.
        """
        if "communication_graph" not in executor.store.extras:
            raise Exception("Environment does not return a communication graph.")

        executor.store.communication_graphs_tuple = self._build_graphs_tuple(
            executor.store.timestep.observation,
            executor.store.extras["communication_graph"],
        )

    def on_execution_observe(self, executor: SystemExecutor) -> None:
        """Create GDN graph from the environment during an episode.

        Args:
            executor: SystemExecutor.

        Returns:
            None.
        """
        if "communication_graph" not in executor.store.next_extras:
            raise Exception("Environment does not return a communication graph.")

        executor.store.communication_graphs_tuple = self._build_graphs_tuple(
            executor.store.next_timestep.observation,
            executor.store.next_extras["communication_graph"],
        )

    # TODO(Matthew): switch the following to assemble a jraph GraphsTuple
    @staticmethod
    def _build_graphs_tuple(
        observations: Dict[str, OLT], communication_graph: np.ndarray
    ) -> TempGraphsTuple:
        """Assemble a GraphsTuple from the observations and graph structure.

        Args:
            observations: Agent observations.
            communication_graph: Comms graph structure from the environment.

        Returns:
            GraphsTuple of 'communication_graph' with node features 'observations'.
        """
        temp = TempGraphsTuple()
        observation_list = [0] * len(observations)
        for agent, observation in observations.items():
            agent_num = int(agent[6:])
            observation_list[agent_num] = observation.observation
        temp.node_features = np.array(observation_list)
        temp.structure = communication_graph
        return temp
