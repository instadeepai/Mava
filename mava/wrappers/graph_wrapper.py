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

# Note this is only here until this is merged into jumanji
# PR: https://github.com/instadeepai/jumanji/pull/223

from functools import cached_property
from typing import Tuple, Union

import chex
import jax
import jax.numpy as jnp
import jraph
from jumanji import specs
from jumanji.env import State
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper

from mava.types import GraphsTuple, MarlEnv, Observation, ObservationGlobalState


class GraphWrapper(Wrapper):
    """Wrapper to convert the observation structure from the environment into a Jraph GraphsTuple
    representing the graph topology of the agents. The default is a fully-connected graph.
    """

    # This init isn't really needed as jumanji.Wrapper will forward the attributes,
    # but mypy doesn't realize this.
    def __init__(self, env: MarlEnv, add_self_loops: bool = True):
        super().__init__(env)
        self._env: MarlEnv

        self.add_self_loops = add_self_loops

        self.num_agents = self._env.num_agents
        self.time_limit = self._env.time_limit
        self.action_dim = self._env.action_dim

    def add_graph_to_observations(
        self, state: State, observation: Union[Observation, ObservationGlobalState]
    ) -> Union[Observation, ObservationGlobalState]:
        """
        Default graph is a fully connected graph with no edge features. Every agent is a node and
        its observation is the node feature.
        """

        # Create a graph for each agent
        def _make_fully_connected_graph(ego_idx: int) -> GraphsTuple:
            base = jraph.get_fully_connected_graph(
                n_node_per_graph=self.num_agents,
                n_graph=1,
                node_features=observation.agents_view,
            )
            return GraphsTuple(
                **base._asdict(),
                ego_node_index=jnp.array([ego_idx]),
            )

        graph = jax.vmap(_make_fully_connected_graph)(jnp.arange(self.num_agents))

        observation = observation._replace(
            graph=graph,
        )
        return observation

    def reset(
        self, key: chex.PRNGKey
    ) -> Tuple[State, TimeStep[Union[Observation, ObservationGlobalState]]]:
        """Reset the environment and add the graph representation to the observation."""
        state, timestep = super().reset(key)
        timestep.observation = self.add_graph_to_observations(state, timestep.observation)
        return state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> Tuple[State, TimeStep[Union[Observation, ObservationGlobalState]]]:
        """Step the environment and add the graph representation to the observation."""
        state, timestep = self._env.step(state, action)
        timestep.observation = self.add_graph_to_observations(state, timestep.observation)
        return state, timestep

    @cached_property
    def observation_spec(
        self,
    ) -> Union[specs.Spec[Observation], specs.Spec[ObservationGlobalState]]:
        """Define the observation spec for the Jraph graph representation."""
        obs_spec = self._env.observation_spec

        max_n_edge = self.num_agents * self.num_agents

        graph_spec = specs.Spec(
            constructor=GraphsTuple,
            name="graph",
            nodes=specs.Array(
                shape=(
                    self.num_agents,
                    self.num_agents,
                    *obs_spec.agents_view.shape[1:],
                ),
                dtype=obs_spec.agents_view.dtype,
                name="nodes",
            ),
            edges=None,
            senders=specs.Array(
                shape=(self.num_agents, max_n_edge), dtype=jnp.int32, name="senders"
            ),
            receivers=specs.Array(
                shape=(self.num_agents, max_n_edge), dtype=jnp.int32, name="receivers"
            ),
            n_node=specs.Array(shape=(self.num_agents, 1), dtype=jnp.int32, name="n_node"),
            n_edge=specs.Array(shape=(self.num_agents, 1), dtype=jnp.int32, name="n_edge"),
            globals=None,
            ego_node_index=specs.Array(
                shape=(self.num_agents, 1), dtype=jnp.int32, name="ego_node_index"
            ),
        )

        return obs_spec.replace(graph=graph_spec)
