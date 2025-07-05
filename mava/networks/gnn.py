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

# Shape conventions:
# T: number of timesteps
# E: number of environments
# N: number of agents
# V: number of nodes per graph
# F: feature dimension

from typing import Sequence

import chex
import jraph
import jraph._src.models as jraph_models
import jraph._src.utils as jraph_utils
from flax import linen as nn
from jax import numpy as jnp
from jraph import GraphsTuple as JraphGraphsTuple

from mava.networks.torsos import MLPTorso, _parse_activation_fn
from mava.types import GraphObservation
from mava.utils.graph.gnn_utils import GNN, batched_graph_to_single_graph


class InforMARLNbrhdAggregationTorso(GNN):
    """InforMARL Actor Network.
    For more details see: https://arxiv.org/abs/2211.02127

    Each agent has its own graph, where the agent is called the ego-agent. This torso uses
    multi-layer multi-head GAT layers to perform local neighborhood aggregation, where each
    node only aggregates information from its direct neighbors using edge information.

    For example, in a graph with nodes:
    A - B
    C - D
    where A is the ego-agent:
    - A's and B's node features will be a function of only A's and B's node features
    - C's and D's node features will be a function of only C's and D's node features

    Since A is the ego-agent, only A's node feature will be taken from this computation
    and concatenated with A's observation.
    """

    attention_query_layer_sizes: Sequence[int]
    use_layer_norm: bool
    activation: str

    num_heads: int
    num_attention_layers: int

    @nn.compact
    def __call__(self, graph_observation: GraphObservation) -> chex.Array:
        observation = graph_observation.observation
        graph = graph_observation.graph
        obs = observation.agents_view
        T, E, N, *_ = graph.nodes_strict.shape
        # one for timesteps, one for envs, one for agents
        graph = batched_graph_to_single_graph(graph, num_batch_dims=3)

        *_graph, ego_node_index = graph
        jraph_graph = JraphGraphsTuple(*_graph)

        # Apply the GAT layer to the graph.
        for i in range(self.num_attention_layers):
            should_avg_multi_head = i < self.num_attention_layers - 1
            jraph_graph = GraphMultiHeadAttentionLayer(
                attention_query_layer_sizes=self.attention_query_layer_sizes,
                use_layer_norm=self.use_layer_norm,
                activation=self.activation,
                num_heads=self.num_heads,
                avg_multi_head=should_avg_multi_head,
            )(jraph_graph)

        ego_node_features = get_ego_node_features(jraph_graph, ego_node_index, T, E, N)
        graph_embedding = jnp.concatenate([obs, ego_node_features], axis=-1)

        return graph_embedding


class InforMARLGlobalAggregationTorso(GNN):
    """InforMARL Actor Network.
    For more details see: https://arxiv.org/abs/2211.02127

    Each agent has its own graph, where the agent is called the ego-agent. This torso uses
    multi-layer multi-head GAT layers to aggregate node features, where edge information
    is used in the aggregation.

    For example, in a graph with nodes:
    A - B
    C - D
    where A is the ego-agent:
    - A's and B's node features will be a function of only A's and B's node features
    - C's and D's node features will be a function of only C's and D's node features

    Unlike the neighborhood aggregation torso, the ego-agent information is not picked after
    the aggregation. All nodes are averaged together and concatenated with A's observation.
    The GAT layers can be disabled by setting num_attention_layers to 0.
    """

    attention_query_layer_sizes: Sequence[int]
    use_layer_norm: bool
    activation: str

    num_heads: int
    num_attention_layers: int

    @nn.compact
    def __call__(self, graph_observation: GraphObservation) -> chex.Array:
        graph = graph_observation.graph
        T, E, N, V, *_ = graph.nodes_strict.shape
        # one for timesteps, one for envs, one for agents
        graph = batched_graph_to_single_graph(graph, num_batch_dims=3)

        *_graph, ego_node_index = graph
        jraph_graph = JraphGraphsTuple(*_graph)

        # Apply the GAT layer to the graph.
        for i in range(self.num_attention_layers):
            should_avg_multi_head = i < self.num_attention_layers - 1
            jraph_graph = GraphMultiHeadAttentionLayer(
                attention_query_layer_sizes=self.attention_query_layer_sizes,
                use_layer_norm=self.use_layer_norm,
                activation=self.activation,
                num_heads=self.num_heads,
                avg_multi_head=should_avg_multi_head,
            )(jraph_graph)

        node_embedding = jraph_graph.nodes

        node_features = node_embedding.reshape(
            T,
            E,
            N,
            V,
            *node_embedding.shape[1:],
        )
        # There is a graph for a given timestep, env, and agent.
        # We want to pool the node features across the timesteps for each env and agent.
        pooled_global_features = jnp.mean(node_features, axis=3)

        return pooled_global_features


class GraphMultiHeadAttentionLayer(nn.Module):
    """A multi-head attention layer for a graph.

    This layer implements Graph Attention Network (GAT) with multi-head attention.
    It uses the same MLP for both sender and receiver projections, and computes attention
    scores between nodes and their neighbors in parallel across multiple heads.
    The output can be either averaged or concatenated across heads based on avg_multi_head.
    """

    attention_query_layer_sizes: Sequence[int]
    use_layer_norm: bool
    activation: str

    num_heads: int
    avg_multi_head: bool

    def setup(self) -> None:
        self.activation_fn = _parse_activation_fn(self.activation)

    @nn.compact
    def __call__(self, graph: JraphGraphsTuple) -> JraphGraphsTuple:
        features_per_head = self.attention_query_layer_sizes[-1]
        total_output_features = features_per_head * self.num_heads

        # Define query MLP here, outside the helper function
        query_mlp = MLPTorso(
            layer_sizes=[*self.attention_query_layer_sizes[:-1], total_output_features],
            use_layer_norm=self.use_layer_norm,
            activation=self.activation,
            activate_final=True,
            name="attention_query_projection",
        )

        def attention_query_fn(nodes: jraph_models.NodeFeatures) -> jraph_models.NodeFeatures:
            """Embeds nodes and reshapes for multi-head attention using the outer query_mlp."""
            embedded_nodes = query_mlp(nodes)
            num_nodes = nodes.shape[0]
            return jnp.reshape(embedded_nodes, (num_nodes, self.num_heads, features_per_head))

        def compute_messages_fn(
            edges: jraph_models.EdgeFeatures,
            sent_attributes: jraph_models.SenderFeatures,
            received_attributes: jraph_models.ReceiverFeatures,
            graph_globals: jraph_models.Globals,
        ) -> jraph_models.EdgeFeatures:
            """Computes GAT messages (weighted, transformed sender features)."""
            sent_attributes = attention_query_fn(sent_attributes)
            received_attributes = attention_query_fn(received_attributes)

            if edges is not None:
                if edges.shape[-1] != features_per_head and edges.shape[-1] != 1:
                    raise ValueError(
                        f"Edge features dim ({edges.shape[-1]}) must match features_per_head "
                        f"({features_per_head}) for logit calculation."
                    )
                # Add edges, broadcasting across heads
                received_attributes = received_attributes + edges[:, None, :]

            logits = jnp.einsum(
                "ehf,ehf->eh", sent_attributes, received_attributes
            )  # dot product of sent and received features
            logits = logits / jnp.sqrt(features_per_head)

            sum_n_node = graph.nodes.shape[0]
            weights = jraph_utils.segment_softmax(
                logits, segment_ids=graph.receivers, num_segments=sum_n_node
            )

            messages = sent_attributes * weights[..., None]
            return messages

        def node_update_fn(
            nodes: jraph_models.NodeFeatures,
            aggregated_sent_attributes: jraph_models.SenderFeatures,
            aggregated_received_attributes: jraph_models.ReceiverFeatures,
            graph_globals: jraph_models.Globals,
        ) -> jraph_models.NodeFeatures:
            """Applies final activation and head aggregation.
            Ego node communication is unidirectional. So we only use the aggregated
            received attributes."""
            aggregated_received_attributes = self.activation_fn(aggregated_received_attributes)

            if self.avg_multi_head:
                return jnp.mean(aggregated_received_attributes, axis=1)
            else:
                num_nodes = aggregated_received_attributes.shape[0]
                return jnp.reshape(aggregated_received_attributes, (num_nodes, -1))

        multi_head_attn_layer = jraph.GraphNetwork(
            update_edge_fn=compute_messages_fn,
            update_node_fn=node_update_fn,
            attention_logit_fn=None,
            attention_reduce_fn=None,
            update_global_fn=None,
            aggregate_edges_for_nodes_fn=jraph_utils.segment_sum,
        )

        return multi_head_attn_layer(graph)


def get_ego_node_features(
    graph: JraphGraphsTuple, ego_node_index: chex.Array, *num_nodes: Sequence[int]
) -> chex.Array:
    """Returns the ego node features from a graph."""
    return graph.nodes[ego_node_index].reshape(*num_nodes, *graph.nodes.shape[1:])
