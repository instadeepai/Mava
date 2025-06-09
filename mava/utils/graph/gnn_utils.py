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

from typing import List, TypeGuard, Union

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing_extensions import TypeIs

from mava.types import (
    GraphObservation,
    GraphsTuple,
    MavaObservationType,
    Observation,
    ObservationGlobalState,
)


class GNN(nn.Module):
    """A parent class for all GNN models.
    This is used so that we can identify GNN models in base actor and critic networks.
    """

    pass


def is_graph_torso(torso: nn.Module) -> TypeGuard[GNN]:
    """Type guard to check if torso is a graph-based network."""
    return isinstance(torso, GNN)


def is_graph_observation(
    obs: Union[Observation, ObservationGlobalState, GraphObservation[MavaObservationType]],
) -> TypeIs[GraphObservation[MavaObservationType]]:
    """Type guard to check if observation is a GraphObservation."""
    return isinstance(obs, GraphObservation)


def validate_graph_components(
    torso: nn.Module, observation: Union[Observation, ObservationGlobalState, GraphObservation]
) -> None:
    """Validate that GNN and GraphObservation are used together."""
    is_graph = is_graph_observation(observation)
    is_gnn = is_graph_torso(torso)

    if is_graph != is_gnn:
        raise ValueError("GraphObservation and GNN must be used together. ")


def validate_num_dims_in_graph_tuple(graph: GraphsTuple, num_batch_dims: int) -> None:
    """Validate that the non feature attributes of a batched graph tuple are valid.

    Ensures that all graph attributes (receivers, senders, ego_node_index, n_node, n_edge)
    have the same number of batch dimensions as specified by num_batch_dims.
    """

    # Adds 1 dimension for number of edges
    num_receivers_dims = num_batch_dims + 1
    num_senders_dims = num_batch_dims + 1

    # Adds 1 dimension since default jraph GraphsTuple have 1 dimension for
    # these attributes when not batched.
    num_ego_node_index_dims = num_batch_dims + 1
    num_n_node_dims = num_batch_dims + 1
    num_n_edge_dims = num_batch_dims + 1

    assert graph.receivers is None or graph.receivers.ndim == num_receivers_dims, (
        "The number of batch dimensions in the receivers must match the number of batch dimensions"
        " in the receiver features"
    )
    assert graph.senders is None or graph.senders.ndim == num_senders_dims, (
        "The number of batch dimensions in the senders must match the number of batch dimensions"
        " in the sender features"
    )
    assert graph.ego_node_index.ndim == num_ego_node_index_dims, (
        "The number of batch dimensions in the ego node index must match the number of"
        " batch dimensions"
    )
    assert graph.n_node.ndim == num_n_node_dims, (
        "The number of batch dimensions in the number of nodes must match the number of"
        " batch dimensions"
    )
    assert graph.n_edge.ndim == num_n_edge_dims, (
        "The number of batch dimensions in the number of edges must match the number of"
        " batch dimensions"
    )


def batched_graph_to_single_graph(graph: GraphsTuple, num_batch_dims: int = 1) -> GraphsTuple:
    """Convert a batched graph to a single graph.

    Jraph's GraphsTuple convention doesn't include batch dimensions in its attributes.
    However, during data collection, multiple graphs are often stacked together,
    creating multiple batch dimensions. This function converts a batched GraphsTuple with
    multiple batch dimensions into a GraphsTuple that strips the batch dimensions in a
    way that's compatible with jraph functions.

    The function handles two cases:
    1. num_batch_dims=1: Simply removes the batch dimension from the leading axis
       of all attributes while preserving other dimensions.
    2. num_batch_dims>1: Removes the first num_batch_dims dimensions and adjusts
       node indices to combine the graphs into a single graph.

    Args:
        graph: A batched GraphsTuple with multiple batch dimensions
        num_batch_dims: Number of batch dimensions to remove (default: 1)

    Returns:
        A single GraphsTuple with batch dimensions removed and node indices adjusted
    """
    validate_num_dims_in_graph_tuple(graph, num_batch_dims)

    # concatenate the batch dimensions while retaining the feature dimensions
    graph = jax.tree.map(lambda x: x.reshape(-1, *x.shape[num_batch_dims:]), graph)

    # split the batch dimension into a list of graphs
    batched_graphs = jax.tree.map(
        lambda x: jnp.split(x, x.shape[0], axis=0),
        graph,
    )

    # remove the batch dimension from the feature dimensions
    batched_graphs = jax.tree.map(lambda y: jnp.squeeze(y, axis=0), batched_graphs)

    # convert graph of lists into a list of graphs
    list_of_graphs = jax.tree.transpose(
        outer_treedef=jax.tree.structure(graph),
        inner_treedef=None,  # Let JAX infer the inner (list) structure
        pytree_to_transpose=batched_graphs,
    )
    return batch(list_of_graphs)


def batch(graphs: List[GraphsTuple]) -> GraphsTuple:
    """Returns batched graph given a list of graphs.

    This is an adapted version of jraph.batch that adds support for ego_node_index
    in mava.types.GraphsTuple. The function:
    1. Calculates offsets for sender and receiver arrays based on node counts
    2. Concatenates all graph attributes (nodes, edges, globals)
    3. Adjusts sender, receiver, and ego_node indices by adding appropriate offsets
       to maintain correct node references in the batched graph

    Args:
        graphs: List of individual GraphsTuple to be batched together

    Returns:
        A single GraphsTuple containing all graphs with adjusted indices
    """
    # Calculates offsets for sender and receiver arrays, caused by concatenating
    # the nodes arrays.
    offsets = jnp.cumsum(jnp.array([0] + [jnp.sum(g.n_node) for g in graphs[:-1]]))

    def _map_concat(nests: List[chex.ArrayTree]) -> chex.ArrayTree:
        concat = lambda *args: jnp.concatenate(args)
        return jax.tree.map(concat, *nests)

    return GraphsTuple(
        n_node=jnp.concatenate([g.n_node for g in graphs]),
        n_edge=jnp.concatenate([g.n_edge for g in graphs]),
        nodes=_map_concat([g.nodes for g in graphs]),
        edges=_map_concat([g.edges for g in graphs]),
        globals=_map_concat([g.globals for g in graphs]),
        senders=jnp.concatenate([g.senders + o for g, o in zip(graphs, offsets, strict=False)]),
        receivers=jnp.concatenate([g.receivers + o for g, o in zip(graphs, offsets, strict=False)]),
        ego_node_index=jnp.concatenate(
            [g.ego_node_index + o for g, o in zip(graphs, offsets, strict=False)]
        ),
    )
