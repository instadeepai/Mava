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

from typing import List

import chex
import jax
import jax.numpy as jnp

from mava.types import GraphsTuple


def batched_graph_to_single_graph(graph: GraphsTuple, num_batch_dims: int = 1) -> GraphsTuple:
    graph = jax.tree.map(lambda x: x.reshape(-1, *x.shape[num_batch_dims:]), graph)
    batched_graphs = jax.tree.map(
        lambda x: jax.tree.map(lambda y: jnp.squeeze(y, axis=0), jnp.split(x, x.shape[0], axis=0)),
        graph,
    )
    list_of_graphs = jax.tree.transpose(
        outer_treedef=jax.tree.structure(graph),
        inner_treedef=None,  # Let JAX infer the inner (list) structure
        pytree_to_transpose=batched_graphs,
    )
    return batch(list_of_graphs)


def batch(graphs: List[GraphsTuple]) -> GraphsTuple:
    """Returns batched graph given a list of graphs.

    This is a adapted version of jraph.batch with support for ego_node_index in
    the mava.types.GraphsTuple.
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
