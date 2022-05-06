from typing import Any

import jax
import jax.numpy as jnp


def add_batch_dim_tree(tree: Any) -> Any:
    return jax.tree_map(lambda leaf: jnp.expand_dims(leaf, 0), tree)


def remove_batch_dim_tree(tree: Any) -> Any:
    return jax.tree_map(lambda leaf: jnp.squeeze(leaf, 0), tree)


def index_stacked_tree(tree: Any, index: int) -> Any:
    return jax.tree_map(lambda leaf: leaf[index], tree)


def stack_trees(list_of_trees):
    return jax.tree_map(lambda *leaves: jnp.stack(leaves), *list_of_trees)
