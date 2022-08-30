from typing import Any, List

import jax
import jax.numpy as jnp


def add_batch_dim_tree(tree: Any) -> Any:
    """_description_"""
    return jax.tree_util.tree_map(lambda leaf: jnp.expand_dims(leaf, 0), tree)


def remove_batch_dim_tree(tree: Any) -> Any:
    """_description_"""
    return jax.tree_util.tree_map(lambda leaf: jnp.squeeze(leaf, 0), tree)


def index_stacked_tree(tree: Any, index: int) -> Any:
    """_description_"""
    return jax.tree_util.tree_map(lambda leaf: leaf[index], tree)


def stack_trees(trees: List) -> Any:
    """_description_"""
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *trees)
