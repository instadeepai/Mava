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

# TODO: Rewrite this file to handle only JAX arrays.

from typing import Any, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import tree
from typing_extensions import TypeAlias

# Different types used for indexing arrays: int/slice or tuple of int/slice
Indexer: TypeAlias = Union[int, slice, Tuple[slice, ...], Tuple[int, ...]]


def tree_slice(pytree: chex.ArrayTree, i: Indexer) -> chex.ArrayTree:
    """Returns: a new pytree where for each leaf: leaf[i] is returned."""
    return tree.map(lambda x: x[i], pytree)


def tree_at_set(old_tree: chex.ArrayTree, i: Indexer, new_tree: chex.ArrayTree) -> chex.ArrayTree:
    """Update `old_tree` at position `i` with `new_tree`.
    Both trees must have equal dtypes and structures.
    """
    chex.assert_trees_all_equal_structs(old_tree, new_tree)
    chex.assert_trees_all_equal_dtypes(old_tree, new_tree)
    return tree.map(lambda old, new: old.at[i].set(new), old_tree, new_tree)


def ndim_at_least(x: chex.Array, num_dims: chex.Numeric) -> chex.Array:
    """Check if the number of dimensions of `x` is at least `num_dims`."""
    if not (isinstance(x, jax.Array) or isinstance(x, np.ndarray)):
        x = jnp.asarray(x)
    return x.ndim >= num_dims


def merge_leading_dims(x: chex.Array, num_dims: chex.Numeric) -> chex.Array:
    """Merge leading dimensions.

    Note:
    ----
        This implementation is a generic function for merging leading dimensions
        extracted from Haiku.
        For the original implementation, please refer to the following link:
        (https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/basic.py#L207)

    """
    # Don't merge if there aren't dimensions to merge.
    if not ndim_at_least(x, num_dims):
        return x

    new_shape = (np.prod(x.shape[:num_dims]),) + x.shape[num_dims:]
    return x.reshape(new_shape)


def concat_time_and_agents(x: chex.Array) -> chex.Array:
    """Concatenates the time and agent dimensions in the input tensor.

    Args:
    ----
        x: Input tensor of shape (Time, Batch, Agents, ...).

    Returns:
    -------
        chex.Array: Tensor of shape (Batch, Time x Agents, ...).
    """
    x = jnp.moveaxis(x, 0, 1)
    x = jnp.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], *x.shape[3:]))
    return x


def unreplicate_n_dims(x: Any, unreplicate_depth: int = 2) -> Any:
    """Unreplicates a pytree by removing the first `unreplicate_depth` axes.

    This function takes a pytree and removes some number of axes, associated with parameter
    duplication for running multiple updates across devices and in parallel with `vmap`.
    This is typically one axis for device replication, and one for the `update batch size`.
    """
    return tree.map(lambda x: x[(0,) * unreplicate_depth], x)  # type: ignore


def unreplicate_batch_dim(x: Any) -> Any:
    """Unreplicated just the update batch dimension.
    (The dimension that is vmapped over when acting and learning)

    In mava's case it is always the second dimension, after the device dimension.
    We simply take element 0 as the params are identical across this dimension.
    """
    return tree.map(lambda x: x[:, 0, ...], x)  # type: ignore


def switch_leading_axes(arr: chex.Array) -> chex.Array:
    """Switches the first two axes, generally used for BT -> TB."""
    return tree.map(lambda x: x.swapaxes(0, 1), arr)
