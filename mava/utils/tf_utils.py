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

"""Utilities for nested data structures involving NumPy and TensorFlow 2.x."""

from typing import List, Optional

import sonnet as snt
import tensorflow as tf
import tree
from acme import types
from acme.tf.utils import add_batch_dim, squeeze_batch_dim, zeros_like

from mava.types import OLT


def ones_like(nest: types.Nest) -> types.NestedTensor:
    """Given a nest of array-like objects, returns similarly nested tf.zeros."""
    return tree.map_structure(lambda x: tf.ones(x.shape, x.dtype), nest)


def create_variables(
    network: snt.Module,
    input_spec: List[OLT],
) -> Optional[tf.TensorSpec]:
    """Builds the network with dummy inputs to create the necessary variables.
    Args:
      network: Sonnet Module whose variables are to be created.
      input_spec: list of input specs to the network. The length of this list
        should match the number of arguments expected by `network`.
    Returns:
      output_spec: only returns an output spec if the output is a tf.Tensor, else
          it doesn't return anything (None); e.g. if the output is a
          tfp.distributions.Distribution.
    """
    # Create a dummy observation with no batch dimension.
    dummy_input = [
        OLT(
            observation=zeros_like(in_spec.observation),
            legal_actions=ones_like(in_spec.legal_actions),
            terminal=zeros_like(in_spec.terminal),
        )
        for in_spec in input_spec
    ]

    # If we have an RNNCore the hidden state will be an additional input.
    if isinstance(network, snt.RNNCore):
        initial_state = squeeze_batch_dim(network.initial_state(1))
        dummy_input += [initial_state]

    # Forward pass of the network which will create variables as a side effect.
    dummy_output = network(*add_batch_dim(dummy_input))

    # Evaluate the input signature by converting the dummy input into a
    # TensorSpec. We then save the signature as a property of the network. This is
    # done so that we can later use it when creating snapshots. We do this here
    # because the snapshot code may not have access to the precise form of the
    # inputs.
    input_signature = tree.map_structure(
        lambda t: tf.TensorSpec((None,) + t.shape, t.dtype), dummy_input
    )
    network._input_signature = input_signature  # pylint: disable=protected-access

    def spec(output: tf.Tensor) -> tf.TensorSpec:
        # If the output is not a Tensor, return None as spec is ill-defined.
        if not isinstance(output, tf.Tensor):
            return None
        # If this is not a scalar Tensor, make sure to squeeze out the batch dim.
        if tf.rank(output) > 0:
            output = squeeze_batch_dim(output)
        return tf.TensorSpec(output.shape, output.dtype)

    return tree.map_structure(spec, dummy_output)
