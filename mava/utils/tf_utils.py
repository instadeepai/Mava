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

from typing import Dict, List, Optional

import sonnet as snt
import tensorflow as tf
import tree
from acme import types
from acme.tf.utils import add_batch_dim, squeeze_batch_dim, zeros_like

from mava.types import OLT


def ones_like(nest: types.Nest) -> types.NestedTensor:
    """Given a nest of array-like objects, returns similarly nested tf.zeros."""
    return tree.map_structure(lambda x: tf.ones(x.shape, x.dtype), nest)


def create_optimizer_variables(
    networks: snt.Module,
    policy_optimizers: Dict[str, snt.Optimizer],
    critic_optimizers: Dict[str, snt.Optimizer],
) -> None:
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
    for net_key in networks["observations"].keys():
        # Check if network has an optimizer
        if net_key in policy_optimizers:
            # Get trainable variables.
            policy_variables = (
                networks["observations"][net_key].trainable_variables
                + networks["policies"][net_key].trainable_variables
            )
            critic_variables = (
                # In this agent, the critic loss trains the observation network.
                networks["observations"][net_key].trainable_variables
                + networks["critics"][net_key].trainable_variables
            )

            # Initialise the optimizer variables.
            policy_optimizers[net_key]._initialize(policy_variables)
            critic_optimizers[net_key]._initialize(critic_variables)
    return
