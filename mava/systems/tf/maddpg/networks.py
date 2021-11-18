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
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
import sonnet as snt
import tensorflow as tf
from acme import types
from acme.tf import utils as tf2_utils
from dm_env import specs

from mava import specs as mava_specs
from mava.components.tf import networks
from mava.components.tf.networks import DiscreteValuedHead
from mava.utils.enums import ArchitectureType

Array = specs.Array
BoundedArray = specs.BoundedArray
DiscreteArray = specs.DiscreteArray


def make_default_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    agent_net_keys: Dict[str, str],
    net_spec_keys: Dict[str, str] = {},
    policy_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = None,
    critic_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (512, 512, 256),
    sigma: float = 0.3,
    archecture_type: ArchitectureType = ArchitectureType.feedforward,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    num_atoms: Optional[int] = None,
    seed: Optional[int] = None,
) -> Mapping[str, types.TensorTransformation]:
    """Default networks for maddpg.

    Args:
        environment_spec: description of the action and
            observation spaces etc. for each agent in the system.
        agent_net_keys: specifies what network each agent uses.
        vmin: hyperparameters for the distributional critic in mad4pg.
        vmax: hyperparameters for the distributional critic in mad4pg.
        net_spec_keys: specifies the specs of each network.
        policy_networks_layer_sizes: size of policy networks.
        critic_networks_layer_sizes: size of critic networks.
        sigma: hyperparameters used to add Gaussian noise
            for simple exploration. Defaults to 0.3.
        archecture_type: archecture used
            for agent networks. Can be feedforward or recurrent.
            Defaults to ArchitectureType.feedforward.

        num_atoms:  hyperparameters for the distributional critic in
            mad4pg.
        seed: random seed for network initialization.

    Returns:
        returned agent networks.
    """
    # Set Policy function and layer size
    # Default size per arch type.
    if archecture_type == ArchitectureType.feedforward:
        if not policy_networks_layer_sizes:
            policy_networks_layer_sizes = (
                256,
                256,
                256,
            )
        policy_network_func = snt.Sequential
    elif archecture_type == ArchitectureType.recurrent:
        if not policy_networks_layer_sizes:
            policy_networks_layer_sizes = (128, 128)
        policy_network_func = snt.DeepRNN

    assert policy_networks_layer_sizes is not None
    assert policy_network_func is not None

    specs = environment_spec.get_agent_specs()

    # Create agent_type specs
    if not net_spec_keys:
        specs = {agent_net_keys[key]: specs[key] for key in specs.keys()}
    else:
        specs = {net_key: specs[value] for net_key, value in net_spec_keys.items()}

    if isinstance(policy_networks_layer_sizes, Sequence):
        policy_networks_layer_sizes = {
            key: policy_networks_layer_sizes for key in specs.keys()
        }
    if isinstance(critic_networks_layer_sizes, Sequence):
        critic_networks_layer_sizes = {
            key: critic_networks_layer_sizes for key in specs.keys()
        }

    observation_networks = {}
    policy_networks = {}
    critic_networks = {}
    for key in specs.keys():
        # TODO (dries): Make specs[key].actions
        #  return a list of specs for hybrid action space
        # Get total number of action dimensions from action spec.
        agent_act_spec = specs[key].actions
        if type(agent_act_spec) == DiscreteArray:
            num_actions = agent_act_spec.num_values
            # Question (dries): Why is the minimum -1 and not 0?
            minimum = [-1.0] * num_actions
            maximum = [1.0] * num_actions
            agent_act_spec = BoundedArray(
                shape=(num_actions,),
                minimum=minimum,
                maximum=maximum,
                dtype="float32",
                name="actions",
            )

        # Get total number of action dimensions from action spec.
        num_dimensions = np.prod(agent_act_spec.shape, dtype=int)

        # An optional network to process observations
        observation_network = tf2_utils.to_sonnet_module(tf.identity)
        # Create the policy network.
        if archecture_type == ArchitectureType.feedforward:
            policy_network = [
                networks.LayerNormMLP(
                    policy_networks_layer_sizes[key], activate_final=True, seed=seed
                ),
            ]
        elif archecture_type == ArchitectureType.recurrent:
            policy_network = [
                networks.LayerNormMLP(
                    policy_networks_layer_sizes[key][:-1],
                    activate_final=True,
                    seed=seed,
                ),
                snt.LSTM(policy_networks_layer_sizes[key][-1]),
            ]

        policy_network += [
            networks.NearZeroInitializedLinear(num_dimensions, seed=seed),
            networks.TanhToSpec(agent_act_spec),
        ]

        # Add Gaussian noise for simple exploration.
        if sigma and sigma > 0.0:
            policy_network += [
                networks.ClippedGaussian(sigma),
                networks.ClipToSpec(agent_act_spec),
            ]

        policy_network = policy_network_func(policy_network)

        # Create the critic network.
        critic_network = [
            # The multiplexer concatenates the observations/actions.
            networks.CriticMultiplexer()
        ]

        # Only for mad4pg
        if vmin and vmax and num_atoms:
            critic_network += [
                networks.LayerNormMLP(
                    critic_networks_layer_sizes[key],
                    activate_final=False,
                    seed=seed,
                ),
                DiscreteValuedHead(vmin, vmax, num_atoms),
            ]
        # maddpg
        else:
            critic_network += [
                networks.LayerNormMLP(
                    list(critic_networks_layer_sizes[key]) + [1],
                    activate_final=False,
                    seed=seed,
                )
            ]

        critic_network = snt.Sequential(critic_network)

        observation_networks[key] = observation_network
        policy_networks[key] = policy_network
        critic_networks[key] = critic_network

    return {
        "policies": policy_networks,
        "critics": critic_networks,
        "observations": observation_networks,
    }
