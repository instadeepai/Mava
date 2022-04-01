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

"""Jax MAPPO system networks."""

# TODO (Dries): make MAPPO networks
from typing import Any, Dict, Optional, Sequence, Union

import dm_env
import haiku as hk  # type: ignore
import jax
import numpy as np
from dm_env import specs

from mava import specs as mava_specs
from mava.utils.enums import ArchitectureType

Array = specs.Array
BoundedArray = specs.BoundedArray
DiscreteArray = specs.DiscreteArray


def softmax(x: np.ndarray) -> np.ndarray:
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


def make_default_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    agent_net_keys: Dict[str, str],
    net_spec_keys: Dict[str, str] = {},
    policy_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (
        256,
        256,
        256,
    ),
    critic_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (512, 512, 256),
    architecture_type: ArchitectureType = ArchitectureType.feedforward,
    observation_network: Any = None,
    seed: Optional[int] = 423,
) -> Dict[str, Any]:
    """Default networks for mappo.
    Args:
        environment_spec (mava_specs.MAEnvironmentSpec): description of the action and
            observation spaces etc. for each agent in the system.
        agent_net_keys: (dict, optional): specifies what network each agent uses.
            Defaults to {}.
        net_spec_keys: specifies the specs of each network.
        policy_networks_layer_sizes (Union[Dict[str, Sequence], Sequence], optional):
            size of policy networks. Defaults to (256, 256, 256).
        critic_networks_layer_sizes (Union[Dict[str, Sequence], Sequence], optional):
            size of critic networks. Defaults to (512, 512, 256).
        seed (int, optional): random seed for network initialization.
    Raises:
        ValueError: Unknown action_spec type, if actions aren't DiscreteArray
            or BoundedArray.
    Returns:
        Dict[str, snt.Module]: returned agent networks.
    """
    # Create agent_type specs.
    specs = environment_spec.get_agent_specs()
    if not net_spec_keys:
        specs = {agent_net_keys[key]: specs[key] for key in specs.keys()}
    else:
        specs = {net_key: specs[value] for net_key, value in net_spec_keys.items()}
    # Set Policy function and layer size
    # Default size per arch type.
    if architecture_type == ArchitectureType.feedforward:
        if not policy_networks_layer_sizes:
            policy_networks_layer_sizes = (
                256,
                256,
                256,
            )
    elif architecture_type == ArchitectureType.recurrent:
        if not policy_networks_layer_sizes:
            policy_networks_layer_sizes = (128, 128)

    if isinstance(policy_networks_layer_sizes, Sequence):
        policy_networks_layer_sizes = {
            key: policy_networks_layer_sizes for key in specs.keys()
        }
    if isinstance(critic_networks_layer_sizes, Sequence):
        critic_networks_layer_sizes = {
            key: critic_networks_layer_sizes for key in specs.keys()
        }

    rng_key = jax.random.PRNGKey(seed)

    observation_networks: Dict[str, Any] = {}
    policy_networks: Dict[str, Any] = {}
    critic_networks: Dict[str, Any] = {}
    for key in specs.keys():
        observation_network = None
        critic_network = None

        # Get the number of actions
        num_actions = (
            specs[key].actions.num_values
            if isinstance(specs[key].actions, dm_env.specs.DiscreteArray)
            else np.prod(specs[key].actions.shape, dtype=int)
        )

        def policy_net(x: np.ndarray) -> np.ndarray:
            mlp = hk.nets.MLP(
                output_sizes=policy_networks_layer_sizes[key]  # type: ignore
                + (num_actions,)
            )
            return softmax(mlp(x))

        model = hk.transform(policy_net)
        obs_size = specs[key].observations.observation.shape[0]
        params = model.init(rng_key, np.random.normal(size=(1, obs_size)))
        policy_network = (model, rng_key, params)
        rng_key, subkey = jax.random.split(rng_key)
        del subkey

        observation_networks[key] = observation_network
        policy_networks[key] = policy_network
        critic_networks[key] = critic_network

    return {
        "observation_networks": observation_networks,
        "policy_networks": policy_networks,
        "critic_networks": critic_networks,
    }
