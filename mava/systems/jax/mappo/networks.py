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
import dataclasses
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import haiku as hk  # type: ignore
import jax
import jax.numpy as jnp
import numpy as np
from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils
from dm_env import specs as dm_specs

from mava import specs as mava_specs

Array = dm_specs.Array
BoundedArray = dm_specs.BoundedArray
DiscreteArray = dm_specs.DiscreteArray
EntropyFn = Callable[[Any], jnp.ndarray]


@dataclasses.dataclass
class PPONetworks:
    """TODO: Add description here."""

    def __init__(
        self,
        network: networks_lib.FeedForwardNetwork,
        params: networks_lib.Params,
        log_prob: Optional[networks_lib.LogProbFn] = None,
        entropy: Optional[EntropyFn] = None,
        sample: Optional[networks_lib.SampleFn] = None,
    ) -> None:
        """TODO: Add description here."""
        self.network = network
        self.params = params
        self.log_prob = log_prob
        self.entropy = entropy
        self.sample = sample

    def get_action(
        self, observations: networks_lib.Observation, key: networks_lib.PRNGKey
    ) -> Tuple[Dict, Dict]:
        """TODO: Add description here."""
        distribution, _ = self.network.apply(self.params, observations)
        actions = np.array(
            jax.numpy.squeeze(distribution.sample(seed=key)), dtype=np.int64
        )
        log_prob = np.array(distribution.log_prob(actions), dtype=np.float32)
        return actions, {"log_prob": log_prob}

    def get_value(self, observations: networks_lib.Observation) -> jnp.ndarray:
        """TODO: Add description here."""
        value, _ = self.network.apply(self.params, observations)
        return value


def make_ppo_polcy_network(
    network: networks_lib.FeedForwardNetwork, params: Dict[str, jnp.ndarray]
) -> PPONetworks:
    """TODO: Add description here."""
    return PPONetworks(
        network=network,
        params=params,
        log_prob=lambda distribution, action: distribution.log_prob(action),
        entropy=lambda distribution: distribution.entropy(),
        sample=lambda distribution, key: distribution.sample(seed=key),
    )


def make_ppo_critic_network(
    network: networks_lib.FeedForwardNetwork, params: Dict[str, jnp.ndarray]
) -> PPONetworks:
    """TODO: Add description here."""
    return PPONetworks(network=network, params=params)


def make_networks(
    spec: specs.EnvironmentSpec,
    key: networks_lib.PRNGKey,
    policy_layer_sizes: Sequence[int] = (64, 64),
    value_layer_sizes: Sequence[int] = (64, 64),
) -> Tuple[PPONetworks, PPONetworks]:
    """TODO: Add description here."""
    if isinstance(spec.actions, specs.DiscreteArray):
        return make_discrete_networks(
            environment_spec=spec,
            key=key,
            policy_layer_sizes=policy_layer_sizes,
            value_layer_sizes=value_layer_sizes,
        )
    else:
        raise NotImplementedError(
            "Continuous networks not implemented yet."
            + "See: https://github.com/deepmind/acme/blob/"
            + "master/acme/agents/jax/ppo/networks.py"
        )


def make_discrete_networks(
    environment_spec: specs.EnvironmentSpec,
    key: networks_lib.PRNGKey,
    policy_layer_sizes: Sequence[int],
    value_layer_sizes: Sequence[int],
) -> Tuple[PPONetworks, PPONetworks]:
    """TODO: Add description here."""

    num_actions = environment_spec.actions.num_values

    def policy_fn(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:
        policy_network = hk.Sequential(
            [
                utils.batch_concat,
                hk.nets.MLP(policy_layer_sizes, activation=jax.nn.relu),
                networks_lib.CategoricalValueHead(num_values=num_actions),
            ]
        )
        return policy_network(inputs)

    def critic_fn(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:
        value_network = hk.Sequential(
            [
                utils.batch_concat,
                hk.nets.MLP(value_layer_sizes, activation=jnp.tanh),
                hk.Linear(1),
                lambda x: jnp.squeeze(x, axis=-1),
            ]
        )
        return value_network(inputs)

    # Transform into pure functions.
    policy_fn = hk.without_apply_rng(hk.transform(policy_fn))
    critic_fn = hk.without_apply_rng(hk.transform(critic_fn))

    dummy_obs = utils.zeros_like(environment_spec.observations.observation)
    dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.

    policy_key, key = jax.random.split(key)
    policy_params = policy_fn.init(policy_key, dummy_obs)  # type: ignore
    critic_key, key = jax.random.split(key)
    critic_params = critic_fn.init(critic_key, dummy_obs)  # type: ignore

    # Create PPONetworks to add functionality required by the agent.
    policy_network = make_ppo_polcy_network(network=policy_fn, params=policy_params)
    critic_network = make_ppo_critic_network(network=critic_fn, params=critic_params)
    return policy_network, critic_network


def make_default_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    agent_net_keys: Dict[str, str],
    rng_key: List[int],
    net_spec_keys: Dict[str, str] = {},
    # policy_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (
    #     256,
    #     256,
    #     256,
    # ),
    # critic_networks_layer_sizes: Union[Dict[str, Sequence], Sequence]
    # = (512, 512, 256),
    # architecture_type: ArchitectureType = ArchitectureType.feedforward,
    # observation_network: Any = None,
) -> Dict[str, Any]:
    """Description here"""

    # Create agent_type specs.
    specs = environment_spec.get_agent_specs()
    if not net_spec_keys:
        specs = {agent_net_keys[key]: specs[key] for key in specs.keys()}
    else:
        specs = {net_key: specs[value] for net_key, value in net_spec_keys.items()}
    # Set Policy function and layer size
    # Default size per arch type.
    # if architecture_type == ArchitectureType.feedforward:
    #     if not policy_networks_layer_sizes:
    #         policy_networks_layer_sizes = (
    #             256,
    #             256,
    #             256,
    #         )
    # elif architecture_type == ArchitectureType.recurrent:
    #     if not policy_networks_layer_sizes:
    #         policy_networks_layer_sizes = (128, 128)

    # if isinstance(policy_networks_layer_sizes, Sequence):
    #     policy_networks_layer_sizes = {
    #         key: policy_networks_layer_sizes for key in specs.keys()
    #     }
    # if isinstance(critic_networks_layer_sizes, Sequence):
    #     critic_networks_layer_sizes = {
    #         key: critic_networks_layer_sizes for key in specs.keys()
    #     }

    policy_networks: Dict[str, Any] = {}
    critic_networks: Dict[str, Any] = {}
    for net_key in specs.keys():
        # Get the number of actions
        # num_actions = (
        #     specs[net_key].actions.num_values
        #     if isinstance(specs[net_key].actions, dm_env.specs.DiscreteArray)
        #     else np.prod(specs[net_key].actions.shape, dtype=int)
        # )
        policy_networks[net_key], critic_networks[net_key] = make_networks(
            specs[net_key], key=rng_key
        )

    return {
        "policy_networks": policy_networks,
        "critic_networks": critic_networks,
    }
