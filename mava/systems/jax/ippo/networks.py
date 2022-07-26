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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import chex
import haiku as hk  # type: ignore
import jax
import jax.numpy as jnp
import numpy as np
from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils
from dm_env import specs as dm_specs
from jax import jit

from mava import specs as mava_specs
from mava.utils.jax_training_utils import action_mask_categorical_policies

Array = dm_specs.Array
BoundedArray = dm_specs.BoundedArray
DiscreteArray = dm_specs.DiscreteArray
EntropyFn = Callable[[Any], jnp.ndarray]

# from jax.config import config
# config.update('jax_disable_jit', True)


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

        @jit
        def forward_fn(
            params: Dict[str, jnp.ndarray],
            observations: networks_lib.Observation,
            key: networks_lib.PRNGKey,
            mask: chex.Array = None,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """TODO: Add description here."""
            # The parameters of the network might change. So it has to
            # be fed into the jitted function.
            distribution, _ = self.network.apply(params, observations)
            if mask is not None:
                distribution = action_mask_categorical_policies(distribution, mask)

            actions = jax.numpy.squeeze(distribution.sample(seed=key))
            log_prob = distribution.log_prob(actions)

            return actions, log_prob

        self.forward_fn = forward_fn

    def get_action(
        self,
        observations: networks_lib.Observation,
        key: networks_lib.PRNGKey,
        mask: chex.Array = None,
    ) -> Tuple[np.ndarray, Dict]:
        """TODO: Add description here."""
        actions, log_prob = self.forward_fn(self.params, observations, key, mask)
        actions = np.array(actions, dtype=np.int64)
        log_prob = np.squeeze(np.array(log_prob, dtype=np.float32))
        return actions, {"log_prob": log_prob}

    def get_value(self, observations: networks_lib.Observation) -> jnp.ndarray:
        """TODO: Add description here."""
        _, value = self.network.apply(self.params, observations)
        return value


# This class is made to replicate the behaviour of the categorical vlaue head
# whcih squeezes the value inside the __call__ method.
class ValueHead(hk.Module):
    """Network head that produces a value."""

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        """_summary_"""
        super().__init__(name=name)
        self._value_layer = hk.Linear(1)

    def __call__(self, inputs: jnp.ndarray) -> Any:
        """_summary_"""
        value = jnp.squeeze(self._value_layer(inputs), axis=-1)
        return value


@dataclasses.dataclass
class PPOSeparateNetworks:
    """TODO: Add description here."""

    def __init__(
        self,
        policy_network: networks_lib.FeedForwardNetwork,
        policy_params: networks_lib.Params,
        critic_network: networks_lib.FeedForwardNetwork,
        critic_params: networks_lib.Params,
        log_prob: Optional[networks_lib.LogProbFn] = None,
        entropy: Optional[EntropyFn] = None,
        sample: Optional[networks_lib.SampleFn] = None,
    ) -> None:
        """TODO: Add description here."""
        self.policy_network = policy_network
        self.policy_params = policy_params
        self.critic_network = critic_network
        self.critic_params = critic_params
        self.log_prob = log_prob
        self.entropy = entropy
        self.sample = sample

        @jit
        def forward_fn(
            policy_params: Dict[str, jnp.ndarray],
            observations: networks_lib.Observation,
            key: networks_lib.PRNGKey,
            mask: chex.Array = None,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """TODO: Add description here."""
            # The parameters of the network might change. So it has to
            # be fed into the jitted function.
            distribution = self.policy_network.apply(policy_params, observations)
            if mask is not None:
                distribution = action_mask_categorical_policies(distribution, mask)

            actions = jax.numpy.squeeze(distribution.sample(seed=key))
            log_prob = distribution.log_prob(actions)

            return actions, log_prob

        self.forward_fn = forward_fn

    def get_action(
        self,
        observations: networks_lib.Observation,
        key: networks_lib.PRNGKey,
        mask: chex.Array = None,
    ) -> Tuple[np.ndarray, Dict]:
        """TODO: Add description here."""
        actions, log_prob = self.forward_fn(self.policy_params, observations, key, mask)
        actions = np.array(actions, dtype=np.int64)
        log_prob = np.squeeze(np.array(log_prob, dtype=np.float32))
        return actions, {"log_prob": log_prob}

    def get_value(self, observations: networks_lib.Observation) -> jnp.ndarray:
        """TODO: Add description here."""
        value = self.critic_network.apply(self.critic_params, observations)
        value = jnp.squeeze(value, axis=-1)
        return value


def make_ppo_network(
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


def make_ppo_networks(
    policy_network: networks_lib.FeedForwardNetwork,
    policy_params: Dict[str, jnp.ndarray],
    critic_network: networks_lib.FeedForwardNetwork,
    critic_params: Dict[str, jnp.ndarray],
) -> PPOSeparateNetworks:
    """TODO: Add description here."""
    return PPOSeparateNetworks(
        policy_network=policy_network,
        policy_params=policy_params,
        critic_network=critic_network,
        critic_params=critic_params,
        log_prob=lambda distribution, action: distribution.log_prob(action),
        entropy=lambda distribution: distribution.entropy(),
        sample=lambda distribution, key: distribution.sample(seed=key),
    )


def make_networks(
    agent_keys: List[str],
    spec: specs.EnvironmentSpec,
    key: networks_lib.PRNGKey,
    policy_layer_sizes: Sequence[int] = (
        256,
        256,
        256,
    ),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    observation_network: Callable = utils.batch_concat,
    single_network: bool = True,
) -> Union[PPONetworks, PPOSeparateNetworks]:
    """TODO: Add description here."""
    if isinstance(spec.actions, specs.DiscreteArray):
        return make_discrete_networks(
            agent_keys=agent_keys,
            environment_spec=spec,
            key=key,
            policy_layer_sizes=policy_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
            observation_network=observation_network,
            single_network=single_network,
        )

    else:
        raise NotImplementedError(
            "Continuous networks not implemented yet."
            + "See: https://github.com/deepmind/acme/blob/"
            + "master/acme/agents/jax/ppo/networks.py"
        )


def make_discrete_networks(
    agent_keys: List[str],
    environment_spec: specs.EnvironmentSpec,
    key: networks_lib.PRNGKey,
    policy_layer_sizes: Sequence[int],
    critic_layer_sizes: Sequence[int],
    observation_network: Callable = utils.batch_concat,
    single_network: bool = False,
    # default behaviour is to flatten observations
) -> Union[PPONetworks, PPOSeparateNetworks]:
    """TODO: Add description here."""

    num_actions = environment_spec.actions.num_values
    # TODO (dries): Investigate if one forward_fn function is slower
    # than having a policy_fn and critic_fn. Maybe jit solves
    # this issue. Having one function makes obs network calculations
    # easier.
    print(agent_keys)

    def policy_fn(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:
        policy_network = hk.Sequential(
            [
                observation_network,
                hk.nets.MLP(policy_layer_sizes, activation=jax.nn.relu),
                networks_lib.CategoricalHead(num_values=num_actions),
            ]
        )
        return policy_network(inputs)

    def critic_fn(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:
        critic_network = hk.Sequential(
            [
                observation_network,
                hk.nets.MLP(critic_layer_sizes, activation=jax.nn.relu),
                ValueHead(),
            ]
        )
        return critic_network(inputs)

    # Transform into pure functions.
    policy_fn = hk.without_apply_rng(hk.transform(policy_fn))
    critic_fn = hk.without_apply_rng(hk.transform(critic_fn))

    dummy_obs = utils.zeros_like(environment_spec.observations.observation)
    dummy_obs_critic = dummy_obs

    agent_num = len(agent_keys)
    for i in range(agent_num - 1):
        dummy_obs_critic = jax.numpy.concatenate([dummy_obs_critic, dummy_obs])

    dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.
    dummy_obs_critic = utils.add_batch_dim(dummy_obs_critic)

    network_key, key = jax.random.split(key)
    policy_params = policy_fn.init(network_key, dummy_obs)  # type: ignore

    network_key, key = jax.random.split(key)
    critic_params = critic_fn.init(network_key, dummy_obs_critic)  # type: ignore

    # Create PPONetworks to add functionality required by the agent.
    return make_ppo_networks(
        policy_network=policy_fn,
        policy_params=policy_params,
        critic_network=critic_fn,
        critic_params=critic_params,
    )


def make_default_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    agent_net_keys: Dict[str, str],
    rng_key: List[int],
    net_spec_keys: Dict[str, str] = {},
    policy_layer_sizes: Sequence[int] = (
        256,
        256,
        256,
    ),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    observation_network: Callable = utils.batch_concat,
    single_network: bool = True,
) -> Dict[str, Any]:
    """Create default PPO networks

    Args:
        environment_spec: mava multi-agent environment spec
        agent_net_keys: dicitonary specifiying which networks are
                        used by which agent
        rng_key: jax random key to be used for network initialization
        net_spec_keys: TODO (Ruan) find these details
        policy_layer_sizes: policy network layers
        critic_layer_sizes: critic network layers
        observation_network: network for processing environment observations
                             defaults to flattening observations but could be
                             a CNN or similar observation processing network
        single_network: details whether a single network with a policy and value
                        heads should be used if true or whether separate policy
                        and critic networks should be used if false.
    """

    # Create agent_type specs.
    specs = environment_spec.get_agent_environment_specs()
    agent_keys = None
    if not net_spec_keys:
        agent_keys = list(specs.keys())
        specs = {agent_net_keys[key]: specs[key] for key in specs.keys()}
    else:
        agent_keys = list(net_spec_keys)
        specs = {net_key: specs[value] for net_key, value in net_spec_keys.items()}

    networks: Dict[str, Any] = {}
    for net_key in specs.keys():
        networks[net_key] = make_networks(
            agent_keys,
            specs[net_key],
            key=rng_key,
            policy_layer_sizes=policy_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
            observation_network=observation_network,
            single_network=single_network,
        )

    return {
        "networks": networks,
    }
