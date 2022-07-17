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

import chex
import haiku as hk  # type: ignore
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability as tfp
from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax.networks.rescaling import TanhToSpec
from dm_env import specs as dm_specs
from jax import jit

from mava import specs as mava_specs
from mava.utils.jax_training_utils import action_mask_categorical_policies

tfd = tfp.distributions
Array = dm_specs.Array
BoundedArray = dm_specs.BoundedArray
DiscreteArray = dm_specs.DiscreteArray
EntropyFn = Callable[[Any], jnp.ndarray]


class ClippedGaussianDistribution:
    def __init__(self, guassian_dist: Any, action_specs: Any):
        self._guassian_dist = guassian_dist
        self.clip_fn = TanhToSpec(action_specs)

    def entropy(self):
        # Note (dries): This calculates the approximate entropy of the
        # clipped Gaussian distribution by setting it to the
        # unclipped guassian entropy.
        return self._guassian_dist.entropy()

    def sample(self, seed):
        # print(self._guassian_dist)
        # return self.clip_fn(self._guassian_dist).sample(seed=seed)
        return self._guassian_dist.sample(seed=seed)

    def log_prob(self, action):
        # return self.clip_fn(self._guassian_dist).log_prob(action)
        return self._guassian_dist.log_prob(action)

    def batch_reshape(self, dims, name: str = "reshaped") -> None:
        self._guassian_dist = tfd.BatchReshape(
            self._guassian_dist, batch_shape=dims, name=name
        )


class ClippedGaussianHead(hk.Module):
    def __init__(
        self,
        action_specs: Any,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self._action_specs = action_specs

    def __call__(self, x: Any) -> ClippedGaussianDistribution:
        return ClippedGaussianDistribution(x, action_specs=self._action_specs)


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

        # @jit
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
            # if mask is not None:
            # distribution = action_mask_categorical_policies(distribution, mask)
            # print(distribution)
            # exit()
            actions = jax.numpy.squeeze(distribution.sample(seed=key))
            print(actions)
            log_prob = distribution.log_prob(actions)
            # print(log_prob)
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
        # actions = np.array(actions, dtype=np.int64)
        # Continuous actions a re floats
        actions = np.array(actions, dtype=np.float64)
        log_prob = np.squeeze(np.array(log_prob, dtype=np.float32))
        return actions, {"log_prob": log_prob}

    def get_value(self, observations: networks_lib.Observation) -> jnp.ndarray:
        """TODO: Add description here."""
        _, value = self.network.apply(self.params, observations)
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


def make_networks(
    spec: specs.EnvironmentSpec,
    key: networks_lib.PRNGKey,
    policy_layer_sizes: Sequence[int] = (
        256,
        256,
        256,
    ),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    observation_network: Callable = utils.batch_concat,
) -> PPONetworks:
    """TODO: Add description here."""
    if isinstance(spec.actions, specs.DiscreteArray):
        return make_discrete_networks(
            environment_spec=spec,
            key=key,
            policy_layer_sizes=policy_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
            observation_network=observation_network,
        )

    else:
        return make_continuous_networks(
            environment_spec=spec,
            key=key,
            policy_layer_sizes=policy_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
            observation_network=observation_network,
        )


def make_discrete_networks(
    environment_spec: specs.EnvironmentSpec,
    key: networks_lib.PRNGKey,
    policy_layer_sizes: Sequence[int],
    critic_layer_sizes: Sequence[int],
    observation_network: Callable = utils.batch_concat,
    # default behaviour is to flatten observations
) -> PPONetworks:
    """TODO: Add description here."""

    num_actions = environment_spec.actions.num_values

    # TODO (dries): Investigate if one forward_fn function is slower
    # than having a policy_fn and critic_fn. Maybe jit solves
    # this issue. Having one function makes obs network calculations
    # easier.
    def forward_fn(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:
        policy_value_network = hk.Sequential(
            [
                observation_network,
                hk.nets.MLP(policy_layer_sizes, activation=jax.nn.relu),
                networks_lib.CategoricalValueHead(num_values=num_actions),
            ]
        )
        return policy_value_network(inputs)

    # Transform into pure functions.
    forward_fn = hk.without_apply_rng(hk.transform(forward_fn))

    dummy_obs = utils.zeros_like(environment_spec.observations.observation)
    dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.

    network_key, key = jax.random.split(key)
    params = forward_fn.init(network_key, dummy_obs)  # type: ignore

    # Create PPONetworks to add functionality required by the agent.
    return make_ppo_network(network=forward_fn, params=params)


def make_continuous_networks(
    environment_spec: specs.EnvironmentSpec,
    key: networks_lib.PRNGKey,
    policy_layer_sizes: Sequence[int],
    critic_layer_sizes: Sequence[int],
    observation_network: Callable = utils.batch_concat,
    # default behaviour is to flatten observations
) -> PPONetworks:
    """TODO: Add description here."""
    specs = environment_spec.actions
    # agent_environment_specs
    # Get total number of action dimensions from action spec.
    num_dimensions = np.prod(environment_spec.actions.shape, dtype=int)

    def forward_fn(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:
        policy_network = hk.Sequential(
            [
                utils.batch_concat,
                observation_network,
                hk.nets.MLP(policy_layer_sizes, activation=jax.nn.relu),
                networks_lib.MultivariateNormalDiagHead(num_dimensions),
                ClippedGaussianHead(specs),
            ]
        )

        value_network = hk.Sequential(
            [
                utils.batch_concat,
                observation_network,
                hk.nets.MLP(critic_layer_sizes, activate_final=True),
                hk.Linear(1),
                lambda x: jnp.squeeze(x, axis=-1),
            ]
        )

        action_distribution = policy_network(inputs)
        value = value_network(inputs)
        # print("*****ACTION*****")
        # print(action_distribution)
        # exit()
        return (action_distribution, value)

    # Transform into pure functions.
    forward_fn = hk.without_apply_rng(hk.transform(forward_fn))

    dummy_obs = utils.zeros_like(environment_spec.observations.observation)
    dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.

    network_key, key = jax.random.split(key)
    params = forward_fn.init(network_key, dummy_obs)  # type: ignore

    # Create PPONetworks to add functionality required by the agent.
    return make_ppo_network(network=forward_fn, params=params)


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
) -> Dict[str, Any]:
    """Description here"""

    # Create agent_type specs.
    specs = environment_spec.get_agent_environment_specs()
    if not net_spec_keys:
        specs = {agent_net_keys[key]: specs[key] for key in specs.keys()}
    else:
        specs = {net_key: specs[value] for net_key, value in net_spec_keys.items()}

    networks: Dict[str, Any] = {}
    for net_key in specs.keys():
        networks[net_key] = make_networks(
            specs[net_key],
            key=rng_key,
            policy_layer_sizes=policy_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
            observation_network=observation_network,
        )

    return {
        "networks": networks,
    }
