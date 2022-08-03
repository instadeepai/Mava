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

# from acme.jax.networks.rescaling import TanhToSpec
from dm_env import specs as dm_specs
from jax import jit

from mava import specs as mava_specs
from mava.utils.jax_training_utils import TanhToSpec, action_mask_categorical_policies

tfd = tfp.distributions
Array = dm_specs.Array
BoundedArray = dm_specs.BoundedArray
DiscreteArray = dm_specs.DiscreteArray
EntropyFn = Callable[[Any], jnp.ndarray]
hk_init = hk.initializers


class ClippedGaussianDistribution:
    """A clipped multivariate gaussian distribution"""

    def __init__(self, guassian_dist: Any, action_specs: Any):
        """Initialises clipped multivariate gaussian distribution"""
        self._guassian_dist = guassian_dist
        self.clip_fn = TanhToSpec(action_specs)

    def entropy(self) -> jnp.ndarray:
        """Get unclipped entropy of distribution"""
        # Note (dries): This calculates the approximate entropy of the
        # clipped Gaussian distribution by setting it to the
        # unclipped guassian entropy.
        return self._guassian_dist.entropy()

    def sample(self, seed: networks_lib.PRNGKey) -> jnp.ndarray:
        """Get sample from clipped distribution"""
        return self.clip_fn(self._guassian_dist).sample(seed=seed)

    def log_prob(self, action: jnp.ndarray) -> jnp.ndarray:
        """Get log prob from clipped distribution"""
        return self.clip_fn(self._guassian_dist).log_prob(action)


class ClippedGaussianHead(hk.Module):
    def __init__(
        self,
        action_specs: Any,
        name: Optional[str] = None,
    ):
        """Initialise clipped multivariate gaussian head"""
        super().__init__(name=name)
        self._action_specs = action_specs

    def __call__(self, x: Any) -> ClippedGaussianDistribution:
        """Clipped multivariate gaussian call"""
        return ClippedGaussianDistribution(x, action_specs=self._action_specs)


@dataclasses.dataclass
class PPONetworks:
    """IPPO network class"""

    def __init__(
        self,
        spec: specs.EnvironmentSpec,
        network: networks_lib.FeedForwardNetwork,
        params: networks_lib.Params,
        log_prob: Optional[networks_lib.LogProbFn] = None,
        entropy: Optional[EntropyFn] = None,
        sample: Optional[networks_lib.SampleFn] = None,
    ) -> None:
        """Intialialise the system executor"""

        self.network = network
        self.params = params
        self.log_prob = log_prob
        self.entropy = entropy
        self.sample = sample
        self.spec = spec

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
            if mask is not None and isinstance(self.spec, DiscreteArray):
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
        """Select an action for a single agent in the system."""

        actions, log_prob = self.forward_fn(self.params, observations, key, mask)
        if isinstance(self.spec, specs.DiscreteArray):
            actions = np.array(actions, dtype=np.int64)
        else:
            actions = np.array(actions, dtype=np.float32)
        log_prob = np.squeeze(np.array(log_prob, dtype=np.float32))
        return actions, {"log_prob": log_prob}

    def get_value(self, observations: networks_lib.Observation) -> jnp.ndarray:
        """Get value for single observation"""

        _, value = self.network.apply(self.params, observations)
        return value


def make_ppo_network(
    network: networks_lib.FeedForwardNetwork,
    params: Dict[str, jnp.ndarray],
    spec: specs.EnvironmentSpec,
) -> PPONetworks:
    """Creates IPPO network."""
    return PPONetworks(
        spec=spec,
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
    """Creates IPPO network."""
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
    """Creates discrete IPPO network."""

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
    return make_ppo_network(
        spec=environment_spec.actions,
        network=forward_fn,
        params=params,
    )


def make_continuous_networks(
    environment_spec: specs.EnvironmentSpec,
    key: networks_lib.PRNGKey,
    policy_layer_sizes: Sequence[int],
    critic_layer_sizes: Sequence[int],
    observation_network: Callable = utils.batch_concat,
    # default behaviour is to flatten observations
) -> PPONetworks:
    """Creates continuous IPPO network."""
    specs = environment_spec.actions
    # agent_environment_specs
    # Get total number of action dimensions from action spec.
    num_dimensions = np.prod(environment_spec.actions.shape, dtype=int)

    def forward_fn(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:
        policy_network = hk.Sequential(
            [
                observation_network,
                hk.nets.MLP(policy_layer_sizes, activation=jax.nn.relu),
                networks_lib.MultivariateNormalDiagHead(specs, num_dimensions),
                ClippedGaussianHead(specs),
            ]
        )

        value_network = hk.Sequential(
            [
                observation_network,
                hk.nets.MLP(critic_layer_sizes, activate_final=True),
                hk.Linear(1),
                lambda x: jnp.squeeze(x, axis=-1),
            ]
        )

        action_distribution = policy_network(inputs)
        value = value_network(inputs)
        return (action_distribution, value)

    # Transform into pure functions.
    forward_fn = hk.without_apply_rng(hk.transform(forward_fn))

    dummy_obs = utils.zeros_like(environment_spec.observations.observation)
    dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.

    # print(dummy_obs.shape)
    # exit()
    network_key, key = jax.random.split(key)
    params = forward_fn.init(network_key, dummy_obs)  # type: ignore

    # Create PPONetworks to add functionality required by the agent.
    return make_ppo_network(
        spec=specs,
        network=forward_fn,
        params=params,
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
) -> Dict[str, Any]:
    """Create default networks"""

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
