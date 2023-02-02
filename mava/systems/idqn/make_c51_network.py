from typing import Any, Callable, Dict, Optional, Sequence

import haiku as hk
import jax
import jax.numpy as jnp
from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils

from mava import specs as mava_specs
from mava.systems.idqn.c51_network import C51DuellingMLP, C51DuellingNetwork


def make_c51_network(
    environment_spec: specs.EnvironmentSpec,
    base_key: jax.random.KeyArray,
    policy_layer_sizes: Sequence[int],
    v_max: int,
    v_min: int,
    num_atoms: int = 51,
    activation_function: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
    observation_network: Optional[Callable] = None,
):
    num_actions = environment_spec.actions.num_values

    @hk.without_apply_rng
    @hk.transform
    def policy_fn(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:
        # Add the observation network and an MLP network.
        value_network = []
        if observation_network is not None:
            value_network.append(observation_network)

        value_network.append(
            hk.nets.MLP(
                (*policy_layer_sizes, num_actions),
                activation=activation_function,
            ),
        )
        value_network.append(
            C51DuellingMLP(
                num_actions,
                policy_layer_sizes,
                v_max=v_max,
                v_min=v_min,
                num_atoms=num_atoms,
            )
        )

        return hk.Sequential(value_network)(inputs)

    dummy_obs = utils.zeros_like(environment_spec.observations.observation)
    dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.

    base_key, network_key = jax.random.split(base_key)

    policy_params = policy_fn.init(network_key, dummy_obs)

    base_key, network_key = jax.random.split(base_key)

    return C51DuellingNetwork(
        network=policy_fn,
        policy_params=policy_params,
    )


def make_c51_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    agent_net_keys: Dict[str, str],
    base_key: jax.random.KeyArray,
    v_min: int,
    v_max: int,
    num_atoms: int = 51,
    net_spec_keys: Dict[str, str] = {},
    policy_layer_sizes: Sequence[int] = (
        64,
        64,
    ),
    activation_function: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
    observation_network: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Create default IDQN networks (one per agent)

    Args:
        environment_spec: mava multi-agent environment spec
        agent_net_keys: dictionary specifying which networks are
            used by which agent
        base_key: jax random key to be used for network initialization
        net_spec_keys: keys for each agent network
        policy_layer_sizes: policy network layers
        activation_function: activation function to be used for network hidden layers.
        observation_network: optional network for processing observations

    Returns:
        networks: IDQN networks created to given spec
    """

    # Create agent_type specs.
    specs = environment_spec.get_agent_environment_specs()
    if not net_spec_keys:
        specs = {
            agent_net_keys[agent_key]: specs[agent_key] for agent_key in specs.keys()
        }
    else:
        specs = {net_key: specs[value] for net_key, value in net_spec_keys.items()}

    networks: Dict[str, Any] = {}
    for net_key in specs.keys():
        networks[net_key] = make_c51_network(
            specs[net_key],
            base_key=base_key,
            v_min=v_min,
            v_max=v_max,
            num_atoms=num_atoms,
            policy_layer_sizes=policy_layer_sizes,
            activation_function=activation_function,
            observation_network=observation_network,
        )

    return networks
