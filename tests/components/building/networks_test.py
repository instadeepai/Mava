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

"""Network unit test"""

from typing import Any, Callable, Dict, List, Sequence

import haiku as hk
import jax
import jax.numpy as jnp
import pytest
from acme.jax import networks as networks_lib
from acme.jax import utils

from mava.components.building import DefaultNetworks
from mava.components.building.networks import Networks, NetworksConfig
from mava.core_jax import SystemBuilder
from mava.specs import MAEnvironmentSpec
from mava.systems import Builder
from mava.utils.networks_utils import MLP_NORM
from tests.mocks import make_fake_env_specs


@pytest.fixture
def test_network_factory() -> Callable:
    """Pytest fixture for network factory.

    Returns:
        Network factory using custom make_default_networks.
    """

    def make_default_networks(
        environment_spec: MAEnvironmentSpec,
        agent_net_keys: Dict[str, str],
        base_key: List[int],
        net_spec_keys: Dict[str, str] = {},
        policy_layer_sizes: Sequence[int] = (
            256,
            256,
            256,
        ),
        critic_layer_sizes: Sequence[int] = (512, 512, 256),
    ) -> Dict[str, Any]:
        """Create default networks"""
        net_keys = {"net_1", "net_2", "net_3"}
        networks = {}

        for net_key in net_keys:
            networks[net_key] = {
                "environment_spec": environment_spec,
                "agent_net_keys": agent_net_keys,
                "base_key": base_key,
                "net_spec_keys": net_spec_keys,
                "policy_layer_sizes": policy_layer_sizes,
                "critic_layer_sizes": critic_layer_sizes,
                "net_key": net_key,
            }

        return networks

    def network_factory(*args: Any, **kwargs: Any) -> Any:
        """Network factory"""
        return make_default_networks(  # type: ignore
            policy_layer_sizes=(64, 64),
            critic_layer_sizes=(64, 64, 64),
            *args,
            **kwargs,
        )

    return network_factory


@pytest.fixture
def test_builder() -> SystemBuilder:
    """Pytest fixture for system builder.

    Adds mock env specs and agent net keys to store.

    Returns:
        System builder with no components.
    """
    test_builder = Builder(components=[])
    test_builder.store.ma_environment_spec = make_fake_env_specs()
    test_builder.store.agent_net_keys = {
        "net_key1": "network_1",
        "net_key2": "network_2",
    }
    return test_builder


@pytest.fixture
def test_default_networks(test_network_factory: Callable) -> Networks:
    """Pytest fixture for default networks.

    Args:
        test_network_factory: factory to use in default networks config.

    Returns:
        Default networks test component.
    """
    networks_config = NetworksConfig()
    networks_config.network_factory = test_network_factory
    networks_config.seed = 919

    return DefaultNetworks(networks_config)


def test_key_in_store(
    test_default_networks: Networks, test_builder: SystemBuilder
) -> None:
    """Test if key is loaded into the store by build.

    Args:
        test_default_networks: Pytest fixture for default networks component.
        test_builder: Pytest fixture for test system builder

    Returns:
        None.
    """
    assert not hasattr(test_builder.store, "base_key")
    test_default_networks.on_building_init_start(test_builder)
    assert hasattr(test_builder.store, "base_key")
    assert isinstance(
        test_builder.store.base_key, jax.random.PRNGKeyArray
    ) or isinstance(test_builder.store.base_key, jax.numpy.DeviceArray)


def test_config_set(test_default_networks: Networks) -> None:
    """Test if config is set by component init.

    Args:
        test_default_networks: Pytest fixture for default networks component.

    Returns:
        None.
    """
    assert test_default_networks.config.seed == 919


def test_network_factory_environment_spec(
    test_default_networks: Networks, test_builder: SystemBuilder
) -> None:
    """Test if environment spec is given to the network factory and stored by build.

    Args:
        test_default_networks: Pytest fixture for default networks component.
        test_builder: Pytest fixture for test system builder

    Returns:
        None.
    """
    test_default_networks.on_building_init_start(test_builder)
    networks = test_builder.store.network_factory()

    for network in networks.values():
        assert list(network["environment_spec"]._keys) == list(
            make_fake_env_specs().get_agent_environment_specs().keys()
        )
        assert (
            network["environment_spec"]
            .get_agent_environment_specs()["agent_0"]
            .observations.observation.shape
            == list(make_fake_env_specs().get_agent_environment_specs().values())[
                0
            ].observations.observation.shape
        )


def test_network_factory_agent_net_keys(
    test_default_networks: Networks, test_builder: SystemBuilder
) -> None:
    """Test if net keys are given to the network factory and stored by build.

    Args:
        test_default_networks: Pytest fixture for default networks component.
        test_builder: Pytest fixture for test system builder

    Returns:
        None.
    """
    test_default_networks.on_building_init_start(test_builder)
    networks = test_builder.store.network_factory()

    for network in networks.values():
        assert network["agent_net_keys"] == {
            "net_key1": "network_1",
            "net_key2": "network_2",
        }


def test_network_factory_rng_keys(
    test_default_networks: Networks, test_builder: SystemBuilder
) -> None:
    """Test if rng keys are given to the network factory and stored by build.

    Args:
        test_default_networks: Pytest fixture for default networks component.
        test_builder: Pytest fixture for test system builder

    Returns:
        None.
    """
    test_default_networks.on_building_init_start(test_builder)
    networks = test_builder.store.network_factory()

    keys = []
    for network in networks.values():
        # Ensure keys are all the correct type
        assert isinstance(network["base_key"], jax.random.PRNGKeyArray) or isinstance(
            network["base_key"], jax.numpy.DeviceArray
        )
        keys.append(tuple(network["base_key"].tolist()))

    # Ensure network factory passes the same key along to each network initialisation
    assert len(set(keys)) == 1


@pytest.fixture
def test_recurrent_network_factory() -> Callable:
    """Pytest fixture for network factory.

    Returns:
        Network factory using custom make_default_networks.
    """

    def make_default_networks(
        environment_spec: MAEnvironmentSpec,
        agent_net_keys: Dict[str, str],
        base_key: List[int],
        net_spec_keys: Dict[str, str] = {},
        policy_layer_sizes: Sequence[int] = (
            256,
            256,
            256,
        ),
        policy_recurrent_layer_sizes: Sequence[int] = (256,),
        critic_layer_sizes: Sequence[int] = (512, 512, 256),
    ) -> Dict[str, Any]:
        """Creates default networks"""
        net_keys = {"net_1", "net_2", "net_3"}
        networks = {}

        for net_key in net_keys:
            networks[net_key] = {
                "environment_spec": environment_spec,
                "agent_net_keys": agent_net_keys,
                "base_key": base_key,
                "net_spec_keys": net_spec_keys,
                "policy_layer_sizes": policy_layer_sizes,
                "critic_layer_sizes": critic_layer_sizes,
                "policy_recurrent_layer_sizes": policy_recurrent_layer_sizes,
                "net_key": net_key,
            }

        return networks

    def network_factory(*args: Any, **kwargs: Any) -> Any:
        return make_default_networks(  # type: ignore
            policy_layer_sizes=(64, 64),
            critic_layer_sizes=(64, 64, 64),
            policy_recurrent_layer_sizes=(256,),
            *args,
            **kwargs,
        )

    return network_factory


@pytest.fixture
def test_recurrent_networks(test_recurrent_network_factory: Callable) -> Networks:
    """Pytest fixture for default networks.

    Args:
        test_recurrent_network_factory: factory to use in recurrent networks config.

    Returns:
        Default networks test component.
    """
    networks_config = NetworksConfig()
    networks_config.network_factory = test_recurrent_network_factory
    networks_config.seed = 919

    return DefaultNetworks(networks_config)


def test_network_factory_recurrent_layers(
    test_recurrent_networks: Networks, test_builder: SystemBuilder
) -> None:
    """Test if rng keys are given to the network factory and stored by build.

    Args:
        test_recurrent_networks: Pytest fixture for recurrent networks component.
        test_builder: Pytest fixture for test system builder

    Returns:
        None.
    """
    test_recurrent_networks.on_building_init_start(test_builder)
    networks = test_builder.store.network_factory()

    assert networks["net_2"]["policy_recurrent_layer_sizes"] == (256,)


def test_no_network_factory_before_build(test_builder: SystemBuilder) -> None:
    """Test if network factory is not in store before build.

    Args:
        test_builder: Pytest fixture for test system builder

    Returns:
        None.
    """
    assert not hasattr(test_builder.store, "network_factory")


@hk.without_apply_rng
@hk.transform
def mlp(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:
    """Creates a standard MLP layer"""

    output_sizes = [64, 64, 64]
    mlp_network = hk.nets.MLP(output_sizes)

    return mlp_network(inputs)


@hk.without_apply_rng
@hk.transform
def mlp_norm(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:
    """Creates normalised MLP layer"""

    output_sizes = [64, 64, 64]
    mlp_network_norm = MLP_NORM(output_sizes, layer_norm=True)

    return mlp_network_norm(inputs)


@hk.without_apply_rng
@hk.transform
def mlp_norm_false(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:
    """Creates un-normalised MLP layer"""

    output_sizes = [64, 64, 64]
    mlp_network_norm = MLP_NORM(output_sizes, layer_norm=False)

    return mlp_network_norm(inputs)


def test_MLP_with_feature_norm() -> None:
    """Test if layer normalisation is added correctly to the hidden layers"""

    base_key = jax.random.PRNGKey(32)
    dummy_obs = jnp.ones(16)
    dummy_obs = utils.add_batch_dim(dummy_obs)
    _, network_key = jax.random.split(base_key)
    params = mlp.init(network_key, dummy_obs)  # type: ignore
    params_norm = mlp_norm.init(network_key, dummy_obs)
    params_norm_false = mlp_norm_false.init(network_key, dummy_obs)

    assert len(params.keys()) == len(params_norm_false.keys())
    assert len(params.keys()) * 2 == len(params_norm.keys())
