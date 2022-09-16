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

from typing import Dict, List, Tuple

import pytest

from mava.components.building.environments import EnvironmentSpec, EnvironmentSpecConfig
from mava.components.building.system_init import (
    CustomSamplingSystemInit,
    CustomSamplingSystemInitConfig,
    FixedNetworkSystemInit,
    FixedNetworkSystemInitConfig,
    RandomSamplingSystemInit,
    RandomSamplingSystemInitConfig,
)
from mava.systems.builder import Builder
from mava.utils import enums
from tests.jax.mocks import make_fake_environment_factory


@pytest.fixture
def environment_spec() -> EnvironmentSpec:
    """Creates mock environment spec."""

    environment_spec = EnvironmentSpec(
        EnvironmentSpecConfig(environment_factory=make_fake_environment_factory())
    )
    return environment_spec


@pytest.fixture
def builder() -> Builder:
    """Creates mock builder."""

    builder = Builder(components=[])
    return builder


@pytest.fixture
def same_network_agent_net_keys() -> Dict[str, str]:
    """Creates mock agent net keys data.

    To be used for fixed agent network sampling with shared weights
    """
    return {
        "agent_0": "network_agent",
        "agent_1": "network_agent",
        "agent_2": "network_agent",
    }


@pytest.fixture
def same_network_network_sampling_setup() -> List[List]:
    """Creates mock network sampling setup data.

    To be used for fixed agent network sampling with shared weights
    """
    return [["network_agent", "network_agent", "network_agent"]]


@pytest.fixture
def same_network_unique_net_keys() -> List[str]:
    """Creates mock unique net keys data.

    To be used for fixed agent network sampling with shared weights
    """
    return ["network_agent"]


@pytest.fixture
def same_network_net_keys_to_ids() -> Dict[str, int]:
    """Creates mock network net keys to ids data.

    To be used for fixed agent network sampling with shared weights
    """
    return {"network_agent": 0}


@pytest.fixture
def different_network_agent_net_keys() -> Dict[str, str]:
    """Creates mock agent net keys data.

    To be used for fixed agent network sampling without shared weights
    """
    return {
        "agent_0": "network_agent_0",
        "agent_1": "network_agent_1",
        "agent_2": "network_agent_2",
    }


@pytest.fixture
def different_network_network_sampling_setup() -> List[List]:
    """Creates mock network sampling setup data.

    To be used for fixed agent network sampling without shared weights
    """
    return [["network_agent_0", "network_agent_1", "network_agent_2"]]


@pytest.fixture
def different_network_unique_net_keys() -> List[str]:
    """Creates mock unique net keys data.

    To be used for fixed agent network sampling without shared weights
    """
    return ["network_agent_0", "network_agent_1", "network_agent_2"]


@pytest.fixture
def different_network_net_keys_to_ids() -> Dict[str, int]:
    """Creates mock network net keys to ids data.

    To be used for fixed agent network sampling without shared weights
    """
    return {"network_agent_0": 0, "network_agent_1": 1, "network_agent_2": 2}


@pytest.fixture
def random_networks_agent_net_keys() -> Dict[str, str]:
    """Creates mock agent net keys data.

    To be used for random agent network sampling without shared weights
    """
    return {"agent_0": "network_0", "agent_1": "network_1", "agent_2": "network_2"}


@pytest.fixture
def random_networks_network_sampling_setup() -> List[List]:
    """Creates mock network sampling setup data.

    To be used for random agent network sampling without shared weights
    """
    return [["network_0"], ["network_1"], ["network_2"]]


@pytest.fixture
def random_networks_unique_net_keys() -> List[str]:
    """Creates mock unique net keys data.

    To be used for random agent network sampling without shared weights
    """
    return ["network_0", "network_1", "network_2"]


@pytest.fixture
def random_networks_net_keys_to_ids() -> Dict[str, int]:
    """Creates mock network net keys to ids data.

    To be used for random agent network sampling without shared weights
    """
    return {"network_0": 0, "network_1": 1, "network_2": 2}


@pytest.fixture
def system_with_random_agent_networks_no_shared_weights(
    environment_spec: EnvironmentSpec, builder: Builder
) -> Tuple[RandomSamplingSystemInit, Builder]:
    """Create system with random agent network sampling without shared weights."""

    environment_spec.on_building_init_start(builder)

    system = RandomSamplingSystemInit(config=RandomSamplingSystemInitConfig())

    return system, builder


@pytest.fixture
def system_with_fixed_agent_networks_no_shared_weights(
    environment_spec: EnvironmentSpec, builder: Builder
) -> Tuple[FixedNetworkSystemInit, Builder]:
    """Create system with fixed agent network sampling without shared weights."""

    environment_spec.on_building_init_start(builder)

    system = FixedNetworkSystemInit(
        config=FixedNetworkSystemInitConfig(shared_weights=False)
    )

    return system, builder


@pytest.fixture
def system_with_fixed_agent_networks_with_shared_weights(
    environment_spec: EnvironmentSpec, builder: Builder
) -> Tuple[FixedNetworkSystemInit, Builder]:
    """Create system with fixed agent network sampling and shared weights."""

    environment_spec.on_building_init_start(builder)

    system = FixedNetworkSystemInit(
        config=FixedNetworkSystemInitConfig(shared_weights=True)
    )

    return system, builder


@pytest.fixture
def system_with_custom_network_sampling_no_shared_weights(
    environment_spec: EnvironmentSpec,
    builder: Builder,
    random_networks_network_sampling_setup: List[List],
) -> Tuple[CustomSamplingSystemInit, Builder]:
    """Create system with custom network sampling setup without shared weights."""

    environment_spec.on_building_init_start(builder)

    system = CustomSamplingSystemInit(
        config=CustomSamplingSystemInitConfig(
            network_sampling_setup=random_networks_network_sampling_setup,
            shared_weights=False,
        )
    )

    return system, builder


@pytest.fixture
def system_with_custom_network_sampling_with_shared_weights(
    environment_spec: EnvironmentSpec,
    builder: Builder,
    same_network_network_sampling_setup: List[List],
) -> Tuple[CustomSamplingSystemInit, Builder]:
    """Create system with custom network sampling setup and shared weights."""

    environment_spec.on_building_init_start(builder)

    system = CustomSamplingSystemInit(
        config=CustomSamplingSystemInitConfig(
            network_sampling_setup=same_network_network_sampling_setup,
            shared_weights=True,
        )
    )

    return system, builder


@pytest.fixture
def fixed_agent_networks_with_incorrect_sampling_setup_type(
    environment_spec: EnvironmentSpec, builder: Builder
) -> Tuple[FixedNetworkSystemInit, Builder]:
    """Create system with incorrect fixed agent network sampling setup."""

    environment_spec.on_building_init_start(builder)

    system = FixedNetworkSystemInit(
        config=FixedNetworkSystemInitConfig(
            network_sampling_setup=enums.NetworkSampler.random_agent_networks,
            shared_weights=True,
        )
    )

    return system, builder


@pytest.fixture
def random_agent_networks_with_incorrect_sampling_setup_type(
    environment_spec: EnvironmentSpec, builder: Builder
) -> Tuple[RandomSamplingSystemInit, Builder]:
    """Create system with incorrect random agent network sampling setup."""

    environment_spec.on_building_init_start(builder)

    system = RandomSamplingSystemInit(
        config=RandomSamplingSystemInitConfig(
            network_sampling_setup=enums.NetworkSampler.fixed_agent_networks,
        )
    )

    return system, builder


@pytest.fixture
def custom_agent_networks_with_incorrect_sampling_setup_type(
    environment_spec: EnvironmentSpec, builder: Builder
) -> Tuple[CustomSamplingSystemInit, Builder]:
    """Create system with incorrect custom agent network sampling setup."""

    environment_spec.on_building_init_start(builder)

    system = CustomSamplingSystemInit(
        config=CustomSamplingSystemInitConfig(shared_weights=False)
    )

    return system, builder


def test_fixed_agent_networks_with_shared_weights(
    system_with_fixed_agent_networks_with_shared_weights: Tuple[
        FixedNetworkSystemInit, Builder
    ],
    same_network_agent_net_keys: Dict[str, str],
    same_network_network_sampling_setup: List[List],
    same_network_unique_net_keys: List[str],
    same_network_net_keys_to_ids: Dict[str, int],
) -> None:
    """Test whether system network sampling is instantiated correctly.

    Done for fixed agent networks with shared weights
    """

    system, builder = system_with_fixed_agent_networks_with_shared_weights
    system.on_building_init(builder)

    assert builder.store.agent_net_keys == same_network_agent_net_keys
    assert builder.store.network_sampling_setup == same_network_network_sampling_setup
    assert builder.store.unique_net_keys == same_network_unique_net_keys
    assert builder.store.net_keys_to_ids == same_network_net_keys_to_ids


def test_fixed_agent_networks_no_shared_weights(
    system_with_fixed_agent_networks_no_shared_weights: Tuple[
        FixedNetworkSystemInit, Builder
    ],
    different_network_agent_net_keys: Dict[str, str],
    different_network_network_sampling_setup: List[List],
    different_network_unique_net_keys: List[str],
    different_network_net_keys_to_ids: Dict[str, int],
) -> None:
    """Test whether system network sampling is instantiated correctly.

    Done for fixed agent networks without shared weights
    """

    system, builder = system_with_fixed_agent_networks_no_shared_weights
    system.on_building_init(builder)

    assert builder.store.agent_net_keys == different_network_agent_net_keys
    assert (
        builder.store.network_sampling_setup == different_network_network_sampling_setup
    )
    assert builder.store.unique_net_keys == different_network_unique_net_keys
    assert builder.store.net_keys_to_ids == different_network_net_keys_to_ids


def test_random_agent_networks_no_shared_weights(
    system_with_random_agent_networks_no_shared_weights: Tuple[
        RandomSamplingSystemInit, Builder
    ],
    random_networks_agent_net_keys: Dict[str, str],
    random_networks_network_sampling_setup: List[List],
    random_networks_unique_net_keys: List[str],
    random_networks_net_keys_to_ids: Dict[str, int],
) -> None:
    """Test whether system network sampling is instantiated correctly.

    Done for random agent networks without shared weights
    """

    system, builder = system_with_random_agent_networks_no_shared_weights
    system.on_building_init(builder)

    assert builder.store.agent_net_keys == random_networks_agent_net_keys
    assert (
        builder.store.network_sampling_setup == random_networks_network_sampling_setup
    )
    assert builder.store.unique_net_keys == random_networks_unique_net_keys
    assert builder.store.net_keys_to_ids == random_networks_net_keys_to_ids


def test_custom_network_sampling_with_shared_weights(
    system_with_custom_network_sampling_with_shared_weights: Tuple[
        CustomSamplingSystemInit, Builder
    ],
    same_network_agent_net_keys: Dict[str, str],
    same_network_unique_net_keys: List[str],
    same_network_net_keys_to_ids: Dict[str, int],
) -> None:
    """Test whether system network sampling is instantiated correctly.

    Done for custom network sampling setup with shared weights
    """

    system, builder = system_with_custom_network_sampling_with_shared_weights
    system.on_building_init(builder)

    assert builder.store.agent_net_keys == same_network_agent_net_keys
    assert builder.store.unique_net_keys == same_network_unique_net_keys
    assert builder.store.net_keys_to_ids == same_network_net_keys_to_ids


def test_custom_network_sampling_no_shared_weights(
    system_with_custom_network_sampling_no_shared_weights: Tuple[
        CustomSamplingSystemInit, Builder
    ],
    random_networks_unique_net_keys: List[str],
    random_networks_net_keys_to_ids: Dict[str, int],
) -> None:
    """Test whether system network sampling is instantiated correctly.

    Done for custom network sampling setup without shared weights
    """

    system, builder = system_with_custom_network_sampling_no_shared_weights
    system.on_building_init(builder)

    assert list(builder.store.agent_net_keys.keys()) == builder.store.agents
    assert set(list(builder.store.agent_net_keys.values())).issubset(
        random_networks_unique_net_keys
    )
    assert builder.store.unique_net_keys == random_networks_unique_net_keys
    assert builder.store.net_keys_to_ids == random_networks_net_keys_to_ids


def test_incorrect_sampling_setup_type(
    fixed_agent_networks_with_incorrect_sampling_setup_type: Tuple[
        FixedNetworkSystemInit, Builder
    ],
    random_agent_networks_with_incorrect_sampling_setup_type: Tuple[
        RandomSamplingSystemInit, Builder
    ],
    custom_agent_networks_with_incorrect_sampling_setup_type: Tuple[
        CustomSamplingSystemInit, Builder
    ],
) -> None:
    """Tests all system network sampling instantiation ValueErrors."""

    (
        fixed_network_system,
        fixed_network_builder,
    ) = fixed_agent_networks_with_incorrect_sampling_setup_type
    with pytest.raises(ValueError):
        fixed_network_system.on_building_init(fixed_network_builder)

    (
        random_network_system,
        random_network_builder,
    ) = random_agent_networks_with_incorrect_sampling_setup_type
    with pytest.raises(ValueError):
        random_network_system.on_building_init(random_network_builder)

    (
        custom_network_system,
        custom_network_builder,
    ) = custom_agent_networks_with_incorrect_sampling_setup_type
    with pytest.raises(ValueError):
        custom_network_system.on_building_init(custom_network_builder)
