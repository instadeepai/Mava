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

"""Tests for config class for Jax-based Mava systems"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pytest
import reverb
from acme import specs

from mava.components.jax import Component
from mava.components.jax.building import adders
from mava.core_jax import SystemBuilder
from mava.specs import DesignSpec, MAEnvironmentSpec
from mava.systems.jax.system import System
from mava.utils.wrapper_utils import parameterized_restart

agents = {"agent_0", "agent_1", "agent_2"}
obs_first = {agent: np.array([0.0, 1.0]) for agent in agents}
default_discount = {agent: 1.0 for agent in agents}
env_restart = parameterized_restart(
    reward={agent: 0.0 for agent in agents},
    discount=default_discount,
    observation=obs_first,
)


def make_fake_env_specs() -> MAEnvironmentSpec:
    """_summary_

    Returns:
        _description_
    """
    agents = ["agent_0", "agent_1"]
    env_spec = {}
    for agent in agents:
        env_spec[agent] = specs.EnvironmentSpec(
            observations=specs.Array(shape=(10, 5), dtype=np.float32),
            actions=specs.DiscreteArray(num_values=3),
            rewards=specs.Array(shape=(), dtype=np.float32),
            discounts=specs.BoundedArray(
                shape=(), dtype=np.float32, minimum=0.0, maximum=1.0
            ),
        )
    return MAEnvironmentSpec(
        environment=None,
        specs=env_spec,
        extra_specs={"extras": specs.Array(shape=(), dtype=np.float32)},
    )


@dataclass
class MockConfig:
    pass


class MockTestSetup(Component):
    """_summary_"""

    def __init__(self, config: MockConfig = MockConfig()) -> None:
        """_summary_"""
        self.config = config

    def on_building_data_server(self, builder: SystemBuilder) -> None:
        """_summary_"""
        env_spec = make_fake_env_specs()
        builder.store.server = reverb.Server(
            [
                reverb.Table.queue(
                    name="data_server",
                    max_size=100,
                    signature=builder.store.adder_signature_fn(
                        env_spec, builder.store.sequence_length, {}
                    ),
                )
            ]
        )
        builder.store.system_data_server = reverb.Client(
            f"localhost:{builder.store.server.port}"
        )
        builder.store.unique_net_keys = ["network_0"]
        builder.store.table_network_config = {"table_0": "network_0"}

    @property
    def name(self) -> str:
        """_summary_"""
        return "setup"


@dataclass
class DistributorDefaultConfig:
    num_executors: int = 1
    nodes_on_gpu: List[str] = field(default_factory=list)
    multi_process: bool = True
    name: str = "system"


class MockDistributor(Component):
    def __init__(
        self, config: DistributorDefaultConfig = DistributorDefaultConfig()
    ) -> None:
        """Mock system distributor component.

        Args:
            config : dataclass configuration for setting component hyperparameters
        """
        self.config = config

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'.

        Returns:
            Component type name
        """
        return "distributor"


class TestSystemWithParallelSequenceAdder(System):
    def design(self) -> DesignSpec:
        """Mock system design with zero components.

        Returns:
            system callback components
        """
        executor_adder = adders.ParallelSequenceAdder
        executor_adder_priority = adders.UniformAdderPriority
        data_server_adder = adders.ParallelSequenceAdderSignature
        components = DesignSpec(
            data_server_adder=data_server_adder,
            executor_adder=executor_adder,
            executor_adder_priority=executor_adder_priority,
            setup=MockTestSetup,
            distributor=MockDistributor,
        )
        return components


@pytest.fixture
def test_system_parallel_sequence_adder() -> System:
    """Dummy system with zero components."""
    return TestSystemWithParallelSequenceAdder()


def test_adders(
    test_system_parallel_sequence_adder: System,
) -> None:
    """Test if system builder instantiates processes as expected."""
    test_system_parallel_sequence_adder.build()
    test_system_parallel_sequence_adder._builder.data_server()
    test_system_parallel_sequence_adder._builder.store.system_executor = None
    test_system_parallel_sequence_adder._builder.executor(
        executor_id="executor", data_server_client=None, parameter_server_client=None
    )
    test_system_parallel_sequence_adder._builder.store.adder.add_first(env_restart)
    test_system_parallel_sequence_adder._builder.store.server.stop()


# TODO (Kale-ab): test adder behaviour in more detail
