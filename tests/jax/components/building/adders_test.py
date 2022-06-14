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
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pytest
from acme import specs

from mava.components.jax import Component, building
from mava.components.jax.building import adders
from mava.components.jax.building.base import SystemInit
from mava.components.jax.building.environments import EnvironmentSpec
from mava.specs import DesignSpec, MAEnvironmentSpec
from mava.systems.jax.system import System
from mava.utils.wrapper_utils import parameterized_restart
from tests.jax.mocks import (
    MockExecutorEnvironmentLoop,
    MockOnPolicyDataServer,
    MockReverbDistributor,
    make_fake_environment_factory,
    MockLogger,
)

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

    @staticmethod
    def name() -> str:
        """Component type name, e.g. 'dataset' or 'executor'.

        Returns:
            Component type name
        """
        return "distributor"

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return DistributorDefaultConfig


class TestSystemWithParallelSequenceAdder(System):
    def design(self) -> Tuple[DesignSpec, Dict]:
        """Mock system design with zero components.

        Returns:
            system callback components
        """
        components = DesignSpec(
            environment_spec=EnvironmentSpec,
            system_init=SystemInit,
            data_server_adder_signature=adders.ParallelSequenceAdderSignature,
            executor_adder=building.ParallelSequenceAdder,
            adder_priority=adders.UniformAdderPriority,
            data_server=MockOnPolicyDataServer,
            distributor=MockReverbDistributor,
            logger=MockLogger,
            executor_environment_loop=MockExecutorEnvironmentLoop,
        )
        return components, {}


@pytest.fixture
def test_system_parallel_sequence_adder() -> System:
    """Dummy system with zero components."""
    return TestSystemWithParallelSequenceAdder()


# TODO Fix test.
def test_adders(
    test_system_parallel_sequence_adder: System,
) -> None:
    """Test if system builder instantiates processes as expected."""
    # TODO Update once rewrite these tests.
    test_system_parallel_sequence_adder.build(
        environment_factory=make_fake_environment_factory()
    )
    test_system_parallel_sequence_adder._builder.data_server()  # to get client
    # test_system_parallel_sequence_adder._builder.store.system_executor = None
    test_system_parallel_sequence_adder._builder.executor(  # _builder.store.data_server_client assigned here
        executor_id="executor",
        data_server_client=test_system_parallel_sequence_adder._builder.store.system_build[
            0
        ],
        parameter_server_client=None,
    )
    test_system_parallel_sequence_adder._builder.store.adder.add_first(env_restart)
    # TODO test add and sample
    # TODO test padding
    # TODO test single_process distributor
    # test_system_parallel_sequence_adder._builder.store.server.stop()


# TODO (Kale-ab): test adder behaviour in more detail
