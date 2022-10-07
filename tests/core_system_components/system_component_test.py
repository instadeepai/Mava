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
"""Tests for Jax-based Mava system implementation."""
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Tuple

import pytest

from mava.components import Component
from mava.core_jax import SystemBuilder
from mava.specs import DesignSpec
from mava.systems.system import System


# Mock components
class MainComponent(Component):
    @staticmethod
    def name() -> str:
        """Component type name, e.g. 'dataset' or 'executor'.

        Returns:
            Component type name
        """
        return "main_component"


class SubComponent(Component):
    @staticmethod
    def name() -> str:
        """Component type name, e.g. 'dataset' or 'executor'.

        Returns:
            Component type name
        """
        return "sub_component"


@dataclass
class ComponentZeroDefaultConfig:
    param_0: int = 1
    param_1: str = "1"


class ComponentZero(MainComponent):
    def __init__(
        self, config: ComponentZeroDefaultConfig = ComponentZeroDefaultConfig()
    ) -> None:
        """Mock system component.

        Args:
            config : dataclass configuration for setting component hyperparameters
        """
        self.config = config

    def on_building_start(self, builder: SystemBuilder) -> None:
        """Dummy component function.

        Returns:
            config int plus string cast to int
        """
        builder.store.int_plus_str = self.config.param_0 + int(self.config.param_1)


@dataclass
class ComponentOneDefaultConfig:
    param_2: float = 1.2
    param_3: bool = True


class ComponentOne(SubComponent):
    def __init__(
        self, config: ComponentOneDefaultConfig = ComponentOneDefaultConfig()
    ) -> None:
        """Mock system component.

        Args:
            config : dataclass configuration for setting component hyperparameters
        """
        self.config = config

    def on_building_start(self, builder: SystemBuilder) -> None:
        """Dummy component function.

        Returns:
            float plus boolean cast as float
        """
        builder.store.float_plus_bool = self.config.param_2 + float(self.config.param_3)


@dataclass
class ComponentTwoDefaultConfig:
    param_4: str = "2"
    param_5: bool = True


class ComponentTwo(MainComponent):
    def __init__(
        self, config: ComponentTwoDefaultConfig = ComponentTwoDefaultConfig()
    ) -> None:
        """Mock system component.

        Args:
            config : dataclass configuration for setting component hyperparameters
        """
        self.config = config

    def on_building_start(self, builder: SystemBuilder) -> None:
        """Dummy component function.

        Returns:
            string cast as int plus boolean cast as in
        """
        builder.store.str_plus_bool = int(self.config.param_4) + int(
            self.config.param_5
        )


@dataclass
class DistributorDefaultConfig:
    num_executors: int = 1
    nodes_on_gpu: List[str] = field(default_factory=list)
    multi_process: bool = True
    name: str = "system"


# Instantiate DistributorDefaultConfig to have access
# to nodes_on_gpu attribute
distributor_default_config = DistributorDefaultConfig()


class MockDistributorComponent(Component):
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


class TestSystemWithZeroComponents(System):
    __test__ = False

    @staticmethod
    def design() -> Tuple[DesignSpec, Dict]:
        """Mock system design with zero components.

        Returns:
            system callback components
        """
        components = DesignSpec(distributor=MockDistributorComponent)
        return components, {}


class TestSystemWithOneComponent(System):
    __test__ = False

    @staticmethod
    def design() -> Tuple[DesignSpec, Dict]:
        """Mock system design with one component.

        Returns:
            system callback components
        """
        components = DesignSpec(
            main_component=ComponentZero, distributor=MockDistributorComponent
        )
        return components, {}


class TestSystemWithTwoComponents(System):
    __test__ = False

    @staticmethod
    def design() -> Tuple[DesignSpec, Dict]:
        """Mock system design with two components.

        Returns:
            system callback components
        """
        components = DesignSpec(
            main_component=ComponentZero,
            sub_component=ComponentOne,
            distributor=MockDistributorComponent,
        )
        return components, {}


# Test fixtures
@pytest.fixture
def system_with_zero_components() -> System:
    """Dummy system with zero components.

    Returns:
        mock system
    """
    return TestSystemWithZeroComponents()


@pytest.fixture
def system_with_one_component() -> System:
    """Dummy system with one component.

    Returns:
        mock system
    """
    return TestSystemWithOneComponent()


@pytest.fixture
def system_with_two_components() -> System:
    """Dummy system with two components.

    Returns:
        mock system
    """
    return TestSystemWithTwoComponents()


# Tests
def test_system_launch_with_build(
    system_with_two_components: System,
) -> None:
    """Test if system can launch having had changed (build) the default \
        config.

    Args:
        system_with_two_components : mock system
    """

    system_with_two_components.build(param_0=2, param_3=False)
    assert system_with_two_components._builder.store.int_plus_str == 3
    assert system_with_two_components._builder.store.float_plus_bool == 1.2
    system_with_two_components.launch()
    assert system_with_two_components._builder.store.global_config == SimpleNamespace(
        multi_process=DistributorDefaultConfig.multi_process,
        name=DistributorDefaultConfig.name,
        nodes_on_gpu=distributor_default_config.nodes_on_gpu,
        num_executors=DistributorDefaultConfig.num_executors,
        param_0=2,
        param_1=ComponentZeroDefaultConfig.param_1,
        param_2=ComponentOneDefaultConfig.param_2,
        param_3=False,
    )


def test_launch_when_not_built(
    system_with_two_components: System,
) -> None:
    """Test that exception is thrown when launch is called before a system\
        has been built."""

    with pytest.raises(Exception):
        system_with_two_components.launch()


def test_system_update_with_existing_component(
    system_with_two_components: System,
) -> None:
    """Test if system can update existing component.

    Args:
        system_with_two_components : mock system
    """
    system_with_two_components.update(ComponentTwo)
    system_with_two_components.build()
    assert system_with_two_components._builder.store.float_plus_bool == 2.2
    assert system_with_two_components._builder.store.str_plus_bool == 3
    assert system_with_two_components._builder.store.global_config == SimpleNamespace(
        multi_process=DistributorDefaultConfig.multi_process,
        name=DistributorDefaultConfig.name,
        nodes_on_gpu=distributor_default_config.nodes_on_gpu,
        num_executors=DistributorDefaultConfig.num_executors,
        param_2=ComponentOneDefaultConfig.param_2,
        param_3=ComponentOneDefaultConfig.param_3,
        param_4=ComponentTwoDefaultConfig.param_4,
        param_5=ComponentTwoDefaultConfig.param_5,
    )


def test_system_update_when_built(
    system_with_two_components: System,
) -> None:
    """Test that exception is raised when system is updated after being built."""
    system_with_two_components.build()

    with pytest.raises(Exception):
        system_with_two_components.update(ComponentTwo)


def test_system_update_with_non_existing_component(
    system_with_one_component: System,
) -> None:
    """Test if system raises an error when trying to update a component that has not \
        yet been added to the system.

    Args:
        system_with_one_component : mock system
    """
    with pytest.raises(Exception):
        system_with_one_component.update(ComponentOne)


def test_system_add_with_existing_component(system_with_one_component: System) -> None:
    """Test if system raises an error when trying to add a component that has already \
        been added to the system, i.e. we don't want to overwrite a component by \
        mistake.

    Args:
        system_with_one_component : mock system
    """
    with pytest.raises(Exception):
        system_with_one_component.add(ComponentTwo)


def test_system_add_with_non_existing_component(
    system_with_one_component: System,
) -> None:
    """Test if system can add a new component.

    Args:
        system_with_one_component : mock system
    """
    system_with_one_component.add(ComponentOne)
    system_with_one_component.build()
    assert system_with_one_component._builder.store.int_plus_str == 2
    assert system_with_one_component._builder.store.float_plus_bool == 2.2
    assert system_with_one_component._builder.store.global_config == SimpleNamespace(
        multi_process=DistributorDefaultConfig.multi_process,
        name=DistributorDefaultConfig.name,
        nodes_on_gpu=distributor_default_config.nodes_on_gpu,
        num_executors=DistributorDefaultConfig.num_executors,
        param_0=ComponentZeroDefaultConfig.param_0,
        param_1=ComponentZeroDefaultConfig.param_1,
        param_2=ComponentOneDefaultConfig.param_2,
        param_3=ComponentOneDefaultConfig.param_3,
    )


def test_system_update_twice(system_with_two_components: System) -> None:
    """Test if system can update a component twice.

    Args:
        system_with_two_components : mock system
    """
    system_with_two_components.update(ComponentTwo)
    system_with_two_components.update(ComponentZero)
    system_with_two_components.build()
    assert system_with_two_components._builder.store.int_plus_str == 2
    assert system_with_two_components._builder.store.float_plus_bool == 2.2
    assert system_with_two_components._builder.store.global_config == SimpleNamespace(
        multi_process=DistributorDefaultConfig.multi_process,
        name=DistributorDefaultConfig.name,
        nodes_on_gpu=distributor_default_config.nodes_on_gpu,
        num_executors=DistributorDefaultConfig.num_executors,
        param_0=ComponentZeroDefaultConfig.param_0,
        param_1=ComponentZeroDefaultConfig.param_1,
        param_2=ComponentOneDefaultConfig.param_2,
        param_3=ComponentOneDefaultConfig.param_3,
    )


def test_system_add_twice(system_with_zero_components: System) -> None:
    """Test if system can add two components.

    Args:
        system_with_zero_components : mock system
    """
    system_with_zero_components.add(ComponentOne)
    system_with_zero_components.add(ComponentTwo)
    system_with_zero_components.build()
    assert system_with_zero_components._builder.store.float_plus_bool == 2.2
    assert system_with_zero_components._builder.store.str_plus_bool == 3
    assert system_with_zero_components._builder.store.global_config == SimpleNamespace(
        multi_process=DistributorDefaultConfig.multi_process,
        name=DistributorDefaultConfig.name,
        nodes_on_gpu=distributor_default_config.nodes_on_gpu,
        num_executors=DistributorDefaultConfig.num_executors,
        param_2=ComponentOneDefaultConfig.param_2,
        param_3=ComponentOneDefaultConfig.param_3,
        param_4=ComponentTwoDefaultConfig.param_4,
        param_5=ComponentTwoDefaultConfig.param_5,
    )


def test_system_add_and_update(system_with_zero_components: System) -> None:
    """Test if system can add and then update a component.

    Args:
        system_with_zero_components : mock system
    """
    system_with_zero_components.add(ComponentZero)
    system_with_zero_components.update(ComponentTwo)
    system_with_zero_components.build()
    assert system_with_zero_components._builder.store.str_plus_bool == 3
    assert system_with_zero_components._builder.store.global_config == SimpleNamespace(
        multi_process=DistributorDefaultConfig.multi_process,
        name=DistributorDefaultConfig.name,
        nodes_on_gpu=distributor_default_config.nodes_on_gpu,
        num_executors=DistributorDefaultConfig.num_executors,
        param_4=ComponentTwoDefaultConfig.param_4,
        param_5=ComponentTwoDefaultConfig.param_5,
    )


def test_add_when_built(system_with_one_component: System) -> None:
    """Test that exception is raised when trying to add to a system that has\
        already been built."""

    system_with_one_component.build()
    with pytest.raises(Exception):
        system_with_one_component.add(ComponentOne)


def test_system_build_one_component_params(
    system_with_two_components: System,
) -> None:
    """Test if system can build a single component's parameters.

    Args:
        system_with_two_components : mock system
    """
    system_with_two_components.build(param_0=2, param_1="2")
    assert system_with_two_components._builder.store.int_plus_str == 4
    assert system_with_two_components._builder.store.float_plus_bool == 2.2
    assert system_with_two_components._builder.store.global_config == SimpleNamespace(
        multi_process=DistributorDefaultConfig.multi_process,
        name=DistributorDefaultConfig.name,
        nodes_on_gpu=distributor_default_config.nodes_on_gpu,
        num_executors=DistributorDefaultConfig.num_executors,
        param_0=2,
        param_1="2",
        param_2=ComponentOneDefaultConfig.param_2,
        param_3=ComponentOneDefaultConfig.param_3,
    )


def test_system_build_two_component_params(
    system_with_two_components: System,
) -> None:
    """Test if system can build multiple component parameters.

    Args:
        system_with_two_components : mock system
    """
    system_with_two_components.build(param_0=2, param_3=False)
    assert system_with_two_components._builder.store.int_plus_str == 3
    assert system_with_two_components._builder.store.float_plus_bool == 1.2
    assert system_with_two_components._builder.store.global_config == SimpleNamespace(
        multi_process=DistributorDefaultConfig.multi_process,
        name=DistributorDefaultConfig.name,
        nodes_on_gpu=distributor_default_config.nodes_on_gpu,
        num_executors=DistributorDefaultConfig.num_executors,
        param_0=2,
        param_1=ComponentZeroDefaultConfig.param_1,
        param_2=ComponentOneDefaultConfig.param_2,
        param_3=False,
    )
