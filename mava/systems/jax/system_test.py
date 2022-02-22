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
from typing import List

import pytest

# TODO: figure out how to solve: Module "mava.systems.jax.system"
# has no attribute "System" error
from mava.systems.jax.system import System  # type: ignore


# test components
@dataclass
class ComponentZeroDefaultConfig:
    param_0: int = 1
    param_1: str = "one"


class ComponentZero:
    def __init__(
        self, config: ComponentZeroDefaultConfig = ComponentZeroDefaultConfig()
    ) -> None:
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def dummy_int_plus_str(self) -> int:
        """_summary_

        Returns:
            _description_
        """
        return self.config.param_0 + int(self.config.param_1)

    @property
    def name(self) -> str:
        """_summary_

        Returns:
            _description_
        """
        return "main_component"


@dataclass
class ComponentOneDefaultConfig:
    param_2: float = 1.2
    param_3: bool = True


class ComponentOne:
    def __init__(
        self, config: ComponentOneDefaultConfig = ComponentOneDefaultConfig()
    ) -> None:
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def dummy_float_plus_bool(self) -> float:
        """_summary_

        Returns:
            _description_
        """
        return self.config.param_2 + float(self.config.param_3)

    @property
    def name(self) -> str:
        """_summary_

        Returns:
            _description_
        """
        return "sub_component"


@dataclass
class ComponentTwoDefaultConfig:
    param_4: str = "one"
    param_5: bool = True


class ComponentTwo:
    def __init__(
        self, config: ComponentTwoDefaultConfig = ComponentTwoDefaultConfig()
    ) -> None:
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def dummy_str_plus_bool(self) -> int:
        """_summary_

        Returns:
            _description_
        """
        return int(self.config.param_4) + int(self.config.param_5)

    @property
    def name(self) -> str:
        """_summary_

        Returns:
            _description_
        """
        return "main_component"


@dataclass
class DistributorDefaultConfig:
    num_executors: int = 1
    nodes_on_gpu: List[str] = field(default_factory=list)
    multi_process: bool = True
    name: str = "system"


class MockDistributorComponent:
    def __init__(
        self, config: DistributorDefaultConfig = DistributorDefaultConfig()
    ) -> None:
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    @property
    def name(self) -> str:
        """_summary_

        Returns:
            _description_
        """
        return "distributor"


class TestSystemWithZeroComponents(System):
    def design(self) -> SimpleNamespace:
        """_summary_

        Returns:
            _description_
        """
        components = SimpleNamespace()
        return components


class TestSystemWithOneComponent(System):
    def design(self) -> SimpleNamespace:
        """_summary_

        Returns:
            _description_
        """
        components = SimpleNamespace(main_component=ComponentZero)
        return components


class TestSystemWithTwoComponents(System):
    def design(self) -> SimpleNamespace:
        """_summary_

        Returns:
            _description_
        """
        components = SimpleNamespace(
            main_component=ComponentZero, sub_component=ComponentOne
        )
        return components


class TestSystemWithTwoComponentsAndDistributor(System):
    def design(self) -> SimpleNamespace:
        """_summary_

        Returns:
            _description_
        """
        components = SimpleNamespace(
            main_component=ComponentZero,
            sub_component=ComponentOne,
            distributor=MockDistributorComponent,
        )
        return components


@pytest.fixture
def system_with_zero_components() -> System:
    """_summary_

    Returns:
        _description_
    """
    return TestSystemWithZeroComponents()


@pytest.fixture
def system_with_one_component() -> System:
    """_summary_

    Returns:
        _description_
    """
    return TestSystemWithOneComponent()


@pytest.fixture
def system_with_two_components() -> System:
    """_summary_

    Returns:
        _description_
    """
    return TestSystemWithTwoComponents()


@pytest.fixture
def system_with_two_components_and_distributor() -> System:
    """_summary_

    Returns:
        _description_
    """
    return TestSystemWithTwoComponentsAndDistributor()


def test_system_update_with_existing_component(
    system_with_two_components: System,
) -> None:
    """_summary_

    Args:
        system_with_two_components : _description_
    """
    system_with_two_components.update(ComponentTwo)


def test_system_update_with_non_existing_component(
    system_with_one_component: System,
) -> None:
    """_summary_

    Args:
        system_with_one_component : _description_
    """
    with pytest.raises(Exception):
        system_with_one_component.update(ComponentOne)


def test_system_add_with_existing_component(system_with_one_component: System) -> None:
    """_summary_

    Args:
        system_with_one_component : _description_
    """
    with pytest.raises(Exception):
        system_with_one_component.add(ComponentTwo)


def test_system_add_with_non_existing_component(
    system_with_one_component: System,
) -> None:
    """_summary_

    Args:
        system_with_one_component : _description_
    """
    system_with_one_component.add(ComponentOne)


def test_system_update_twice(system_with_two_components: System) -> None:
    """_summary_

    Args:
        system_with_two_components : _description_
    """
    system_with_two_components.update(ComponentTwo)
    system_with_two_components.update(ComponentZero)


def test_system_add_twice(system_with_zero_components: System) -> None:
    """_summary_

    Args:
        system_with_zero_components : _description_
    """
    system_with_zero_components.add(ComponentZero)
    system_with_zero_components.add(ComponentOne)


def test_system_add_and_update(system_with_zero_components: System) -> None:
    """_summary_

    Args:
        system_with_zero_components : _description_
    """
    system_with_zero_components.add(ComponentZero)
    system_with_zero_components.update(ComponentTwo)


def test_system_configure_one_component_params(
    system_with_two_components: System,
) -> None:
    """_summary_

    Args:
        system_with_two_components : _description_
    """
    system_with_two_components.configure(param_0=2, param_1="two")
    config = system_with_two_components.config.get()
    assert config.param_0 == 2
    assert config.param_1 == "two"
    assert config.param_2 == 1.2
    assert config.param_3 is True


def test_system_configure_two_component_params(
    system_with_two_components: System,
) -> None:
    """_summary_

    Args:
        system_with_two_components : _description_
    """
    system_with_two_components.configure(param_0=2, param_3=False)
    config = system_with_two_components.config.get()
    assert config.param_0 == 2
    assert config.param_1 == "one"
    assert config.param_2 == 1.2
    assert config.param_3 is False


def test_system_launch_without_configure(
    system_with_two_components_and_distributor: System,
) -> None:
    """_summary_

    Args:
        system_with_two_components_and_distributor : _description_
    """
    system_with_two_components_and_distributor.launch(
        num_executors=1, nodes_on_gpu=["process"]
    )
