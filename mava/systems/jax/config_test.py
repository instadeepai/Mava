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

from dataclasses import dataclass

import pytest

from mava.systems.jax import Config


@dataclass
class ComponentConfig:
    name: str
    setting: int


@dataclass
class HyperparameterConfig:
    param_0: float
    param_1: float


@dataclass
class SameParameterNameConfig:
    param_0: int
    param_2: str


@pytest.fixture
def dummy_component_config() -> ComponentConfig:
    """Dummy config dataclass for a component.

    Returns:
        config dataclass
    """
    return ComponentConfig(name="component", setting=5)


@pytest.fixture
def dummy_hyperparameter_config() -> HyperparameterConfig:
    """Dummy config dataclass for component hyperparameters.

    Returns:
        config dataclass
    """
    return HyperparameterConfig(param_0=2.7, param_1=3.8)


@pytest.fixture
def config() -> Config:
    """Config instance.

    Returns:
        instantiation of a Mava Config class
    """
    return Config()


def test_add_single_config(config: Config, dummy_component_config: type) -> None:
    """Test adding a single config.

    Args:
        config : Mava config
        dummy_component_config : component config dataclass
    """
    config.add(component=dummy_component_config)
    config.build()
    conf = config.get()

    assert conf.name == "component"
    assert conf.setting == 5


def test_add_multiple_configs(
    config: Config, dummy_component_config: type, dummy_hyperparameter_config: type
) -> None:
    """Test adding multiple configs at the same time.

    Args:
        config : Mava config
        dummy_component_config : component config dataclass
        dummy_hyperparameter_config : component config dataclass of hyperparameters
    """
    config.add(
        component=dummy_component_config, hyperparameter=dummy_hyperparameter_config
    )
    config.build()
    conf = config.get()

    assert conf.name == "component"
    assert conf.setting == 5
    assert conf.param_0 == 2.7
    assert conf.param_1 == 3.8


def test_add_config_twice(
    config: Config, dummy_component_config: type, dummy_hyperparameter_config: type
) -> None:
    """Test add two configs, one after the other.

    Args:
        config : Mava config
        dummy_component_config : component config dataclass
        dummy_hyperparameter_config : component config dataclass of hyperparameters
    """
    config.add(component=dummy_component_config)
    config.add(hyperparameter=dummy_hyperparameter_config)
    config.build()
    conf = config.get()

    assert conf.name == "component"
    assert conf.setting == 5
    assert conf.param_0 == 2.7
    assert conf.param_1 == 3.8


def test_update_config(
    config: Config, dummy_component_config: type, dummy_hyperparameter_config: type
) -> None:
    """Test add two configs, one after the other.

    Args:
        config : Mava config
        dummy_component_config : component config dataclass
        dummy_hyperparameter_config : component config dataclass of hyperparameters
    """
    config.add(component=dummy_component_config)
    config.update(component=dummy_hyperparameter_config)
    config.build()
    conf = config.get()

    assert conf.param_0 == 2.7
    assert conf.param_1 == 3.8
    assert not hasattr(config, "name")
    assert not hasattr(config, "setting")


def test_update_config_twice(
    config: Config, dummy_component_config: type, dummy_hyperparameter_config: type
) -> None:
    """Test add two configs, one after the other.

    Args:
        config : Mava config
        dummy_component_config : component config dataclass
        dummy_hyperparameter_config : component config dataclass of hyperparameters
    """
    config.add(component=dummy_component_config)
    config.update(component=dummy_hyperparameter_config)
    config.update(component=dummy_component_config)
    config.build()
    conf = config.get()

    assert conf.name == "component"
    assert conf.setting == 5
    assert not hasattr(config, "param_0")
    assert not hasattr(config, "param_1")


def test_set_existing_parameter_on_the_fly(
    config: Config, dummy_component_config: type
) -> None:
    """Test updating a hyperparameter on the fly after the config has been built.

    Args:
        config : Mava config
        dummy_component_config : component config dataclass
    """

    # add component dataclasses and build config
    config.add(component=dummy_component_config)
    config.build()

    # set config parameters on the fly
    config.set_parameters(name="new_component_name")
    conf = config.get()

    assert conf.name == "new_component_name"
    assert conf.setting == 5


def test_set_before_build_exception(
    config: Config, dummy_component_config: type
) -> None:
    """Test that exception is thrown if it is attempted to set a hyperparameter \
        before the config has been built.

    Args:
        config : Mava config
        dummy_component_config : component config dataclass
    """

    with pytest.raises(Exception):
        # add component dataclasses and build config
        config.add(component=dummy_component_config)

        # Try setting parameters without having built first
        config.set_parameters(name="new_component_name")


def test_get_before_build_exception(
    config: Config, dummy_component_config: type
) -> None:
    """Test that exception is thrown if it is attempted to call .get() \
        before the config has been built.

    Args:
        config : Mava config
        dummy_component_config : component config dataclass
    """

    with pytest.raises(Exception):
        # add component dataclasses and build config
        config.add(component=dummy_component_config)

        # Try getting without having built first
        config.get()


def test_parameter_setting_that_does_not_exist_exception(
    config: Config, dummy_component_config: type
) -> None:
    """Test that exception is thrown if it is attempted to set a hyperparameter \
        that does not exist.

    Args:
        config : Mava config
        dummy_component_config : component config dataclass
    """

    with pytest.raises(Exception):
        # add component dataclasses and build config
        config.add(component=dummy_component_config)
        config.build()

        # Try setting a parameter that does not exist
        config.set_parameters(unknown_param="new_value")


def test_accidental_parameter_override_with_add_exception(
    config: Config, dummy_hyperparameter_config: type
) -> None:
    """Test that exception is thrown when two component config dataclasses share the \
        same name for a specific hyperparameter when adding a new config.

    Args:
        config : Mava config
        dummy_hyperparameter_config : component config dataclass of hyperparameters
    """

    with pytest.raises(Exception):
        # add component dataclasses and build config
        config.add(hyperparameter=dummy_hyperparameter_config)

        # add new component dataclass with a parameter of the same name
        # as an already existing component parameter name
        other_hyperparamter_config = SameParameterNameConfig(param_0=2, param_2="param")
        config.add(other_hyperparameter=other_hyperparamter_config)


def test_accidental_parameter_override_with_update_exception(
    config: Config, dummy_component_config: type, dummy_hyperparameter_config: type
) -> None:
    """Test that exception is thrown when two component config dataclasses share the \
        same name for a specific hyperparameter when updating an existing config.

    Args:
        config : Mava config
        dummy_component_config : component config dataclass
        dummy_hyperparameter_config : component config dataclass of hyperparameters
    """

    with pytest.raises(Exception):
        # add component dataclasses and build config
        config.add(component_0=dummy_component_config)
        config.add(component_1=dummy_hyperparameter_config)

        # add new component dataclass with a parameter of the same name
        # as an already existing component parameter name
        other_hyperparameter_config = SameParameterNameConfig(
            param_0=2, param_2="param"
        )
        config.update(component_0=other_hyperparameter_config)
