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

"""Config builder class for Mava systems"""

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


@pytest.fixture
def dummy_component_config() -> ComponentConfig:
    """_summary_

    Returns:
        _description_
    """
    return ComponentConfig(name="component", setting=5)


@pytest.fixture
def dummy_hyperparameter_config() -> HyperparameterConfig:
    """_summary_

    Returns:
        _description_
    """
    return HyperparameterConfig(param_0=2.7, param_1=3.8)


@pytest.fixture
def config() -> Config:
    """_summary_

    Returns:
        _description_
    """
    return Config()


def test_add_single_config(config: Config, dummy_component_config: type) -> None:
    """_summary_

    Args:
        config : _description_
        dummy_component_config : _description_
    """
    config.add(component=dummy_component_config)
    config.build()
    conf = config.get()

    assert conf.name == "component"
    assert conf.setting == 5


def test_add_multiple_configs(
    config: Config, dummy_component_config: type, dummy_hyperparameter_config: type
) -> None:
    """_summary_

    Args:
        config : _description_
        dummy_component_config : _description_
        dummy_hyperparameter_config : _description_
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


def test_add_configs_twice(
    config: Config, dummy_component_config: type, dummy_hyperparameter_config: type
) -> None:
    """_summary_

    Args:
        config : _description_
        dummy_component_config : _description_
        dummy_hyperparameter_config : _description_
    """
    config.add(component=dummy_component_config)
    config.add(hyperparameter=dummy_hyperparameter_config)
    config.build()
    conf = config.get()

    assert conf.name == "component"
    assert conf.setting == 5
    assert conf.param_0 == 2.7
    assert conf.param_1 == 3.8


def test_update_existing_parameter_on_the_fly(
    config: Config, dummy_component_config: type
) -> None:
    """_summary_

    Args:
        config : _description_
        dummy_component_config : _description_
    """

    # add component dataclasses and build config
    config.add(component=dummy_component_config)
    config.build()

    # update config on the fly
    config.update(name="new_component_name")
    conf = config.get()

    assert conf.name == "new_component_name"
    assert conf.setting == 5


# TODO (Arnu): add more specific tests

# def test_update_new_parameter_on_the_fly(config, dummy_component_config) -> None:
#     # add component dataclasses and build config
#     config.add(component=dummy_component_config)
#     config.build()

#     # update config on the fly
#     config.update(new_param="new_value")
#     conf = config.get()
