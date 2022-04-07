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
import functools
from dataclasses import dataclass, field
from typing import List

import acme
import optax
import pytest

from mava.components.jax import Component
from mava.components.jax.building.adders import (
    ParallelSequenceAdder,
    ParallelSequenceAdderSignature,
)
from mava.components.jax.building.data_server import OnPolicyDataServer
from mava.components.jax.building.datasets import TrajectoryDataset
from mava.components.jax.building.distributor import Distributor
from mava.components.jax.building.environments import ParallelExecutorEnvironmentLoop
from mava.components.jax.building.loggers import Logger
from mava.components.jax.building.networks import DefaultNetworks
from mava.components.jax.building.parameter_client import (
    ExecutorParameterClient,
    TrainerParameterClient,
)
from mava.components.jax.updating.parameter_server import DefaultParameterServer
from mava.core_jax import SystemBuilder
from mava.specs import DesignSpec
from mava.systems.jax import mappo
from mava.systems.jax.mappo import EXECUTOR_SPEC, TRAINER_SPEC
from mava.systems.jax.mappo.components import ExtrasLogProbSpec
from mava.systems.jax.system import System
from mava.utils.environments import debugging_utils
from mava.utils.loggers import logger_utils


# Mock components
class MainComponent(Component):
    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'.

        Returns:
            Component type name
        """
        return "main_component"


class SubComponent(Component):
    @property
    def name(self) -> str:
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


class MockDistributorComponent(Component):
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


class TestSystemWithZeroComponents(System):
    def design(self) -> DesignSpec:
        """Mock system design with zero components.

        Returns:
            system callback components
        """
        components = DesignSpec(distributor=MockDistributorComponent)
        return components


class TestSystemWithOneComponent(System):
    def design(self) -> DesignSpec:
        """Mock system design with one component.

        Returns:
            system callback components
        """
        components = DesignSpec(
            main_component=ComponentZero, distributor=MockDistributorComponent
        )
        return components


class TestSystemWithTwoComponents(System):
    def design(self) -> DesignSpec:
        """Mock system design with two components.

        Returns:
            system callback components
        """
        components = DesignSpec(
            main_component=ComponentZero,
            sub_component=ComponentOne,
            distributor=MockDistributorComponent,
        )
        return components


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
    """Test if system can launch having had changed (buildd) the default \
        config.

    Args:
        system_with_two_components : mock system
    """
    system_with_two_components.build(param_0=2, param_3=False)
    assert system_with_two_components._builder.store.int_plus_str == 3
    assert system_with_two_components._builder.store.float_plus_bool == 1.2
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


def test_system_add_and_update(system_with_zero_components: System) -> None:
    """Test if system can add and then update a component.

    Args:
        system_with_zero_components : mock system
    """
    system_with_zero_components.add(ComponentZero)
    system_with_zero_components.update(ComponentTwo)
    system_with_zero_components.build()
    assert system_with_zero_components._builder.store.str_plus_bool == 3


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


#########################################################################
# Full system integration test.
class TestFullSystem(System):
    def design(self) -> DesignSpec:
        """Mock system design with zero components.

        Returns:
            system callback components
        """
        executor = EXECUTOR_SPEC.get()
        trainer = TRAINER_SPEC.get()
        components = DesignSpec(
            data_server=OnPolicyDataServer,
            data_server_adder_signature=ParallelSequenceAdderSignature,
            extras_spec=ExtrasLogProbSpec,
            parameter_server=DefaultParameterServer,
            executor_parameter_client=ExecutorParameterClient,
            **executor,
            executor_environment_loop=ParallelExecutorEnvironmentLoop,
            executor_adder=ParallelSequenceAdder,
            networks=DefaultNetworks,
            **trainer,
            distributor=Distributor,
            trainer_parameter_client=TrainerParameterClient,
            trainer_dataset=TrajectoryDataset,
            logger=Logger,
        )
        return components


@pytest.fixture
def test_full_system() -> System:
    """Add description here."""
    return TestFullSystem()


def test_except_trainer(
    test_full_system: System,
) -> None:
    """Test if the parameter server instantiates processes as expected."""

    # Environment.
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name="simple_spread",
        action_space="discrete",
    )

    # Networks.
    network_factory = mappo.make_default_networks

    # Checkpointer appends "Checkpoints" to checkpoint_dir.
    base_dir = "~/mava"
    mava_id = "12345"
    checkpoint_subpath = f"{base_dir}/{mava_id}"

    # Log every [log_every] seconds.
    log_every = 10
    logger_factory = functools.partial(
        logger_utils.make_logger,
        directory=base_dir,
        to_terminal=True,
        to_tensorboard=True,
        time_stamp=mava_id,
        time_delta=log_every,
    )

    # Optimizer.
    optimizer = optax.chain(
        optax.clip_by_global_norm(40.0), optax.scale_by_adam(), optax.scale(-1e-4)
    )

    # Build the system
    test_full_system.build(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        checkpoint_subpath=checkpoint_subpath,
        optimizer=optimizer,
        executor_parameter_update_period=20,
        multi_process=False,
        run_evaluator=True,
        num_executors=1,
        use_next_extras=False,
        sample_batch_size=2,
    )

    (
        data_server,
        parameter_server,
        executor,
        evaluator,
        trainer,
    ) = test_full_system._builder.store.system_build

    assert isinstance(executor, acme.core.Worker)

    # Step the executor
    executor.run_episode()

    # Step the trainer
    trainer.step()
