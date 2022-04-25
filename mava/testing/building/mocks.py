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
from typing import Callable, Dict, List, Optional

import dm_env
import jax
from acme import types

from mava import specs
from mava.callbacks import Callback
from mava.core_jax import SystemBuilder
from mava.environment_loop import ParallelEnvironmentLoop


# Mock components to feed to the builder
@dataclass
class MockAdderSignatureConfig:
    adder_signature_param: int = 1


class MockAdderSignature(Callback):
    def __init__(
        self,
        config: MockAdderSignatureConfig = MockAdderSignatureConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_data_server_adder_signature(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.store.adder_signature_fn = self.config.adder_signature_param

    @staticmethod
    def name() -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "adder_signature"


@dataclass
class MockAdderConfig:
    adder_param: float = 2.7


class MockAdderClass:
    def __init__(
        self,
    ) -> None:
        """_summary_"""
        pass

    def add_first(
        self, timestep: dm_env.TimeStep, extras: Dict[str, types.NestedArray] = {}
    ) -> None:
        """_summary_

        Args:
            timestep : _description_
            extras : _description_.
        """
        pass

    def add(
        self,
        actions: Dict[str, types.NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """_summary_

        Args:
            actions : _description_
            next_timestep : _description_
            next_extras : _description_.
        """
        pass


class MockAdder(Callback):
    def __init__(self, config: MockAdderConfig = MockAdderConfig()) -> None:
        """Mock system component."""
        self.config = config

    def on_building_executor_adder(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.store.adder = MockAdderClass()

    @staticmethod
    def name() -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "adder"


@dataclass
class MockDataServerConfig:
    data_server_param: int = 2


class MockDataServer(Callback):
    def __init__(
        self,
        config: MockDataServerConfig = MockDataServerConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_data_server(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.store.data_tables = self.config.data_server_param

    @staticmethod
    def name() -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "data_server"


@dataclass
class MockParameterServerConfig:
    parameter_server_param: int = 2


class MockParameterServer(Callback):
    def __init__(
        self,
        config: MockParameterServerConfig = MockParameterServerConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_parameter_server(self, builder: SystemBuilder) -> None:
        """_summary_"""

    @staticmethod
    def name() -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "parameter_server"


@dataclass
class MockLoggerConfig:
    logger_param_0: int = 1
    logger_param_1: int = 2


class MockLogerClass:
    def __init__(self) -> None:
        """Mock logger component."""
        pass

    def write(self, data: Dict) -> None:
        """_summary_"""
        pass


class MockLogger(Callback):
    def __init__(
        self,
        config: MockLoggerConfig = MockLoggerConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_executor_logger(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.store.executor_logger = MockLogerClass()

    def on_building_trainer_logger(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.store.trainer_logger = MockLogerClass()

    @staticmethod
    def name() -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "logger"


@dataclass
class MockExecutorParameterClientConfig:
    executor_parameter_client_param_0: int = 1
    executor_parameter_client_param_1: str = "param"


class MockExecutorParameterClient(Callback):
    def __init__(
        self,
        config: MockExecutorParameterClientConfig = MockExecutorParameterClientConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_executor_parameter_client(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.store.executor_parameter_client = None

    @staticmethod
    def name() -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "executor_parameter_client"


@dataclass
class MockTrainerParameterClientConfig:
    trainer_parameter_client_param_0: int = 1
    trainer_parameter_client_param_1: str = "param"


class MockTrainerParameterClient(Callback):
    def __init__(
        self,
        config: MockTrainerParameterClientConfig = MockTrainerParameterClientConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_trainer_parameter_client(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.store.trainer_parameter_client = None

    @staticmethod
    def name() -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "trainer_parameter_client"


@dataclass
class MockExecutorDefaultConfig:
    executor_param: int = 1


class MockExecutor(Callback):
    def __init__(
        self,
        config: MockExecutorDefaultConfig = MockExecutorDefaultConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_init(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.store.agent_net_keys = {
            "agent_0": "network_agent",
            "agent_1": "network_agent",
            "agent_2": "network_agent",
        }

    @staticmethod
    def name() -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "executor"


@dataclass
class MockExecutorEnvironmentLoopConfig:
    environment_factory: str = "param"
    should_update: bool = True


class MockExecutorEnvironmentLoop(Callback):
    def __init__(
        self,
        config: MockExecutorEnvironmentLoopConfig = MockExecutorEnvironmentLoopConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_init_start(self, builder: SystemBuilder) -> None:
        """[summary]"""
        if not isinstance(self.config.environment_factory, str):
            builder.store.executor_environment = self.config.environment_factory(
                evaluation=False
            )  # type: ignore
            builder.store.environment_spec = specs.MAEnvironmentSpec(
                builder.store.executor_environment
            )
        else:
            # Just assign a None for the environment for testing.
            builder.store.executor_environment = None

    def on_building_executor_environment(self, builder: SystemBuilder) -> None:
        """_summary_"""

    def on_building_executor_environment_loop(self, builder: SystemBuilder) -> None:
        """_summary_"""

        executor_environment_loop = ParallelEnvironmentLoop(
            environment=builder.store.executor_environment,
            executor=builder.store.executor,
            logger=builder.store.executor_logger,
            should_update=self.config.should_update,
        )
        del builder.store.executor_logger

        if builder.store.executor_id == "evaluator":
            builder.store.system_evaluator = executor_environment_loop
        else:
            builder.store.system_executor = executor_environment_loop

    @staticmethod
    def name() -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "executor_environment_loop"


@dataclass
class MockNetworksConfig:
    network_factory: Optional[Callable[[str], dm_env.Environment]] = None
    shared_weights: bool = True
    seed: int = 1234


class MockNetworks(Callback):
    def __init__(
        self,
        config: MockNetworksConfig = MockNetworksConfig(),
    ):
        """[summary]"""
        self.config = config

    def on_building_init_start(self, builder: SystemBuilder) -> None:
        """Summary"""
        # Set the shared weights
        builder.store.shared_networks = self.config.shared_weights

        # Setup the jax key for network initialisations
        builder.store.key = jax.random.PRNGKey(self.config.seed)

        # Build network function here
        network_key, builder.store.key = jax.random.split(builder.store.key)
        builder.store.network_factory = (
            lambda: self.config.network_factory(  # type: ignore
                environment_spec=builder.store.environment_spec,
                agent_net_keys=builder.store.agent_net_keys,
                rng_key=network_key,
            )
        )

    @staticmethod
    def name() -> str:
        """_summary_"""
        return "networks"


@dataclass
class MockTrainerDatasetConfig:
    trainer_dataset_param: int = 5


class MockTrainerDataset(Callback):
    def __init__(
        self,
        config: MockTrainerDatasetConfig = MockTrainerDatasetConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_init_end(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.store.net_spec_keys = {"network_agent": "agent_0"}

    def on_building_trainer_dataset(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.store.dataset = self.config.trainer_dataset_param

    @staticmethod
    def name() -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "trainer_dataset"


@dataclass
class MockTrainerConfig:
    trainer_param_0: int = 2
    trainer_param_1: str = "train"


class MockTrainer(Callback):
    def __init__(
        self,
        config: MockTrainerConfig = MockTrainerConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_init_end(self, builder: SystemBuilder) -> None:
        """TODO: Add description here."""
        builder.store.table_network_config = {
            "trainer": ["network_agent", "network_agent", "network_agent"]
        }
        builder.store.trainer_networks = {"trainer": ["network_agent"]}

    def on_building_trainer(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.store.system_trainer = (
            builder.store.trainer_id,
            builder.store.trainer_logger,
            builder.store.dataset,
            builder.store.trainer_parameter_client,
        )

    @staticmethod
    def name() -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "trainer"


@dataclass
class DistributorConfig:
    num_executors: int = 1
    nodes_on_gpu: List[str] = field(default_factory=list)
    multi_process: bool = True
    name: str = "system"


class MockDistributor(Callback):
    def __init__(self, config: DistributorConfig = DistributorConfig()) -> None:
        """Mock system distributor component.

        Args:
            config : dataclass configuration for setting component hyperparameters
        """
        self.config = config

    def on_building_program_nodes(self, builder: SystemBuilder) -> None:
        """_summary_"""
        data_server = builder.data_server()
        parameter_server = builder.parameter_server()

        trainer = builder.trainer(
            trainer_id="trainer",
            data_server_client=data_server,
            parameter_server_client=parameter_server,
        )
        executor = builder.executor(
            executor_id="executor",
            data_server_client=data_server,
            parameter_server_client=parameter_server,
        )
        evaluator = builder.executor(
            executor_id="evaluator",
            data_server_client=None,
            parameter_server_client=parameter_server,
        )
        builder.store.system_build = (
            data_server,
            parameter_server,
            executor,
            evaluator,
            trainer,
        )

    @staticmethod
    def name() -> str:
        """Component type name, e.g. 'dataset' or 'executor'.

        Returns:
            Component type name
        """
        return "distributor"
