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

import abc
import copy
import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import dm_env
import jax
import numpy as np
import reverb
from acme import specs as acme_specs
from acme import types
from reverb import rate_limiters, reverb_types

from mava import specs
from mava.components.jax import Component
from mava.components.jax.building.data_server import (
    OffPolicyDataServerConfig,
    OnPolicyDataServerConfig,
)
from mava.core_jax import SystemBuilder
from mava.environment_loop import ParallelEnvironmentLoop
from mava.specs import DesignSpec, MAEnvironmentSpec
from mava.systems.jax.system import System
from mava.utils.builder_utils import convert_specs
from tests.tf.enums import EnvType, MockedEnvironments
from tests.tf.mocks import (
    ParallelMAContinuousEnvironment,
    ParallelMADiscreteEnvironment,
    SequentialMAContinuousEnvironment,
    SequentialMADiscreteEnvironment,
)

# Mock components to feed to the builder


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


class MockAdder(Component):
    def __init__(self, config: MockAdderConfig = MockAdderConfig()) -> None:
        """Mock system component."""
        self.config = config

    def on_building_executor_adder(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.store.adder = MockAdderClass()

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "executor_adder"


def make_fake_env_specs() -> MAEnvironmentSpec:
    """_summary_

    Returns:
        _description_
    """
    agents = ["agent_0", "agent_1"]
    env_spec = {}
    for agent in agents:
        env_spec[agent] = acme_specs.EnvironmentSpec(
            observations=acme_specs.Array(shape=(10, 5), dtype=np.float32),
            actions=acme_specs.DiscreteArray(num_values=3),
            rewards=acme_specs.Array(shape=(), dtype=np.float32),
            discounts=acme_specs.BoundedArray(
                shape=(), dtype=np.float32, minimum=0.0, maximum=1.0
            ),
        )
    return MAEnvironmentSpec(
        environment=None,
        agent_environment_specs=env_spec,
        extras_specs={"extras": acme_specs.Array(shape=(), dtype=np.float32)},
    )


def make_fake_env(
    env_name: MockedEnvironments = MockedEnvironments.Mocked_Dicrete,
    env_type: EnvType = EnvType.Parallel,
    evaluation: bool = False,
) -> Any:
    """Func that creates a fake env.

    Args:
        env_name : env name.
        env_type : type of env.
        evaluation: whether env is used for eval or not.
            Not sure we should use this in spec.

    Raises:
        Exception: no matching env.

    Returns:
        mock env.
    """
    del evaluation
    if env_name is MockedEnvironments.Mocked_Dicrete:
        if env_type == EnvType.Parallel:
            env = ParallelMADiscreteEnvironment(
                num_actions=18,
                num_observations=2,
                obs_shape=(84, 84, 4),
                obs_dtype=np.float32,
                episode_length=10,
            )
        elif env_type == EnvType.Sequential:
            env = SequentialMADiscreteEnvironment(
                num_actions=18,
                num_observations=2,
                obs_shape=(84, 84, 4),
                obs_dtype=np.float32,
                episode_length=10,
            )
    elif env_name is MockedEnvironments.Mocked_Continous:
        if env_type == EnvType.Parallel:
            env = ParallelMAContinuousEnvironment(
                action_dim=2,
                observation_dim=2,
                bounded=True,
                episode_length=10,
            )
        elif env_type == EnvType.Sequential:
            env = SequentialMAContinuousEnvironment(
                action_dim=2,
                observation_dim=2,
                bounded=True,
                episode_length=10,
            )

    if env is None:
        raise Exception("Env_spec is not valid.")

    return env


def make_fake_environment_factory(
    env_name: MockedEnvironments = MockedEnvironments.Mocked_Dicrete,
    env_type: EnvType = EnvType.Parallel,
) -> Any:
    """Returns a mock env factory.

    Args:
        env_name : env name.
        env_type : env type.

    Returns:
        a mocked env factory.
    """
    return functools.partial(
        make_fake_env,
        env_name=env_name,
        env_type=env_type,
    )


def mock_table(
    name: str = "mock_table",
    sampler: reverb_types.SelectorType = reverb.selectors.Uniform(),
    remover: reverb_types.SelectorType = reverb.selectors.Fifo(),
    max_size: int = 100,
    rate_limiter: rate_limiters.RateLimiter = reverb.rate_limiters.MinSize(1),
    max_times_sampled: int = 0,
    signature: Any = None,
) -> reverb.Table:
    """Func returns mock table used in testing.

    Args:
        name : table name.
        sampler : reverb sampler.
        remover : reverb remover.
        max_size : max size of table.
        rate_limiter : rate limiter.
        max_times_sampled : max times sampled.
        signature : signature.

    Returns:
        mock reverb table.
    """
    return reverb.Table(
        name=name,
        sampler=sampler,
        remover=remover,
        max_size=max_size,
        rate_limiter=rate_limiter,
        signature=signature,
        max_times_sampled=max_times_sampled,
    )


def mock_queue(
    name: str = "mock_table",
    max_queue_size: int = 100,
    signature: Any = None,
) -> reverb.Table:
    """Func returns mock queue.

    Args:
        name : table name.
        max_queue_size : max queue size.
        signature : signature.

    Returns:
        mock queue.
    """
    return reverb.Table.queue(name=name, max_size=max_queue_size, signature=signature)


@dataclass
class MockDataServerConfig:
    pass


class MockDataServer(Component):
    def __init__(self, config: MockDataServerConfig = MockDataServerConfig()) -> None:
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def _create_table_per_trainer(self, builder: SystemBuilder) -> List[reverb.Table]:
        """Create table for each trainer"""
        builder.store.table_network_config = {"table_0": "network_0"}
        data_tables = []
        extras_spec: dict = {}
        for table_key in builder.store.table_network_config.keys():
            num_networks = len(builder.store.table_network_config[table_key])
            env_spec = copy.deepcopy(builder.store.ma_environment_spec)
            env_spec.set_agent_environment_specs(
                convert_specs(
                    builder.store.agent_net_keys,
                    env_spec.get_agent_environment_specs(),
                    num_networks,
                )
            )
            table = self.table(table_key, env_spec, extras_spec, builder)
            data_tables.append(table)
        return data_tables

    @abc.abstractmethod
    def table(
        self,
        table_key: str,
        environment_spec: specs.MAEnvironmentSpec,
        extras_spec: Dict[str, Any],
        builder: SystemBuilder,
    ) -> reverb.Table:
        """_summary_"""

    def on_building_data_server(self, builder: SystemBuilder) -> None:
        """[summary]"""
        builder.store.data_tables = self._create_table_per_trainer(builder)

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "data_server"


class MockOnPolicyDataServer(MockDataServer):
    def __init__(
        self, config: OnPolicyDataServerConfig = OnPolicyDataServerConfig()
    ) -> None:
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def _create_table_per_trainer(self, builder: SystemBuilder) -> List[reverb.Table]:
        """Create table for each trainer"""
        builder.store.table_network_config = {"table_0": "network_0"}
        data_tables = []
        extras_spec: dict = {}
        for table_key in builder.store.table_network_config.keys():
            num_networks = len(builder.store.table_network_config[table_key])
            env_spec = copy.deepcopy(builder.store.ma_environment_spec)
            env_spec.set_agent_environment_specs(
                convert_specs(
                    builder.store.agent_net_keys,
                    env_spec.get_agent_environment_specs(),
                    num_networks,
                )
            )
            table = self.table(table_key, env_spec, extras_spec, builder)
            data_tables.append(table)
        return data_tables

    def table(
        self,
        table_key: str,
        environment_spec: specs.MAEnvironmentSpec,
        extras_spec: Dict[str, Any],
        builder: SystemBuilder,
    ) -> reverb.Table:
        """Func returns mock table used in testing.

        Args:
            table_key: key for specific table.
            environment_spec: env spec.
            extras_spec: extras spec.
            builder: builder used for building this component.

        Returns:
            mock reverb table.
        """
        if builder.store.__dict__.get("sequence_length"):
            signature = builder.store.adder_signature_fn(
                environment_spec, builder.store.sequence_length, extras_spec
            )
        else:
            signature = builder.store.adder_signature_fn(environment_spec, extras_spec)
        return mock_queue(
            name=table_key,
            max_queue_size=self.config.max_queue_size,
            signature=signature,
        )


class MockOffPolicyDataServer(MockDataServer):
    def __init__(
        self, config: OffPolicyDataServerConfig = OffPolicyDataServerConfig()
    ) -> None:
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def _create_table_per_trainer(self, builder: SystemBuilder) -> List[reverb.Table]:
        """Create table for each trainer"""
        builder.store.table_network_config = {"table_0": "network_0"}
        data_tables = []
        extras_spec: dict = {}
        for table_key in builder.store.table_network_config.keys():
            num_networks = len(builder.store.table_network_config[table_key])
            env_spec = copy.deepcopy(builder.store.ma_environment_spec)
            env_spec.set_agent_environment_specs(
                convert_specs(
                    builder.store.agent_net_keys,
                    env_spec.get_agent_environment_specs(),
                    num_networks,
                )
            )
            table = self.table(table_key, env_spec, extras_spec, builder)
            data_tables.append(table)
        return data_tables

    def table(
        self,
        table_key: str,
        environment_spec: specs.MAEnvironmentSpec,
        extras_spec: Dict[str, Any],
        builder: SystemBuilder,
    ) -> reverb.Table:
        """Func returns mock table used in testing.

        Args:
            table_key: key for specific table.
            environment_spec: env spec.
            extras_spec: extras spec.
            builder: builder used for building this component.

        Returns:
            mock reverb table.
        """
        return mock_table(
            name=table_key,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=self.config.max_size,
            max_times_sampled=self.config.max_times_sampled,
            rate_limiter=builder.store.rate_limiter_fn(),
            signature=builder.store.adder_signature_fn(environment_spec, extras_spec),
        )


@dataclass
class MockParameterServerConfig:
    """Mock parameter server config"""

    parameter_server_param: int = 2


class MockParameterServer(Component):
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
        """Static method that returns component name."""
        return "parameter_server"


@dataclass
class MockLoggerConfig:
    """Mock logger config"""

    logger_param_0: int = 1
    logger_param_1: int = 2


class MockLoggerClass:
    def __init__(self) -> None:
        """Mock logger component."""
        self._label = "logger_label"
        self._directory = "logger_directory"
        self._logger_info = (True, False, False, 10, print, 0)

    def write(self, data: Dict) -> None:
        """_summary_"""
        pass


class MockLogger(Component):
    def __init__(
        self,
        config: MockLoggerConfig = MockLoggerConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_executor_logger(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.store.executor_logger = MockLoggerClass()

    def on_building_trainer_logger(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.store.trainer_logger = MockLoggerClass()

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "logger"


@dataclass
class MockExecutorParameterClientConfig:
    """Mock parameter client config"""

    executor_parameter_client_param_0: int = 1
    executor_parameter_client_param_1: str = "param"


class MockExecutorParameterClient(Component):
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
        """Static method that returns component name."""
        return "executor_parameter_client"


@dataclass
class MockTrainerParameterClientConfig:
    """Mock parameter client config"""

    trainer_parameter_client_param_0: int = 1
    trainer_parameter_client_param_1: str = "param"
    trainer_parameter_update_period: int = 1


class MockTrainerParameterClient(Component):
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
        """Static method that returns component name."""
        return "trainer_parameter_client"


@dataclass
class MockExecutorDefaultConfig:
    """Mock executor config"""

    executor_param: int = 1


class MockExecutor(Component):
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
        """Static method that returns component name."""
        return "executor"


@dataclass
class MockExecutorEnvironmentLoopConfig:
    """Mock executor environment loop config"""

    should_update: bool = True


class MockExecutorEnvironmentLoop(Component):
    def __init__(
        self,
        config: MockExecutorEnvironmentLoopConfig = MockExecutorEnvironmentLoopConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_init_start(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_executor_environment(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.store.executor_environment = (
            builder.store.global_config.environment_factory(evaluation=False)
        )

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
        """Static method that returns component name."""
        return "executor_environment_loop"


@dataclass
class MockNetworksConfig:
    """Mock networks config"""

    network_factory: Optional[Callable[[str], dm_env.Environment]] = None
    seed: int = 1234


class MockNetworks(Component):
    def __init__(
        self,
        config: MockNetworksConfig = MockNetworksConfig(),
    ):
        """[summary]"""
        self.config = config

    def on_building_init_start(self, builder: SystemBuilder) -> None:
        """Summary"""

        # Setup the jax key for network initialisations
        builder.store.base_key = jax.random.PRNGKey(self.config.seed)

        # Build network function here
        network_key, builder.store.base_key = jax.random.split(builder.store.base_key)
        builder.store.network_factory = (
            lambda: self.config.network_factory(  # type: ignore
                environment_spec=builder.store.ma_environment_spec,
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
    """Mock trainer dataset config"""

    trainer_dataset_param: int = 5


class MockTrainerDataset(Component):
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
        """Static method that returns component name."""
        return "trainer_dataset"


@dataclass
class MockTrainerConfig:
    """Mock trainer config"""

    trainer_param_0: int = 2
    trainer_param_1: str = "train"


class MockTrainer(Component):
    def __init__(
        self,
        config: MockTrainerConfig = MockTrainerConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_init_end(self, builder: SystemBuilder) -> None:
        """TODO: Add description here."""
        builder.store.table_network_config = {
            "trainer_0": ["network_agent", "network_agent", "network_agent"]
        }
        builder.store.trainer_networks = {"trainer_0": ["network_agent"]}

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
        """Static method that returns component name."""
        return "trainer"


@dataclass
class DistributorConfig:
    """Mock distributor config"""

    num_executors: int = 1
    nodes_on_gpu: List[str] = field(default_factory=list)
    multi_process: bool = True
    name: str = "system"


class MockDistributor(Component):
    def __init__(self, config: DistributorConfig = DistributorConfig()) -> None:
        """Mock system distributor component.

        Args:
            config : dataclass configuration for setting component hyperparameters
        """
        self.config = config

    def on_building_program_nodes(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.store.data_key = [1234, 1234]
        builder.store.eval_key = [1234, 1234]
        builder.store.param_key = [1234, 1234]
        builder.store.executor_keys = [[1234, 1234]]
        builder.store.trainer_keys = [[1234, 1234]]

        data_server = builder.data_server()

        parameter_server = builder.parameter_server()

        trainer = builder.trainer(
            trainer_id="trainer_0",
            data_server_client=data_server,
            parameter_server_client=parameter_server,
        )
        executor = builder.executor(
            executor_id="executor_0",
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


def return_test_system(components: Dict) -> System:
    """Func that generates a test system based on a dict of components.

    Args:
        components : components that are part of the system.

    Returns:
        a system.
    """

    class TestSystem(System):
        def design(self) -> Tuple[DesignSpec, Dict]:
            """Mock system design with zero components.

            Returns:
                system callback components
            """
            return DesignSpec(**components), {}

    return TestSystem()
