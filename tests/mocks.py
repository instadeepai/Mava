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

import abc
import copy
import functools
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import dm_env
import jax
import numpy as np
import reverb
from acme import specs as acme_specs
from acme.specs import EnvironmentSpec
from acme.testing.fakes import Actor as ActorMock
from acme.testing.fakes import ContinuousEnvironment, DiscreteEnvironment
from acme.testing.fakes import Environment as MockedEnvironment
from acme.testing.fakes import _generate_from_spec, _validate_spec
from reverb import rate_limiters, reverb_types

from mava import core, specs
from mava.components import Component
from mava.components.building.data_server import (
    OffPolicyDataServerConfig,
    OnPolicyDataServerConfig,
)
from mava.core_jax import SystemBuilder
from mava.environment_loop import ParallelEnvironmentLoop
from mava.specs import DesignSpec, MAEnvironmentSpec
from mava.systems.system import System
from mava.types import OLT, NestedArray, Observation
from mava.utils.builder_utils import convert_specs
from mava.utils.wrapper_utils import convert_np_type, parameterized_restart
from mava.wrappers.env_wrappers import ParallelEnvWrapper
from tests.enums import MockedEnvironments

"""Mock Objects for Tests"""


class MockedExecutor(ActorMock, core.Executor):
    """Mock Exexutor Class."""

    def __init__(self, spec: specs.EnvironmentSpec):
        super().__init__(spec)
        self._specs = spec
        self._evaluator = False

    def select_actions(
        self, observations: Dict[str, NestedArray]
    ) -> Dict[str, NestedArray]:
        return {
            agent: _generate_from_spec(self._spec[agent].actions)
            for agent, observation in observations.items()
        }

    def observe_first(
        self,
        timestep: dm_env.TimeStep,
        extras: Dict[str, NestedArray] = {},
    ) -> None:
        for agent, observation_spec in self._specs.items():
            _validate_spec(
                observation_spec.observations,
                timestep.observation[agent],
            )
        if extras:
            _validate_spec(extras)

    def agent_observe_first(self, agent: str, timestep: dm_env.TimeStep) -> None:
        _validate_spec(self._spec[agent].observations, timestep.observation)

    def observe(
        self,
        action: Dict[str, NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, NestedArray] = {},
    ) -> None:

        for agent, observation_spec in self._spec.items():
            if agent in action.keys():
                _validate_spec(observation_spec.actions, action[agent])

            if agent in next_timestep.reward.keys():
                _validate_spec(observation_spec.rewards, next_timestep.reward[agent])

            if agent in next_timestep.discount.keys():
                _validate_spec(
                    observation_spec.discounts, next_timestep.discount[agent]
                )

            if next_timestep.observation and agent in next_timestep.observation.keys():
                _validate_spec(
                    observation_spec.observations, next_timestep.observation[agent]
                )
        if next_extras:
            _validate_spec(next_extras)

    def agent_observe(
        self,
        agent: str,
        action: Union[float, int, NestedArray],
        next_timestep: dm_env.TimeStep,
    ) -> None:
        observation_spec = self._spec[agent]
        _validate_spec(observation_spec.actions, action)
        _validate_spec(observation_spec.rewards, next_timestep.reward)
        _validate_spec(observation_spec.discounts, next_timestep.discount)


class MockedSystem(MockedExecutor):
    """Mocked System Class."""

    def __init__(
        self,
        specs: specs.EnvironmentSpec,
    ):
        super().__init__(specs)
        self._specs = specs

        # Initialize Mock Vars
        self.variables: Dict = {}
        network_type = "mlp"
        self.variables[network_type] = {}
        for agent in self._specs.keys():
            self.variables[network_type][agent] = np.random.rand(5, 5)

    def get_variables(self, names: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        variables: Dict = {}
        for network_type in names:
            variables[network_type] = {
                agent: self.variables[network_type][agent] for agent in self.agents
            }
        return variables


"""Function returns a Multi-agent env, of type base_class.
base_class: DiscreteEnvironment or ContinuousEnvironment. """


def get_ma_environment(
    base_class: Union[DiscreteEnvironment, ContinuousEnvironment]
) -> Any:
    class MockedEnvironment(base_class):  # type: ignore
        """Mocked Multi-Agent Environment.
        This simply creates multiple agents, with a spec per agent
        and updates the spec functions of base_class."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            base_class.__init__(self, *args, **kwargs)
            self._agents = ["agent_0", "agent_1", "agent_2"]
            self._possible_agents = self.agents
            self.num_agents = len(self.agents)

            multi_agent_specs = {}
            for agent in self.agents:
                spec = self._spec
                actions = spec.actions
                rewards = spec.rewards
                discounts = spec.discounts

                # Observation spec needs to be an OLT
                ma_observation_spec = self.observation_spec()
                multi_agent_specs[agent] = EnvironmentSpec(
                    observations=ma_observation_spec,
                    actions=actions,
                    rewards=rewards,
                    discounts=discounts,
                )

            self._specs = multi_agent_specs

        def extras_spec(self) -> Dict:
            return {}

        def reward_spec(self) -> Dict[str, acme_specs.Array]:
            reward_specs = {}
            for agent in self.agents:
                reward_specs[agent] = super().reward_spec()
            return reward_specs

        def discount_spec(self) -> Dict[str, acme_specs.BoundedArray]:
            discount_specs = {}
            for agent in self.agents:
                discount_specs[agent] = super().discount_spec()
            return discount_specs

        @property
        def agents(self) -> List:
            return self._agents

        @property
        def possible_agents(self) -> List:
            return self._possible_agents

        @property
        def env_done(self) -> bool:
            return not self.agents

    return MockedEnvironment


"""Class that updates functions for parallel environment.
This class should be inherited with a MockedEnvironment. """


class ParallelEnvironment(MockedEnvironment, ParallelEnvWrapper):
    def __init__(self, agents: List, specs: EnvironmentSpec) -> None:
        self._agents = agents
        self._specs = specs

    def action_spec(
        self,
    ) -> Dict[str, Union[acme_specs.DiscreteArray, acme_specs.BoundedArray]]:
        action_spec = {}
        for agent in self.agents:
            action_spec[agent] = super().action_spec()
        return action_spec

    def observation_spec(self) -> Observation:
        observation_specs = {}
        for agent in self.agents:
            legals = self.action_spec()[agent]
            terminal = acme_specs.Array(
                (1,),
                np.float32,
            )

            observation_specs[agent] = OLT(
                observation=super().observation_spec(),
                legal_actions=legals,
                terminal=terminal,
            )
        return observation_specs

    def _generate_fake_observation(self) -> Observation:
        return _generate_from_spec(self.observation_spec())

    def reset(self) -> dm_env.TimeStep:
        observations = {}
        for agent in self.agents:
            observation = self._generate_fake_observation()
            observations[agent] = observation

        rewards = {agent: convert_np_type("float32", 0) for agent in self.agents}
        discounts = {agent: convert_np_type("float32", 1) for agent in self.agents}

        self._step = 1
        return parameterized_restart(rewards, discounts, observations)  # type: ignore

    def step(
        self, actions: Dict[str, Union[float, int, NestedArray]]
    ) -> dm_env.TimeStep:

        # Return a reset timestep if we haven't touched the environment yet.
        if not self._step:
            return self.reset()

        for agent, action in actions.items():
            _validate_spec(self._specs[agent].actions, action)

        observation = {
            agent: self._generate_fake_observation() for agent in self.agents
        }
        reward = {agent: self._generate_fake_reward() for agent in self.agents}
        discount = {agent: self._generate_fake_discount() for agent in self.agents}

        if self._episode_length and (self._step == self._episode_length):
            self._step = 0
            # We can't use dm_env.termination directly because then the discount
            # wouldn't necessarily conform to the spec (if eg. we want float32).
            return dm_env.TimeStep(dm_env.StepType.LAST, reward, discount, observation)
        else:
            self._step += 1
            return dm_env.transition(
                reward=reward, observation=observation, discount=discount
            )


"""Mocked Multi-Agent Discrete Environment"""


DiscreteMAEnvironment = get_ma_environment(DiscreteEnvironment)
ContinuousMAEnvironment = get_ma_environment(ContinuousEnvironment)


class MockedMADiscreteEnvironment(DiscreteMAEnvironment, DiscreteEnvironment):  # type: ignore
    def __init__(self, *args: Any, **kwargs: Any):
        DiscreteMAEnvironment.__init__(self, *args, **kwargs)


"""Mocked Multi-Agent Continuous Environment"""


class MockedMAContinuousEnvironment(
    ContinuousMAEnvironment, ContinuousEnvironment  # type: ignore
):
    def __init__(self, *args: Any, **kwargs: Any):
        ContinuousMAEnvironment.__init__(self, *args, **kwargs)


"""Mocked Multi-Agent Parallel Discrete Environment"""


class ParallelMADiscreteEnvironment(ParallelEnvironment, MockedMADiscreteEnvironment):
    def __init__(self, *args: Any, **kwargs: Any):
        MockedMADiscreteEnvironment.__init__(self, *args, **kwargs)
        ParallelEnvironment.__init__(self, self.agents, self._specs)


"""Mocked Multi-Agent Parallel Continuous Environment"""


class ParallelMAContinuousEnvironment(
    ParallelEnvironment, MockedMAContinuousEnvironment
):
    def __init__(self, *args: Any, **kwargs: Any):
        MockedMAContinuousEnvironment.__init__(self, *args, **kwargs)
        ParallelEnvironment.__init__(self, self.agents, self._specs)


"""Mocked Multi-Agent Sequential Continuous Environment"""


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
        self, timestep: dm_env.TimeStep, extras: Dict[str, NestedArray] = {}
    ) -> None:
        """_summary_

        Args:
            timestep : _description_
            extras : _description_.
        """
        pass

    def add(
        self,
        actions: Dict[str, NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, NestedArray] = {},
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
    evaluation: bool = False,
) -> Any:
    """Func that creates a fake env.

    Args:
        env_name : env name.
        evaluation: whether env is used for eval or not.
            Not sure we should use this in spec.

    Raises:
        Exception: no matching env.

    Returns:
        mock env.
    """
    del evaluation
    env = ParallelMADiscreteEnvironment(
        num_actions=18,
        num_observations=2,
        obs_shape=(84, 84, 4),
        obs_dtype=np.float32,
        episode_length=10,
    )

    if env is None:
        raise Exception("Env_spec is not valid.")

    return env


def make_fake_environment_factory(
    env_name: MockedEnvironments = MockedEnvironments.Mocked_Dicrete,
) -> Any:
    """Returns a mock env factory.

    Args:
        env_name : env name.

    Returns:
        a mocked env factory.
    """
    return functools.partial(
        make_fake_env,
        env_name=env_name,
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


class MockDataServer(Component):
    def __init__(self, config: SimpleNamespace = SimpleNamespace()) -> None:
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
        builder.store.base_key, network_key = jax.random.split(builder.store.base_key)
        builder.store.network_factory = (
            lambda: self.config.network_factory(  # type: ignore
                environment_spec=builder.store.ma_environment_spec,
                agent_net_keys=builder.store.agent_net_keys,
                base_key=network_key,
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
        @staticmethod
        def design() -> Tuple[DesignSpec, Dict]:
            """Mock system design with zero components.

            Returns:
                system callback components
            """
            return DesignSpec(**components), {}

    return TestSystem()
