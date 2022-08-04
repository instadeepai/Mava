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

"""Tests for executor class for Jax-based Mava systems"""

import functools
from typing import Dict, Tuple

import pytest

from mava.components.jax import building, executing
from mava.components.jax.building.adders import (
    ParallelSequenceAdderSignature,
    UniformAdderPriority,
)
from mava.components.jax.building.data_server import OnPolicyDataServer
from mava.components.jax.building.distributor import Distributor
from mava.components.jax.building.parameter_client import ExecutorParameterClient
from mava.components.jax.updating.parameter_server import DefaultParameterServer
from mava.specs import DesignSpec
from mava.systems.jax import mappo
from mava.systems.jax.mappo.components import ExtrasLogProbSpec
from mava.systems.jax.system import System
from mava.types import OLT
from mava.utils.environments import debugging_utils
from tests.jax import mocks

system_init = DesignSpec(
    environment_spec=building.EnvironmentSpec,
    system_init=building.FixedNetworkSystemInit,
).get()
executor = DesignSpec(
    executor_init=executing.ExecutorInit,
    executor_observe=executing.FeedforwardExecutorObserve,
    executor_select_action=executing.FeedforwardExecutorSelectAction,
    executor_adder=building.ParallelSequenceAdder,
    adder_priority=UniformAdderPriority,
    executor_environment_loop=building.ParallelExecutorEnvironmentLoop,
    networks=building.DefaultNetworks,
).get()


class TestSystemExceptTrainer(System):
    def design(self) -> Tuple[DesignSpec, Dict]:
        """Mock system without trainer

        Returns:
            system callback components
        """
        components = DesignSpec(
            **system_init,
            data_server=OnPolicyDataServer,
            data_server_adder_signature=ParallelSequenceAdderSignature,
            extras_spec=ExtrasLogProbSpec,
            parameter_server=DefaultParameterServer,
            executor_parameter_client=ExecutorParameterClient,
            **executor,
            distributor=Distributor,
            trainer_parameter_client=mocks.MockTrainerParameterClient,
            logger=mocks.MockLogger,
            trainer=mocks.MockTrainer,
            trainer_dataset=mocks.MockTrainerDataset,
        )
        return components, {}


@pytest.fixture
def test_system_except_trainer() -> System:
    """Create system mock"""
    return TestSystemExceptTrainer()


def test_executor_behavior_witohut_adder(
    test_system_except_trainer: System,
) -> None:
    """Test if the executor instantiates processes as expected."""

    # Environment.
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name="simple_spread",
        action_space="discrete",
    )

    # Networks.
    network_factory = mappo.make_default_networks

    # Build the system
    test_system_except_trainer.build(
        environment_factory=environment_factory,
        network_factory=network_factory,
        executor_parameter_update_period=20,
        multi_process=False,
        run_evaluator=True,  # run evaluator will remove the adder
        num_executors=1,
        use_next_extras=False,
    )

    (
        data_server,
        parameter_server,
        executor,
        evaluator,
        trainer,
    ) = test_system_except_trainer._builder.store.system_build

    # Run an episode
    executor.run_episode()

    # Observe first (without adder)
    assert executor._executor.store.adder is None

    # Select actions and select action
    assert list(executor._executor.store.actions_info.keys()) == [
        "agent_0",
        "agent_1",
        "agent_2",
    ]
    assert list(executor._executor.store.policies_info.keys()) == [
        "agent_0",
        "agent_1",
        "agent_2",
    ]
    num_possible_actions = environment_factory().action_spec()["agent_0"].num_values
    assert (
        lambda: x in range(0, num_possible_actions)
        for x in list(executor._executor.store.actions_info.values())
    )

    assert (
        lambda: key == "log_prob"
        for key in executor._executor.store.policies_info.values()
    )

    # Observe (without adder)
    assert not hasattr(executor._executor.store.adder, "add")


def test_executor_behavior(
    test_system_except_trainer: System,
) -> None:
    """Test if the executor instantiates processes as expected."""

    # Environment.
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name="simple_spread",
        action_space="discrete",
    )

    # Networks.
    network_factory = mappo.make_default_networks

    # Build the system
    test_system_except_trainer.build(
        environment_factory=environment_factory,
        network_factory=network_factory,
        executor_parameter_update_period=20,
        multi_process=False,
        run_evaluator=False,
        num_executors=1,
        use_next_extras=False,
    )

    (
        data_server,
        parameter_server,
        executor,
        trainer,
    ) = test_system_except_trainer._builder.store.system_build

    # _writer.append needs to be called once to get _writer.history
    # _writer.append called in observe_first and observe
    with pytest.raises(RuntimeError):
        assert executor._executor.store.adder._writer.history

    # Run an episode
    executor.run_episode()

    # Observe first and observe
    assert executor._executor.store.adder._writer.history
    assert list(executor._executor.store.adder._writer.history.keys()) == [
        "observations",
        "start_of_episode",
        "actions",
        "rewards",
        "discounts",
        "extras",
    ]
    assert list(
        executor._executor.store.adder._writer.history["observations"].keys()
    ) == ["agent_0", "agent_1", "agent_2"]
    assert (
        type(executor._executor.store.adder._writer.history["observations"]["agent_0"])
        == OLT
    )

    assert len(executor._executor.store.adder._writer._column_history) != 0

    # Select actions and select action
    assert list(executor._executor.store.actions_info.keys()) == [
        "agent_0",
        "agent_1",
        "agent_2",
    ]
    assert list(executor._executor.store.policies_info.keys()) == [
        "agent_0",
        "agent_1",
        "agent_2",
    ]
    num_possible_actions = environment_factory().action_spec()["agent_0"].num_values
    assert (
        lambda: x in range(0, num_possible_actions)
        for x in list(executor._executor.store.actions_info.values())
    )
    assert (
        lambda: key == "log_prob"
        for key in executor._executor.store.policies_info.values()
    )
