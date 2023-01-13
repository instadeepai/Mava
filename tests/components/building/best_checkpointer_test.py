from types import SimpleNamespace
from typing import Any

import pytest

from mava.components.building import BestCheckpointer
from mava.components.building.best_checkpointer import BestCheckpointerConfig
from mava.core_jax import SystemBuilder, SystemParameterServer
from mava.systems.builder import Builder
from mava.systems.parameter_server import ParameterServer


class MockBuilder(Builder):
    def __init__(self, store: SimpleNamespace) -> None:
        """Initialises mock builder"""
        self.store = store

    def has(self, instance: Any) -> bool:
        """Has: mock method"""
        return True


class MockParameterServer(ParameterServer):
    def __init__(self, store: SimpleNamespace) -> None:
        """Initialises mock parameter server"""
        self.store = store
        self.calculate_absolute_metric = False

    def has(self, instance: Any) -> bool:
        """Has: mock method"""
        return False


class MockNetwork:
    def __init__(self) -> None:
        """Initialises mock network"""
        self.policy_params = {"w": [1, 2, 3]}
        self.critic_params = {"w": [1, 2, 3]}


@pytest.fixture
def checkpointer() -> BestCheckpointer:
    """Creates a BestCheckpointer"""
    conf = BestCheckpointerConfig(
        checkpointing_metric=("mean_episode_return",), checkpoint_best_perf=True
    )
    return BestCheckpointer(conf)


@pytest.fixture
def checkpointer_absolute_metric() -> BestCheckpointer:
    """Creates a BestCheckpointer"""
    conf = BestCheckpointerConfig(
        checkpointing_metric=("mean_episode_return",), absolute_metric=True
    )
    return BestCheckpointer(conf)


@pytest.fixture
def builder() -> Builder:
    """Creates a builder"""
    store = SimpleNamespace(
        is_evaluator=True,
        networks={"agent_0": MockNetwork()},
        policy_opt_states={"agent_0": {"opt_state": [1, 2, 3]}},
        critic_opt_states={"agent_0": {"opt_state": [1, 2, 3]}},
        norm_params={"agent_0": [0.2, 0.3, 0.5]},
        global_config=SimpleNamespace(
            evaluation_duration={"evaluator_episodes": 32},
            normalise_observations=True,
            normalise_target_values=True,
        ),
    )
    return MockBuilder(store)


@pytest.fixture
def parameter_server() -> ParameterServer:
    """Creates a parameter server"""
    store = SimpleNamespace(
        parameters={"some_params": [1, 2, 3]},
        networks={"agent_0": MockNetwork()},
        policy_opt_states={"agent_0": {"opt_state": [1, 2, 3]}},
        critic_opt_states={"agent_0": {"opt_state": [1, 2, 3]}},
    )
    return MockParameterServer(store)


@pytest.fixture
def parameter_server_with_termination_cond() -> ParameterServer:
    """Creates a parameter server"""
    store = SimpleNamespace(
        parameters={"some_params": [1, 2, 3]},
        networks={"agent_0": MockNetwork()},
        policy_opt_states={"agent_0": {"opt_state": [1, 2, 3]}},
        critic_opt_states={"agent_0": {"opt_state": [1, 2, 3]}},
        global_config=SimpleNamespace(termination_condition={"executor_steps": 10000}),
    )
    return MockParameterServer(store)


@pytest.fixture
def parameter_server_without_termination_cond() -> ParameterServer:
    """Creates a parameter server"""
    store = SimpleNamespace(
        parameters={"some_params": [1, 2, 3]},
        networks={"agent_0": MockNetwork()},
        policy_opt_states={"agent_0": {"opt_state": [1, 2, 3]}},
        critic_opt_states={"agent_0": {"opt_state": [1, 2, 3]}},
        global_config=SimpleNamespace(termination_condition=None),
    )
    return MockParameterServer(store)


def test___init__(checkpointer: BestCheckpointer) -> None:
    """Test BestCheckpointer init"""
    assert checkpointer.config.checkpointing_metric == ("mean_episode_return",)
    assert checkpointer.config.checkpoint_best_perf


def test_on_building_init(
    checkpointer: BestCheckpointer, builder: SystemBuilder
) -> None:
    """Test checkpointing dict is initialised in the store"""
    checkpointer.on_building_init(builder)
    assert builder.store.checkpointing_metric == {"mean_episode_return": None}


def test_on_building_executor_start(
    checkpointer: BestCheckpointer, builder: SystemBuilder
) -> None:
    """Test checkpointing params are initialised in the store"""
    # Testing on executor
    builder.store.is_evaluator = False

    checkpointer.on_building_init(builder)
    checkpointer.on_building_executor_start(builder)

    assert not hasattr(builder.store, "best_checkpoint")

    # Testing on evaluator
    builder.store.is_evaluator = True

    checkpointer.on_building_executor_start(builder)
    assert builder.store.best_checkpoint == checkpointer.init_checkpointing_params(
        builder
    )


def test_on_parameter_server_init(
    checkpointer: BestCheckpointer,
    parameter_server: SystemParameterServer,
    checkpointer_absolute_metric: BestCheckpointer,
    parameter_server_with_termination_cond: SystemParameterServer,
    parameter_server_without_termination_cond: SystemParameterServer,
) -> None:
    """Tests that checkpointing parameters are added to parameter server"""
    checkpointer.config.checkpoint_best_perf = False

    checkpointer.on_building_init(parameter_server)  # type: ignore
    checkpointer.on_parameter_server_init(parameter_server)

    # Testing when not checkpointing best perf
    assert parameter_server.store.parameters == {"some_params": [1, 2, 3]}

    checkpointer.config.checkpoint_best_perf = True
    checkpointer.on_parameter_server_init(parameter_server)

    # Testing when checkpointing best perf
    assert parameter_server.store.parameters == {
        "some_params": [1, 2, 3],
        "best_checkpoint": {
            **checkpointer.init_checkpointing_params(parameter_server),
        },
    }

    # Test the case of absolute metric and termination condition
    checkpointer_absolute_metric.on_building_init(
        parameter_server_with_termination_cond  # type:ignore
    )
    checkpointer_absolute_metric.on_parameter_server_init(
        parameter_server_with_termination_cond
    )

    assert (
        parameter_server_with_termination_cond.calculate_absolute_metric  # type:ignore
    )

    # Test the case of absolute metric and without termination condition
    with pytest.raises(AttributeError):
        # Error will be caused by calling termination_fn
        checkpointer_absolute_metric.on_building_init(
            parameter_server_without_termination_cond  # type:ignore
        )
        checkpointer_absolute_metric.on_parameter_server_init(
            parameter_server_without_termination_cond
        )


def test_init_checkpointing_params(
    checkpointer: BestCheckpointer, builder: SystemBuilder
) -> None:
    """Tests parameters are initialised correctly for checkpointing"""
    params = checkpointer.init_checkpointing_params(builder)
    assert params == {
        "mean_episode_return": {
            "best_performance": None,
            "policy_network-agent_0": {"w": [1, 2, 3]},
            "critic_network-agent_0": {"w": [1, 2, 3]},
            "policy_opt_state-agent_0": {"opt_state": [1, 2, 3]},
            "critic_opt_state-agent_0": {"opt_state": [1, 2, 3]},
            "norm_params": {"agent_0": [0.2, 0.3, 0.5]},
        }
    }
