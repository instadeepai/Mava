import functools
from typing import Dict, Tuple

import numpy as np
import pytest

from mava.components.jax import building
from mava.components.jax.building.adders import ParallelTransitionAdderSignature
from mava.components.jax.updating.parameter_server import DefaultParameterServer
from mava.specs import DesignSpec
from mava.systems.jax import ParameterServer, mappo
from mava.systems.jax.system import System
from mava.utils.environments import debugging_utils
from tests.jax import mocks


class TestSystem(System):
    def design(self) -> Tuple[DesignSpec, Dict]:
        """Mock system design with zero components.

        Returns:
            system callback components
        """
        components = DesignSpec(
            environment_spec=building.EnvironmentSpec,
            system_init=building.FixedNetworkSystemInit,
            data_server=mocks.MockOnPolicyDataServer,
            data_server_adder_signature=ParallelTransitionAdderSignature,
            parameter_server=DefaultParameterServer,
            executor_parameter_client=mocks.MockExecutorParameterClient,
            trainer_parameter_client=mocks.MockTrainerParameterClient,
            logger=mocks.MockLogger,
            executor=mocks.MockExecutor,
            executor_adder=mocks.MockAdder,
            executor_environment_loop=mocks.MockExecutorEnvironmentLoop,
            networks=mocks.MockNetworks,
            trainer=mocks.MockTrainer,
            trainer_dataset=mocks.MockTrainerDataset,
            distributor=mocks.MockDistributor,
        )
        return components, {}


@pytest.fixture
def test_system() -> System:
    """Dummy system with zero components."""
    return TestSystem()


def test_parameter_server_process_instantiate(
    test_system: System,
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

    test_system.build(
        environment_factory=environment_factory,
        network_factory=network_factory,
        non_blocking_sleep_seconds=0,
    )
    (
        data_server,
        parameter_server,
        executor,
        evaluator,
        trainer,
    ) = test_system._builder.store.system_build
    assert type(parameter_server) == ParameterServer

    step_var = parameter_server.get_parameters("trainer_steps")
    assert type(step_var) == np.ndarray
    assert step_var[0] == 0

    parameter_server.set_parameters({"trainer_steps": np.ones(1, dtype=np.int32)})
    assert parameter_server.get_parameters("trainer_steps")[0] == 1

    parameter_server.add_to_parameters({"trainer_steps": np.ones(1, dtype=np.int32)})
    assert parameter_server.get_parameters("trainer_steps")[0] == 2

    # Step the parameter sever
    parameter_server.step()
