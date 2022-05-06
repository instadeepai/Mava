import functools
from typing import Tuple

import acme
import optax
import pytest

from mava.systems.jax import mappo
from mava.systems.jax.mappo import MAPPOSystem
from mava.utils.environments import debugging_utils
from mava.utils.loggers import logger_utils

#########################################################################
# Full system integration test.
# TODO: This test is too expensive as it creates a full system. Many of the
#  system parts
# could be replaced with mock objects.

# TODO: These tests could be moved to executer_test.py.
#########################################################################


@pytest.fixture
def test_full_system() -> Tuple:
    """Creates a full system."""
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
    system = MAPPOSystem()
    system.build(
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
    ) = system._builder.store.system_build

    return data_server, parameter_server, executor, evaluator, trainer


def test_except_trainer(test_full_system: Tuple) -> None:
    """Test if the parameter server instantiates processes as expected."""
    data_server, parameter_server, executor, evaluator, trainer = test_full_system
    assert isinstance(executor, acme.core.Worker)


def test_executor_counter_init(test_full_system: Tuple) -> None:
    """Test if the counter in the executor is initialized correctly."""
    _, _, executor, _, _ = test_full_system
    assert executor._executor.store.steps_count == 0


def test_executor_counter_update(test_full_system: Tuple) -> None:
    """Test if the counter in the executor is updated correctly."""
    _, _, executor, _, _ = test_full_system
    nr_steps_per_episode = 50
    executor.run_episode()
    assert executor._executor.store.steps_count == nr_steps_per_episode
    executor.run_episode()
    assert executor._executor.store.steps_count == 2 * nr_steps_per_episode
