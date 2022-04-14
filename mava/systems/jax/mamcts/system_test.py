import functools

import acme
import optax
import pytest

from mava.systems.jax import mamcts
from mava.systems.jax.system import System
from mava.utils.environments import debugging_utils
from mava.utils.loggers import logger_utils

#########################################################################
# Full system integration test.


@pytest.fixture
def test_full_system() -> System:
    """Add description here."""
    return mamcts.MAMCTSSystem()


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
    network_factory = mamcts.make_default_networks

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
        n_step=2,
        discount_factor=0.9,
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
