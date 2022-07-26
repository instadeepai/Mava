import functools
from datetime import datetime
from typing import Any

import launchpad as lp
import optax
import pytest
from launchpad.launch.test_multi_threading import (
    address_builder as test_address_builder,
)

from mava.systems.jax import mappo
from mava.systems.jax.system import System
from mava.utils.environments import debugging_utils
from mava.utils.loggers import logger_utils

#########################################################################
# Full system integration test.


@pytest.fixture
def test_full_system() -> System:
    """Full mava system fixture for testing"""
    return mappo.MAPPOSystem()


def test_mappo(
    test_full_system: System,
) -> None:
    """Full integration test of mappo system."""

    # Environment.
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name="simple_spread",
        action_space="discrete",
    )

    # Networks.
    def network_factory(*args: Any, **kwargs: Any) -> Any:
        return mappo.make_default_networks(  # type: ignore
            policy_layer_sizes=(32, 32),
            critic_layer_sizes=(64, 64),
            *args,
            **kwargs,
        )

    # Checkpointer appends "Checkpoints" to checkpoint_dir.
    base_dir = "~/mava"
    mava_id = str(datetime.now())
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
        optax.clip_by_global_norm(40.0),
        optax.adam(1e-4),
    )

    # Build the system
    test_full_system.build(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        experiment_path=checkpoint_subpath,
        optimizer=optimizer,
        executor_parameter_update_period=1,
        multi_process=True,
        run_evaluator=False,
        num_executors=1,
        use_next_extras=False,
        sample_batch_size=5,
        checkpoint=True,
        nodes_on_gpu=[],
        lp_launch_type=lp.LaunchType.TEST_MULTI_THREADING,
    )

    (trainer_node,) = test_full_system._builder.store.program._program._groups[
        "trainer"
    ]
    trainer_node.disable_run()
    test_address_builder.bind_addresses([trainer_node])

    test_full_system.launch()
    trainer_run = trainer_node.create_handle().dereference()

    for _ in range(2):
        trainer_run.step()
