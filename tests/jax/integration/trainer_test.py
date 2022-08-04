import functools
from datetime import datetime
from typing import Any

import jax.numpy as jnp
import launchpad as lp
import optax
import pytest

from mava.systems.jax import System, mappo
from mava.utils.environments import debugging_utils
from mava.utils.loggers import logger_utils


@pytest.fixture
def test_system() -> System:
    """Dummy system with zero components."""
    return mappo.MAPPOSystem()


def test_parameter_server_process_instantiate(test_system: System) -> None:
    """Main script for running system."""

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
    log_every = 1
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

    # Build the test_system
    test_system.build(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        experiment_path=checkpoint_subpath,
        optimizer=optimizer,
        executor_parameter_update_period=10,
        multi_process=False,
        run_evaluator=True,
        num_executors=1,
        use_next_extras=False,
        max_queue_size=5000,
        sample_batch_size=2,
        max_in_flight_samples_per_worker=4,
        num_workers_per_iterator=-1,
        rate_limiter_timeout_ms=-1,
        checkpoint=True,
        nodes_on_gpu=[],
        lp_launch_type=lp.LaunchType.LOCAL_MULTI_PROCESSING,
        sequence_length=4,
        period=4,
    )

    (
        data_server,
        parameter_server,
        executor,
        evaluator,
        trainer,
    ) = test_system._builder.store.system_build

    for i in range(0, 4):
        executor.run_episode()

    # Before run step method
    for net_key in trainer.store.networks["networks"].keys():
        mu = trainer.store.opt_states[net_key][1][0][-1]  # network
        for categorical_value_head in mu.values():
            assert jnp.all(categorical_value_head["b"] == 0)
            assert jnp.all(categorical_value_head["w"] == 0)

    trainer.step()

    # After run step method
    for net_key in trainer.store.networks["networks"].keys():
        mu = trainer.store.opt_states[net_key][1][0][-1]
        for categorical_value_head in mu.values():
            assert not jnp.all(categorical_value_head["b"] == 0)
            assert not jnp.all(categorical_value_head["w"] == 0)
