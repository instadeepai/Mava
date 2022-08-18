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

"""Integration test of the Trainer for Jax-based Mava"""

import functools
import time
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
    """Mappo system"""
    return mappo.MAPPOSystem()


def test_trainer_single_process(test_system: System) -> None:
    """Test if the trainer instantiates processes as expected."""

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
        nodes_on_gpu=[],
        lp_launch_type=lp.LaunchType.TEST_MULTI_THREADING,
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

    for _ in range(5):
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


def test_trainer_multi_process(test_system: System) -> None:
    """Test if the trainer instantiates processes as expected."""
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
    test_system.build(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        experiment_path=checkpoint_subpath,
        optimizer=optimizer,
        executor_parameter_update_period=1,
        multi_process=True,
        run_evaluator=True,
        num_executors=1,
        max_queue_size=500,
        use_next_extras=False,
        sample_batch_size=5,
        nodes_on_gpu=[],
        lp_launch_type=lp.LaunchType.TEST_MULTI_THREADING,
    )

    # Disable the run of the trainer node
    (trainer_node,) = test_system._builder.store.program._program._groups["trainer"]
    trainer_node.disable_run()

    # launch the system
    test_system.launch()
    time.sleep(10)  # wait till the executor has run

    trainer = trainer_node._construct_instance()

    # Before run step function
    for net_key in trainer.store.networks["networks"].keys():
        mu = trainer.store.opt_states[net_key][1][0][-1]  # network
        for categorical_value_head in mu.values():
            assert jnp.all(categorical_value_head["b"] == 0)
            assert jnp.all(categorical_value_head["w"] == 0)

    # Step function
    trainer.step()

    # Check that the trainer update the network
    for net_key in trainer.store.networks["networks"].keys():
        mu = trainer.store.opt_states[net_key][1][0][-1]
        for categorical_value_head in mu.values():
            assert not jnp.all(categorical_value_head["b"] == 0)
            assert not jnp.all(categorical_value_head["w"] == 0)
