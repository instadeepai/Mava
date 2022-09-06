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

"""Built systems to be used in the integration tests"""

import functools
import tempfile
from datetime import datetime
from typing import Any

import optax

from mava.components.jax.updating.checkpointer import Checkpointer
from mava.systems.jax import System, ippo
from mava.utils.environments import debugging_utils
from mava.utils.loggers import logger_utils


def ippo_system_single_process() -> System:
    """Single process IPPO test system"""
    # Environment.
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name="simple_spread",
        action_space="discrete",
    )

    # Networks.
    def network_factory(*args: Any, **kwargs: Any) -> Any:
        return ippo.make_default_networks(  # type: ignore
            policy_layer_sizes=(32, 32),
            critic_layer_sizes=(64, 64),
            *args,
            **kwargs,
        )

    # Checkpointer appends "Checkpoints" to checkpoint_dir.
    base_dir = tempfile.mkdtemp()
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

    # Create ippo system
    test_system = ippo.IPPOSystem()
    test_system.add(Checkpointer)

    # Build the test_system
    test_system.build(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        experiment_path=checkpoint_subpath,
        optimizer=optimizer,
        executor_parameter_update_period=1,
        multi_process=False,  # Single process case
        run_evaluator=True,
        num_executors=1,
        use_next_extras=False,
        max_queue_size=5000,
        sample_batch_size=2,
        max_in_flight_samples_per_worker=4,
        num_workers_per_iterator=-1,
        rate_limiter_timeout_ms=-1,
        nodes_on_gpu=[],
        sequence_length=4,
        period=4,
        checkpoint_minute_interval=3 / 60,
        trainer_parameter_update_period=1,
    )

    return test_system


def ippo_system_multi_thread() -> System:
    """Multi thread IPPO test system using Launchpad"""
    # Environment.
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name="simple_spread",
        action_space="discrete",
    )

    # Networks.
    def network_factory(*args: Any, **kwargs: Any) -> Any:
        return ippo.make_default_networks(  # type: ignore
            policy_layer_sizes=(256, 256, 256),
            critic_layer_sizes=(512, 512, 256),
            *args,
            **kwargs,
        )

    # Checkpointer appends "Checkpoints" to checkpoint_dir.
    base_dir = tempfile.mkdtemp()
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
        optax.clip_by_global_norm(40.0), optax.scale_by_adam(), optax.scale(-1e-4)
    )

    # Create the ippo system
    test_system = ippo.IPPOSystem()
    test_system.add(Checkpointer)

    # Build the system.
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
        is_test=True,
        checkpoint_minute_interval=3 / 60,
        trainer_parameter_update_period=1,
    )
    return test_system
