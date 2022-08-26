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

import functools
from datetime import datetime
from typing import Any

import launchpad as lp
import optax
import pytest
from launchpad.launch.test_multi_threading import (
    address_builder as test_address_builder,
)

from mava.systems.jax import ippo
from mava.systems.jax.system import System
from mava.utils.environments import debugging_utils
from mava.utils.loggers import logger_utils

#########################################################################
# Full system integration test.


@pytest.fixture
def test_full_system() -> System:
    """Full mava system fixture for testing"""
    return ippo.IPPOSystem()


def test_ippo(
    test_full_system: System,
) -> None:
    """Full integration test of ippo system."""

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
        run_evaluator=True,
        num_executors=1,
        max_queue_size=500,
        use_next_extras=False,
        sample_batch_size=5,
        nodes_on_gpu=[],
        is_test=True,
    )

    (trainer_node,) = test_full_system._builder.store.program._program._groups[
        "trainer"
    ]
    trainer_node.disable_run()
    test_address_builder.bind_addresses([trainer_node])

    test_full_system.launch()
    trainer_run = trainer_node.create_handle().dereference()

    for _ in range(5):
        trainer_run.step()
