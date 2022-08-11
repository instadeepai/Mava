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

"""Tests for parameter server for Jax-based Mava systems"""
import functools
import time
from datetime import datetime
from typing import Any, Dict

import jax.numpy as jnp
import launchpad as lp
import optax
import pytest
from acme.jax import savers

from mava.systems.jax import System, mappo
from mava.utils.environments import debugging_utils
from mava.utils.loggers import logger_utils


@pytest.fixture
def test_system() -> System:
    """Mappo system"""
    return mappo.MAPPOSystem()


def test_parameter_server(test_system: System) -> None:
    """Test if the paraameter server instantiates processes as expected."""

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

    # System parameter server
    assert type(parameter_server.store.system_checkpointer) == savers.Checkpointer

    for network in parameter_server.store.parameters["networks-network_agent"].values():
        assert list(network.keys()) == ["w", "b"]
        assert jnp.size(network["w"]) != 0
        assert jnp.size(network["b"]) != 0

    param_without_net = parameter_server.store.parameters.copy()
    del param_without_net["networks-network_agent"]
    assert param_without_net == {
        "trainer_steps": jnp.zeros(1, dtype=jnp.int32),
        "trainer_walltime": jnp.zeros(1, dtype=jnp.float32),
        "evaluator_steps": jnp.zeros(1, dtype=jnp.int32),
        "evaluator_episodes": jnp.zeros(1, dtype=jnp.int32),
        "executor_episodes": jnp.zeros(1, dtype=jnp.int32),
        "executor_steps": jnp.zeros(1, dtype=jnp.int32),
    }

    ############################get_parameters test#####################################

    parameter_server.get_parameters("trainer_steps")
    assert parameter_server.store.get_parameters == jnp.zeros(1, dtype=jnp.int32)

    parameter_server.get_parameters("networks-network_agent")
    assert (
        parameter_server.store.get_parameters
        == parameter_server.store.parameters["networks-network_agent"]
    )

    # get multiple params
    parameter_server.get_parameters(["executor_episodes", "executor_steps"])
    assert parameter_server.store.get_parameters == {
        "executor_episodes": jnp.zeros(1, dtype=jnp.int32),
        "executor_steps": jnp.zeros(1, dtype=jnp.int32),
    }

    ############################set_parameters test#####################################
    params = {
        "trainer_steps": jnp.zeros(2, dtype=jnp.int32),
        "trainer_walltime": jnp.zeros(2, dtype=jnp.float32),
        "evaluator_steps": jnp.zeros(2, dtype=jnp.int32),
        "evaluator_episodes": jnp.zeros(2, dtype=jnp.int32),
        "executor_episodes": jnp.zeros(2, dtype=jnp.int32),
        "executor_steps": jnp.zeros(2, dtype=jnp.int32),
    }

    parameter_server.set_parameters(params)
    list_param = [
        "trainer_steps",
        "trainer_walltime",
        "evaluator_steps",
        "evaluator_episodes",
        "executor_episodes",
        "executor_steps",
    ]
    for param in list_param:
        assert jnp.array_equal(
            parameter_server.store.parameters[param], jnp.array([0, 0])
        )

    # set non existing param
    param1: Dict[str, str] = {"wrong_param": "test"}
    with pytest.raises(AssertionError):
        assert parameter_server.set_parameters(param1)

    ############################add_to_parameters_test#####################################
    param2: Dict[str, Any] = {
        "trainer_steps": jnp.array([1, 3]),
    }
    parameter_server.add_to_parameters(param2)

    assert jnp.array_equal(
        parameter_server.store.parameters["trainer_steps"], jnp.array([1, 3])
    )  # [0,0]+[1,3]

    ###################################step_test##########################################
    # before running step
    assert not parameter_server.store.last_checkpoint_time
    assert parameter_server.store.system_checkpointer._last_saved == 0

    # run step function
    parameter_server.step()

    assert parameter_server.store.last_checkpoint_time
    assert parameter_server.store.last_checkpoint_time < time.time()
    assert parameter_server.store.system_checkpointer._last_saved != 0
    assert parameter_server.store.system_checkpointer._last_saved < time.time()
