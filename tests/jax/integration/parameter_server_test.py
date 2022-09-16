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
import time

import jax.numpy as jnp
import pytest
from acme.jax import savers as acme_savers

from mava.systems import System
from tests.jax.systems.systems_test_data import ippo_system_single_process


@pytest.fixture
def test_system_sp() -> System:
    """A single process built system"""
    return ippo_system_single_process()


def test_parameter_server_single_process(test_system_sp: System) -> None:
    """Test if the parameter server instantiates processes as expected."""
    (
        data_server,
        parameter_server,
        executor,
        evaluator,
        trainer,
    ) = test_system_sp._builder.store.system_build

    # Initial state of the parameter_server
    assert type(parameter_server.store.system_checkpointer) == acme_savers.Checkpointer

    param_without_net_and_opt = parameter_server.store.parameters.copy()
    del param_without_net_and_opt["policy_networks-network_agent"]
    del param_without_net_and_opt["critic_networks-network_agent"]
    del param_without_net_and_opt["policy_opt_state-network_agent"]
    del param_without_net_and_opt["critic_opt_state-network_agent"]
    assert param_without_net_and_opt == {
        "trainer_steps": jnp.zeros(1, dtype=jnp.int32),
        "trainer_walltime": jnp.zeros(1, dtype=jnp.float32),
        "evaluator_steps": jnp.zeros(1, dtype=jnp.int32),
        "evaluator_episodes": jnp.zeros(1, dtype=jnp.int32),
        "executor_episodes": jnp.zeros(1, dtype=jnp.int32),
        "executor_steps": jnp.zeros(1, dtype=jnp.int32),
    }

    # Check that checkpoint not yet saved
    assert parameter_server.store.system_checkpointer._last_saved == 0
    checkpoint_init_time = parameter_server.store.last_checkpoint_time

    first_network_param = parameter_server.store.parameters[
        "policy_networks-network_agent"
    ]

    # Test get and set parameters
    for _ in range(3):
        executor.run_episode()
    trainer.step()

    trainer_steps = parameter_server.get_parameters("trainer_steps")
    executor_episodes = parameter_server.get_parameters("executor_episodes")
    assert list(trainer_steps) == [1]  # trainer.step one time
    assert list(executor_episodes) == [3]  # run episodes three times

    # Check that the network is updated (at least one of the values updated)
    updated_networks_param = parameter_server.get_parameters(
        "policy_networks-network_agent"
    )
    at_least_one_changed = False
    for key in updated_networks_param.keys():
        assert sorted(list(updated_networks_param[key].keys())) == ["b", "w"]
        if not jnp.array_equal(
            updated_networks_param[key]["w"], first_network_param[key]["w"]
        ) or not jnp.array_equal(
            updated_networks_param[key]["b"], first_network_param[key]["b"]
        ):
            at_least_one_changed = True
            break
    assert at_least_one_changed

    #  Sleep until checkpoint_minute_interval elapses
    time.sleep(parameter_server.store.checkpoint_minute_interval * 60 + 2)

    # Run step function
    parameter_server.step()

    # Check that the checkpoint is saved thanks to the step function
    assert parameter_server.store.last_checkpoint_time > checkpoint_init_time
    assert parameter_server.store.last_checkpoint_time < time.time()
    assert parameter_server.store.system_checkpointer._last_saved != 0
    assert parameter_server.store.system_checkpointer._last_saved < time.time()
