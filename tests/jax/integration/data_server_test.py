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
"""Tests for data server for Jax-based Mava systems"""

import jax.numpy as jnp
import pytest
import reverb
from tensorflow.python.data.ops import dataset_ops

from mava.systems.jax import System
from tests.jax.systems.test_systems import test_ippo_system_single_process


@pytest.fixture
def test_system() -> System:
    """A built IPPO system"""
    return test_ippo_system_single_process()


def get_dataset(data_server: reverb.client.Client) -> dataset_ops.DatasetV1Adapter:
    """Batches 2 sequences to get samples"""
    dataset = reverb.TrajectoryDataset.from_table_signature(
        server_address=data_server._server_address,
        table="trainer",
        max_in_flight_samples_per_worker=10,
    )
    dataset = dataset.batch(2)
    return dataset


def test_data_server(test_system: System) -> None:
    """Test if the data server instantiates processes as expected."""
    (
        data_server,
        parameter_server,
        executor,
        evaluator,
        trainer,
    ) = test_system._builder.store.system_build

    table_trainer = data_server.server_info()["trainer"]
    assert table_trainer.name == "trainer"
    assert table_trainer.max_size == 5000  # max_queue_size
    assert table_trainer.max_times_sampled == 1

    rate_limiter = table_trainer.rate_limiter_info
    assert rate_limiter.samples_per_insert == 1
    assert rate_limiter.max_diff == 5000
    assert rate_limiter.min_size_to_sample == 1

    signature = table_trainer.signature
    assert sorted(list(signature.observations.keys())) == [
        "agent_0",
        "agent_1",
        "agent_2",
    ]
    for observation in signature.observations.values():
        assert observation.observation
        assert observation.legal_actions
        assert observation.terminal
    assert sorted(list(signature.actions.keys())) == ["agent_0", "agent_1", "agent_2"]
    assert sorted(list(signature.rewards.keys())) == ["agent_0", "agent_1", "agent_2"]
    assert sorted(list(signature.discounts.keys())) == ["agent_0", "agent_1", "agent_2"]
    assert sorted(list(signature.extras.keys())) == ["network_keys", "policy_info"]
    assert sorted(list(signature.extras["network_keys"].keys())) == [
        "agent_0",
        "agent_1",
        "agent_2",
    ]
    assert sorted(list(signature.extras["policy_info"].keys())) == [
        "agent_0",
        "agent_1",
        "agent_2",
    ]

    assert table_trainer.num_episodes == 0
    assert table_trainer.num_unique_samples == 0

    # Run episodes
    for _ in range(0, 5):
        executor.run_episode()

    table_trainer = data_server.server_info()["trainer"]
    assert table_trainer.num_episodes == 5  # 5 episodes
    assert table_trainer.num_unique_samples != 0

    # dataset added by the executor via the adders
    dataset = get_dataset(data_server)
    for sample in dataset.take(1):
        assert sorted(list(sample.data.observations.keys())) == [
            "agent_0",
            "agent_1",
            "agent_2",
        ]
        for observation in sample.data.observations.values():
            assert jnp.size(observation.observation) != 0
            assert jnp.size(observation.legal_actions) != 0
            assert jnp.size(observation.terminal) != 0
        assert sorted(list(sample.data.actions.keys())) == [
            "agent_0",
            "agent_1",
            "agent_2",
        ]
        for agent_action in sample.data.actions.values():
            assert jnp.size(agent_action) != 0
        assert sorted(list(sample.data.rewards.keys())) == [
            "agent_0",
            "agent_1",
            "agent_2",
        ]
        for agent_reward in sample.data.rewards.values():
            assert jnp.size(agent_reward) != 0
        assert sorted(list(sample.data.discounts.keys())) == [
            "agent_0",
            "agent_1",
            "agent_2",
        ]
        for agent_discount in sample.data.discounts.values():
            assert jnp.size(agent_discount) != 0
        assert sorted(list(sample.data.extras.keys())) == [
            "network_keys",
            "policy_info",
        ]
        assert sorted(list(sample.data.extras["network_keys"].keys())) == [
            "agent_0",
            "agent_1",
            "agent_2",
        ]
        assert sorted(list(sample.data.extras["policy_info"].keys())) == [
            "agent_0",
            "agent_1",
            "agent_2",
        ]
