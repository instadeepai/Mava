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

"""Tests for Step components jax-based Mava systems"""
import copy
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import pytest
import rlax

from mava.components.jax.training.step import (
    DefaultTrainerStep,
    MAPGWithTrustRegionStep,
    MAPGWithTrustRegionStepSeparateNetworks,
)
from mava.systems.jax.trainer import Trainer
from tests.jax.components.training.step_test_data import dummy_sample


def step_fn(sample: int) -> Dict[str, int]:
    """Step function to test DefaultTrainerStep component

    Args:
        sample: data sample

    Returns:
        Dictionary
    """
    return {"sample": sample}


def apply(params: Any, observations: Any) -> Tuple:
    """Apply function used to test step_fn"""
    return params, jnp.array([[0.1, 0.5], [0.1, 0.5], [0.1, 0.5]])


def critic_apply(params: Any, observations: Any) -> Tuple:
    """Apply function used to test step_fn"""
    return jnp.array([[0.1, 0.5], [0.1, 0.5], [0.1, 0.5]])


def gae_advantages(
    rewards: jnp.ndarray, discounts: jnp.ndarray, values: jnp.ndarray
) -> Tuple:
    """Uses GAE to compute advantages."""
    # Apply reward clipping.
    max_abs_reward = jnp.inf
    rewards = jnp.clip(rewards, -max_abs_reward, max_abs_reward)

    advantages = rlax.truncated_generalized_advantage_estimation(
        rewards[:-1], discounts[:-1], 0.95, values
    )
    advantages = jax.lax.stop_gradient(advantages)

    # Exclude the bootstrap value
    target_values = values[:-1] + advantages
    target_values = jax.lax.stop_gradient(target_values)

    return advantages, target_values


def epoch_update(carry: Tuple, unused_t: Tuple[()]) -> Tuple:
    """Performs model updates based on one epoch of data."""
    results = jax.tree_util.tree_map(lambda x: x + 1, carry)
    return results, {}


class MockTrainerLogger:
    """Mock of TrainerLogger to test DefaultTrainerStep component"""

    def __init__(self) -> None:
        """Initialize mock logger."""
        self.written = None

    def write(self, results: Any) -> None:
        """Mock write method for logger."""
        self.written = results


class MockParameterClient:
    """Mock of ParameterClient to test DefaultTrainerStep component"""

    def __init__(self) -> None:
        """Initialize mock parameter client"""
        self.params = {
            "trainer_steps": 0,
            "trainer_walltime": -1,
        }
        self.call_set_and_get_async = False

    def add_async(self, params: Any) -> None:
        """Mock add_async method."""
        self.params = params

    def set_and_get_async(self) -> None:
        """Mock set_and_get_async method."""
        self.call_set_and_get_async = True


class MockTrainer(Trainer):
    """Mock of Trainer"""

    def __init__(self) -> None:
        """Initialize mock trainer"""
        trainer_agent_net_keys = {
            "agent_0": "network_agent_0",
            "agent_1": "network_agent_1",
            "agent_2": "network_agent_2",
        }
        networks = {
            "networks": {
                "network_agent_0": SimpleNamespace(
                    params={"key": jnp.array([0.0, 0.0, 0.0])},
                    network=SimpleNamespace(apply=apply),
                ),
                "network_agent_1": SimpleNamespace(
                    params={"key": jnp.array([1.0, 1.0, 1.0])},
                    network=SimpleNamespace(apply=apply),
                ),
                "network_agent_2": SimpleNamespace(
                    params={"key": jnp.array([2.0, 2.0, 2.0])},
                    network=SimpleNamespace(apply=apply),
                ),
            }
        }
        opt_states = {
            "network_agent_0": 0,
            "network_agent_1": 1,
            "network_agent_2": 2,
        }
        store = SimpleNamespace(
            dataset_iterator=iter([1, 2, 3]),
            step_fn=step_fn,
            timestamp=1657703548.5225394,  # time.time() format
            trainer_parameter_client=MockParameterClient(),
            trainer_counts={"next_sample": 2},
            trainer_logger=MockTrainerLogger(),
            trainer_agent_net_keys=trainer_agent_net_keys,
            networks=networks,
            gae_fn=gae_advantages,
            opt_states=opt_states,
            key=jax.random.PRNGKey(5),
            epoch_update_fn=epoch_update,
            global_config=SimpleNamespace(
                num_minibatches=1, num_epochs=2, sample_batch_size=2, sequence_length=3
            ),
        )
        self.store = store


class MockTrainerSeparateNetworks(Trainer):
    """Mock of Trainer"""

    def __init__(self) -> None:
        """Initialize mock trainer"""
        trainer_agent_net_keys = {
            "agent_0": "network_agent_0",
            "agent_1": "network_agent_1",
            "agent_2": "network_agent_2",
        }
        networks = {
            "networks": {
                "network_agent_0": SimpleNamespace(
                    policy_params={"key": jnp.array([0.0, 0.0, 0.0])},
                    critic_params={"key": jnp.array([0.0, 0.0, 0.0])},
                    policy_network=SimpleNamespace(apply=apply),
                    critic_network=SimpleNamespace(apply=critic_apply),
                ),
                "network_agent_1": SimpleNamespace(
                    policy_params={"key": jnp.array([1.0, 1.0, 1.0])},
                    critic_params={"key": jnp.array([1.0, 1.0, 1.0])},
                    policy_network=SimpleNamespace(apply=apply),
                    critic_network=SimpleNamespace(apply=critic_apply),
                ),
                "network_agent_2": SimpleNamespace(
                    policy_params={"key": jnp.array([2.0, 2.0, 2.0])},
                    critic_params={"key": jnp.array([2.0, 2.0, 2.0])},
                    policy_network=SimpleNamespace(apply=apply),
                    critic_network=SimpleNamespace(apply=critic_apply),
                ),
            }
        }
        opt_states = {
            "network_agent_0": 0,
            "network_agent_1": 1,
            "network_agent_2": 2,
        }
        store = SimpleNamespace(
            dataset_iterator=iter([1, 2, 3]),
            step_fn=step_fn,
            timestamp=1657703548.5225394,  # time.time() format
            trainer_parameter_client=MockParameterClient(),
            trainer_counts={"next_sample": 2},
            trainer_logger=MockTrainerLogger(),
            trainer_agent_net_keys=trainer_agent_net_keys,
            networks=networks,
            gae_fn=gae_advantages,
            policy_opt_states=copy.copy(opt_states),
            critic_opt_states=copy.copy(opt_states),
            key=jax.random.PRNGKey(5),
            epoch_update_fn=epoch_update,
            global_config=SimpleNamespace(
                num_minibatches=1, num_epochs=2, sample_batch_size=2, sequence_length=3
            ),
        )
        self.store = store


@pytest.fixture
def mock_trainer() -> MockTrainer:
    """Build fixture from MockTrainer"""
    return MockTrainer()


@pytest.fixture
def mock_trainer_separate_networks() -> MockTrainerSeparateNetworks:
    """Build fixture from MockTrainerSeparateNetworks"""
    return MockTrainerSeparateNetworks()


##############
# SHARED TESTS
##############


def test_default_trainer_step_initiator() -> None:
    """Test constructor of DefaultTrainerStep component"""
    trainer_step = DefaultTrainerStep()
    assert trainer_step.config.random_key == 42


######################
# SINGLE NETWORK TESTS
######################


def test_on_training_step_with_timestamp(mock_trainer: Trainer) -> None:
    """Test on_training_step method from TrainerStep case of existing timestamp"""
    trainer_step = DefaultTrainerStep()
    old_timestamp = mock_trainer.store.timestamp
    trainer_step.on_training_step(trainer=mock_trainer)

    assert mock_trainer.store.timestamp > old_timestamp

    assert list(mock_trainer.store.trainer_parameter_client.params.keys()) == [
        "trainer_steps",
        "trainer_walltime",
    ]
    assert mock_trainer.store.trainer_parameter_client.params["trainer_steps"] == 1

    assert (
        int(mock_trainer.store.trainer_parameter_client.params["trainer_walltime"]) > 0
    )

    assert mock_trainer.store.trainer_parameter_client.call_set_and_get_async is True

    assert mock_trainer.store.trainer_logger.written == {"next_sample": 2, "sample": 1}


def test_on_training_step_without_timestamp(mock_trainer: Trainer) -> None:
    """Test on_training_step method from TrainerStep case of no timestamp"""
    trainer_step = DefaultTrainerStep()
    del mock_trainer.store.timestamp
    trainer_step.on_training_step(trainer=mock_trainer)

    assert mock_trainer.store.timestamp != 0

    assert list(mock_trainer.store.trainer_parameter_client.params.keys()) == [
        "trainer_steps",
        "trainer_walltime",
    ]
    assert mock_trainer.store.trainer_parameter_client.params["trainer_steps"] == 1
    assert (
        int(mock_trainer.store.trainer_parameter_client.params["trainer_walltime"]) == 0
    )

    assert mock_trainer.store.trainer_parameter_client.call_set_and_get_async is True

    assert mock_trainer.store.trainer_logger.written == {"next_sample": 2, "sample": 1}


def test_mapg_with_trust_region_step_initiator() -> None:
    """Test constructor of MAPGWITHTrustRegionStep component"""
    mapg_with_trust_region_step = MAPGWithTrustRegionStep()
    assert mapg_with_trust_region_step.config.discount == 0.99


def test_on_training_init_start(mock_trainer: MockTrainer) -> None:
    """Test on_training_init_start method from MAPGWITHTrustRegionStep component"""
    mapg_with_trust_region_step = MAPGWithTrustRegionStep()
    mapg_with_trust_region_step.on_training_init_start(trainer=mock_trainer)

    assert mock_trainer.store.full_batch_size == 4


def test_on_training_step_fn(mock_trainer: MockTrainer) -> None:
    """Test on_training_step_fn method from MAPGWITHTrustRegionStep component"""
    mapg_with_trust_region_step = MAPGWithTrustRegionStep()
    del mock_trainer.store.step_fn
    mapg_with_trust_region_step.on_training_step_fn(trainer=mock_trainer)

    assert callable(mock_trainer.store.step_fn)


def test_step(mock_trainer: MockTrainer) -> None:
    """Test step function"""
    mapg_with_trust_region_step = MAPGWithTrustRegionStep()
    del mock_trainer.store.step_fn
    mapg_with_trust_region_step.on_training_step_fn(trainer=mock_trainer)
    old_key = mock_trainer.store.base_key
    metrics = mock_trainer.store.step_fn(dummy_sample)

    # Check that metrics were correctly computed
    assert list(metrics.keys()) == [
        "norm_params",
        "observations_mean",
        "observations_std",
        "rewards_mean",
        "rewards_std",
    ]

    assert metrics["norm_params"] == 3.8729835

    assert metrics["observations_mean"] == 0.5667871

    assert metrics["observations_std"] == 0.104980744

    assert sorted(list(metrics["rewards_mean"].keys())) == [
        "agent_0",
        "agent_1",
        "agent_2",
    ]
    sorted_reward_mean = sorted(list(metrics["rewards_mean"].values()))
    assert round(float(sorted_reward_mean[0]), 3) == 0.000
    assert round(float(sorted_reward_mean[1]), 3) == 0.072
    assert round(float(sorted_reward_mean[2]), 3) == 0.083

    assert sorted(list(metrics["rewards_std"].keys())) == [
        "agent_0",
        "agent_1",
        "agent_2",
    ]
    sorted_reward_std = sorted(list(metrics["rewards_std"].values()))
    assert round(float(sorted_reward_std[0]), 3) == 0.000
    assert round(float(sorted_reward_std[1]), 3) == 0.070
    assert round(float(sorted_reward_std[2]), 3) == 0.077

    # check that trainer random key has been updated
    assert list(mock_trainer.store.base_key) != list(old_key)
    num_expected_update_steps = (
        mock_trainer.store.global_config.num_epochs
        * mock_trainer.store.global_config.num_minibatches
    )

    # check that network parameters and optimizer states were updated the correct
    # number of times
    for i, net_key in enumerate(mock_trainer.store.networks["networks"]):
        assert jnp.array_equal(
            mock_trainer.store.networks["networks"][net_key].params["key"],
            jnp.array(
                [
                    i + num_expected_update_steps,
                    i + num_expected_update_steps,
                    i + num_expected_update_steps,
                ]
            ),
        )

    assert mock_trainer.store.opt_states == {
        "network_agent_0": 0 + num_expected_update_steps,
        "network_agent_1": 1 + num_expected_update_steps,
        "network_agent_2": 2 + num_expected_update_steps,
    }


########################
# SEPARATE NETWORK TESTS
########################


def test_on_training_step_with_timestamp_separate_networks(
    mock_trainer_separate_networks: Trainer,
) -> None:
    """Test on_training_step method from TrainerStep case of existing timestamp"""
    trainer_step = DefaultTrainerStep()
    mock_trainer = mock_trainer_separate_networks

    old_timestamp = mock_trainer.store.timestamp
    trainer_step.on_training_step(trainer=mock_trainer)

    assert mock_trainer.store.timestamp > old_timestamp

    assert list(mock_trainer.store.trainer_parameter_client.params.keys()) == [
        "trainer_steps",
        "trainer_walltime",
    ]
    assert mock_trainer.store.trainer_parameter_client.params["trainer_steps"] == 1

    assert (
        int(mock_trainer.store.trainer_parameter_client.params["trainer_walltime"]) > 0
    )

    assert mock_trainer.store.trainer_parameter_client.call_set_and_get_async is True

    assert mock_trainer.store.trainer_logger.written == {"next_sample": 2, "sample": 1}


def test_on_training_step_without_timestamp_separate_networks(
    mock_trainer_separate_networks: Trainer,
) -> None:
    """Test on_training_step method from TrainerStep case of no timestamp"""
    trainer_step = DefaultTrainerStep()
    mock_trainer = mock_trainer_separate_networks

    del mock_trainer.store.timestamp
    trainer_step.on_training_step(trainer=mock_trainer)

    assert mock_trainer.store.timestamp != 0

    assert list(mock_trainer.store.trainer_parameter_client.params.keys()) == [
        "trainer_steps",
        "trainer_walltime",
    ]
    assert mock_trainer.store.trainer_parameter_client.params["trainer_steps"] == 1
    assert (
        int(mock_trainer.store.trainer_parameter_client.params["trainer_walltime"]) == 0
    )

    assert mock_trainer.store.trainer_parameter_client.call_set_and_get_async is True

    assert mock_trainer.store.trainer_logger.written == {"next_sample": 2, "sample": 1}


def test_mapg_with_trust_region_step_initiator_separate_networks() -> None:
    """Test constructor of MAPGWITHTrustRegionStep component"""
    mapg_with_trust_region_step = MAPGWithTrustRegionStepSeparateNetworks()
    assert mapg_with_trust_region_step.config.discount == 0.99


def test_on_training_init_start_separate_networks(
    mock_trainer_separate_networks: Trainer,
) -> None:
    """Test on_training_init_start method from \
        MAPGWITHTrustRegionStepSeparateNetworks component"""
    mapg_with_trust_region_step = MAPGWithTrustRegionStepSeparateNetworks()
    mock_trainer = mock_trainer_separate_networks

    mapg_with_trust_region_step.on_training_init_start(trainer=mock_trainer)

    assert mock_trainer.store.full_batch_size == 4


def test_on_training_step_fn_separate_networks(
    mock_trainer_separate_networks: Trainer,
) -> None:
    """Test on_training_step_fn method from \
        MAPGWITHTrustRegionStepSeparateNetworks component"""

    mapg_with_trust_region_step = MAPGWithTrustRegionStepSeparateNetworks()
    mock_trainer = mock_trainer_separate_networks

    del mock_trainer.store.step_fn
    mapg_with_trust_region_step.on_training_step_fn(trainer=mock_trainer)

    assert callable(mock_trainer.store.step_fn)


def test_step_separate_networks(mock_trainer_separate_networks: Trainer) -> None:
    """Test step function"""
    mapg_with_trust_region_step = MAPGWithTrustRegionStepSeparateNetworks()
    mock_trainer = mock_trainer_separate_networks
    del mock_trainer.store.step_fn

    mapg_with_trust_region_step.on_training_step_fn(trainer=mock_trainer)
    old_key = mock_trainer.store.base_key
    metrics = mock_trainer.store.step_fn(dummy_sample)

    # Check that metrics were correctly computed
    assert sorted(list(metrics.keys())) == [
        "norm_critic_params",
        "norm_policy_params",
        "observations_mean",
        "observations_std",
        "rewards_mean",
        "rewards_std",
    ]
    assert metrics["norm_policy_params"] == 3.8729835
    assert metrics["norm_critic_params"] == 3.8729835

    assert metrics["observations_mean"] == 0.5667871
    assert metrics["observations_std"] == 0.104980744

    assert sorted(list(metrics["rewards_mean"].keys())) == [
        "agent_0",
        "agent_1",
        "agent_2",
    ]
    sorted_reward_mean = sorted(list(metrics["rewards_mean"].values()))
    assert round(float(sorted_reward_mean[0]), 3) == 0.000
    assert round(float(sorted_reward_mean[1]), 3) == 0.072
    assert round(float(sorted_reward_mean[2]), 3) == 0.083

    assert sorted(list(metrics["rewards_std"].keys())) == [
        "agent_0",
        "agent_1",
        "agent_2",
    ]
    sorted_reward_std = sorted(list(metrics["rewards_std"].values()))
    assert round(float(sorted_reward_std[0]), 3) == 0.000
    assert round(float(sorted_reward_std[1]), 3) == 0.070
    assert round(float(sorted_reward_std[2]), 3) == 0.077

    # check that trainer random key has been updated
    assert list(mock_trainer.store.base_key) != list(old_key)
    num_expected_update_steps = (
        mock_trainer.store.global_config.num_epochs
        * mock_trainer.store.global_config.num_minibatches
    )

    # check that network parameters and optimizer states were updated the correct
    # number of times
    for i, net_key in enumerate(mock_trainer.store.networks["networks"]):
        assert jnp.array_equal(
            mock_trainer.store.networks["networks"][net_key].policy_params["key"],
            jnp.array(
                [
                    i + num_expected_update_steps,
                    i + num_expected_update_steps,
                    i + num_expected_update_steps,
                ]
            ),
        )

        assert jnp.array_equal(
            mock_trainer.store.networks["networks"][net_key].critic_params["key"],
            jnp.array(
                [
                    i + num_expected_update_steps,
                    i + num_expected_update_steps,
                    i + num_expected_update_steps,
                ]
            ),
        )

    assert mock_trainer.store.policy_opt_states == {
        "network_agent_0": 0 + num_expected_update_steps,
        "network_agent_1": 1 + num_expected_update_steps,
        "network_agent_2": 2 + num_expected_update_steps,
    }

    assert mock_trainer.store.critic_opt_states == {
        "network_agent_0": 0 + num_expected_update_steps,
        "network_agent_1": 1 + num_expected_update_steps,
        "network_agent_2": 2 + num_expected_update_steps,
    }
