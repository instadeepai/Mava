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
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
import pytest
import rlax

from mava.components.jax.training.step import (
    DefaultTrainerStep,
    MAPGWithTrustRegionStep,
)
from mava.systems.jax.trainer import Trainer
from tests.jax.components.training.step_test_data import DummySample


def step_fn(sample: int) -> Dict[str, int]:
    """step_fn to test DefaultTrainerStep component

    Args:
        sample
    Returns:
        Dictionary
    """
    return {"sample": sample}


def apply(params: Any, observations: Any) -> Tuple:
    """apply function used to test step_fn"""
    return params, jnp.array([[0.1, 0.5], [0.1, 0.5], [0.1, 0.5]])


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
    results = jax.tree_map(lambda x: x + 1, carry)
    return results, {}


class MockTrainerLogger:
    """Mock of TrainerLogger to test DefaultTrainerStep component"""

    def __init__(self) -> None:
        self.written = None

    def write(self, results: Any) -> None:
        self.written = results


class MockParameterClient:
    """Mock of ParameterClient to test DefaultTrainerStep component"""

    def __init__(self) -> None:
        self.params = {
            "trainer_steps": 0,
            "trainer_walltime": -1,
        }
        self.call_set_and_get_async = False

    def add_async(self, params: Any) -> None:
        self.params = params

    def set_and_get_async(self) -> None:
        self.call_set_and_get_async = True


class MockTrainer(Trainer):
    """Mock of Trainer"""

    def __init__(self) -> None:
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
            sample_batch_size=2,
            sequence_length=3,
            trainer_agent_net_keys=trainer_agent_net_keys,
            networks=networks,
            gae_fn=gae_advantages,
            opt_states=opt_states,
            key=jax.random.PRNGKey(5),
            num_minibatches=1,
            epoch_update_fn=epoch_update,
            num_epochs=2,
        )
        self.store = store


@pytest.fixture
def mock_trainer() -> MockTrainer:
    """Build fixture from MockTrainer"""
    return MockTrainer()


@pytest.fixture
def dummy_sample() -> DummySample:
    """Build fixture from DummySample"""
    return DummySample()


def test_default_trainer_step_initiator() -> None:
    """Test constructor of DefaultTrainerStep component"""
    trainer_step = DefaultTrainerStep()
    assert trainer_step.config.random_key == 42


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

    assert mock_trainer.store.trainer_parameter_client.call_set_and_get_async == True

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

    assert mock_trainer.store.trainer_parameter_client.call_set_and_get_async == True

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


def test_step(mock_trainer: MockTrainer, dummy_sample: DummySample) -> None:
    """Test step function"""
    mapg_with_trust_region_step = MAPGWithTrustRegionStep()
    del mock_trainer.store.step_fn
    mapg_with_trust_region_step.on_training_step_fn(trainer=mock_trainer)
    old_key = mock_trainer.store.key
    with jax.disable_jit():
        metrics = mock_trainer.store.step_fn(dummy_sample)

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

    assert list(metrics["rewards_mean"].keys()) == ["agent_0", "agent_1", "agent_2"]
    assert float(list(metrics["rewards_mean"].values())[0]) == 0.0
    assert float(list(metrics["rewards_mean"].values())[1]) == 0.08261970430612564
    assert float(list(metrics["rewards_mean"].values())[2]) == 0.07156813144683838

    assert list(metrics["rewards_std"].keys()) == ["agent_0", "agent_1", "agent_2"]
    assert float(list(metrics["rewards_std"].values())[0]) == 0.0
    assert float(list(metrics["rewards_std"].values())[1]) == 0.07048596441745758
    assert float(list(metrics["rewards_std"].values())[2]) == 0.07740379124879837

    assert list(mock_trainer.store.key) != list(old_key)

    assert sorted(list(mock_trainer.store.opt_states.keys())) == sorted(
        ["network_agent_0", "network_agent_1", "network_agent_2"]
    )
    assert mock_trainer.store.opt_states != {
        "network_agent_0": 0,
        "network_agent_1": 1,
        "network_agent_2": 2,
    }
