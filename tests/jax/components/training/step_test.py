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

import time
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import optax
import jax
import jax.numpy as jnp
import pytest
from acme.jax import utils
from optax._src import numerics

from mava.components.jax.training.step import (
    DefaultTrainerStep,
    MAPGWithTrustRegionStep,
)
from mava.systems.jax.trainer import Trainer
from mava.types import OLT


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
    return params, jnp.array([[0.1, 0.5], [0.1, 0.5]])


def gae_advantages(
    rewards: jnp.ndarray, discounts: jnp.ndarray, values: jnp.ndarray
) -> Tuple:
    """Uses GAE to compute advantages."""
    return jnp.array([1]), jnp.array([1])


def epoch_update(carry: Tuple, unused_t: Tuple[()]) -> Tuple:
    """Performs model updates based on one epoch of data."""
    return carry, {}


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


class DummySample:
    """Dummy Sample for step function from MAPGWITHTrustRegionStep component"""

    def __init__(self) -> None:
        self.data = SimpleNamespace(
            observations={
                "agent_0": OLT(
                    observation=jnp.array(
                        [
                            [
                                [0.1, 0.5, 0.7, 0.1, 0.5, 0.7],
                                [0.1, 0.5, 0.7, 0.1, 0.56, 0.87],
                            ],
                            [
                                [0.1, 0.5, 0.7, 0.1, 0.5, 0.7],
                                [0.1, 0.5, 0.7, 0.13, 0.5, 0.78],
                            ],
                        ]
                    ),
                    legal_actions=jnp.array(
                        [
                            [
                                [1, 1, 1, 1, 0, 0],
                                [0, 1, 1, 0, 1, 0],
                            ],
                            [
                                [0, 0, 0, 1, 0, 1],
                                [1, 1, 1, 0, 1, 0],
                            ],
                        ]
                    ),
                    terminal=jnp.array(
                        [
                            [
                                [0, 0, 0, 0, 0, 1],
                                [0, 1, 0, 0, 0, 0],
                            ],
                            [
                                [0, 1, 1, 0, 1, 1],
                                [0, 0, 0, 1, 1, 0],
                            ],
                        ]
                    ),
                ),
                "agent_1": OLT(
                    observation=jnp.array(
                        [
                            [
                                [0.1, 0.53, 0.79, 0.8, 0.3, 0.7],
                                [0.1, 0.5, 0.7, 0.8, 0.3, 0.7],
                            ],
                            [
                                [0.1, 0.5, 0.7, 0.1, 0.5, 0.7],
                                [0.1, 0.5, 0.7, 0.1, 0.5, 0.7],
                            ],
                        ]
                    ),
                    legal_actions=jnp.array(
                        [
                            [
                                [1, 1, 0, 1, 0, 0],
                                [0, 1, 0, 0, 1, 0],
                            ],
                            [
                                [0, 1, 1, 1, 0, 1],
                                [1, 0, 1, 0, 1, 0],
                            ],
                        ]
                    ),
                    terminal=jnp.array(
                        [
                            [
                                [0, 0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 0, 0],
                            ],
                            [
                                [0, 1, 1, 0, 0, 1],
                                [0, 0, 0, 1, 1, 0],
                            ],
                        ]
                    ),
                ),
                "agent_2": OLT(
                    observation=jnp.array(
                        [
                            [
                                [0.1, 0.5, 0.7, 0.9, 0.9, 0.8],
                                [0.1, 0.5, 0.7, 0.9, 0.9, 0.8],
                            ],
                            [
                                [0.1, 0.5, 0.7, 0.1, 0.5, 0.7],
                                [0.1, 0.5, 0.7, 0.1, 0.5, 0.7],
                            ],
                        ]
                    ),
                    legal_actions=jnp.array(
                        [
                            [
                                [1, 1, 0, 1, 0, 0],
                                [0, 1, 1, 0, 0, 0],
                            ],
                            [
                                [0, 0, 0, 1, 0, 1],
                                [0, 0, 1, 0, 1, 0],
                            ],
                        ]
                    ),
                    terminal=jnp.array(
                        [
                            [
                                [1, 0, 0, 0, 0, 1],
                                [0, 1, 0, 0, 0, 0],
                            ],
                            [
                                [1, 1, 1, 0, 1, 1],
                                [0, 0, 0, 1, 1, 0],
                            ],
                        ]
                    ),
                ),
            },
            actions={
                "agent_0": jnp.array(
                    [
                        [[1, 2, 1, 1, 3, 1], [4, 2, 0, 0, 0, 0]],
                        [[1, 1, 1, 1, 2, 1], [1, 3, 0, 4, 0, 0]],
                    ]
                ),
                "agent_1": jnp.array(
                    [
                        [[2, 3, 3, 0, 0, 0], [4, 1, 1, 3, 1, 1]],
                        [[1, 1, 1, 1, 1, 1], [2, 0, 0, 0, 0, 0]],
                    ]
                ),
                "agent_2": jnp.array(
                    [
                        [[1, 2, 1, 4, 0, 1], [0, 0, 0, 3, 0, 0]],
                        [[1, 1, 1, 2, 0, 1], [0, 3, 0, 0, 0, 0]],
                    ]
                ),
            },
            rewards={
                "agent_0": jnp.array([[0.96, 0.34], [0.22, 0.34]]),
                "agent_1": jnp.array([[0.22, 0.33], [0.25, 0.84]]),
                "agent_2": jnp.array([[0.43, 0.45], [0.43, 0.25]]),
            },
            discounts={
                "agent_0": jnp.array([[1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 1, 1]]),
                "agent_1": jnp.array([[1, 0, 0, 1, 0, 0], [1, 0, 1, 1, 1, 1]]),
                "agent_2": jnp.array([[1, 0, 0, 1, 1, 1], [0, 0, 0, 1, 0, 1]]),
            },
            extras={
                "policy_info": {
                    "agent_0": jnp.array(
                        [
                            [
                                [-1.23, -1.3, -2.45, -1.45, -0.21, -1.34],
                                [-2.11, -0.55, -1.17, -1.11, -1.15, -1.57],
                            ],
                            [
                                [-1.53, -1.38, -1.45, -2.45, -0.21, -1.34],
                                [-2.11, -0.55, -1.17, -2.11, -1.85, -1.57],
                            ],
                        ]
                    ),
                    "agent_1": jnp.array(
                        [
                            [
                                [-1.23, -1.3, -2.45, -1.45, -0.21, -1.34],
                                [-2.11, -0.55, -1.17, -1.11, -1.15, -1.57],
                            ],
                            [
                                [-1.53, -1.38, -1.45, -2.45, -0.21, -1.34],
                                [-2.11, -0.55, -1.17, -2.11, -1.85, -1.57],
                            ],
                        ]
                    ),
                    "agent_2": jnp.array(
                        [
                            [
                                [-1.23, -1.3, -2.45, -1.45, -0.21, -1.34],
                                [-2.11, -0.55, -1.17, -1.11, -1.15, -1.57],
                            ],
                            [
                                [-1.53, -1.38, -1.45, -2.45, -0.21, -1.34],
                                [-2.11, -0.55, -1.17, -2.11, -1.85, -1.57],
                            ],
                        ]
                    ),
                }
            },
        )


@pytest.fixture
def mock_trainer() -> MockTrainer:
    """Build fixture from MockTrainer"""
    return MockTrainer()


@pytest.fixture
def dummy_sample() -> DummySample:
    """Build fixture from DUmmySample"""
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

    assert int(mock_trainer.store.timestamp) == int(time.time())

    assert list(mock_trainer.store.trainer_parameter_client.params.keys()) == [
        "trainer_steps",
        "trainer_walltime",
    ]
    assert mock_trainer.store.trainer_parameter_client.params["trainer_steps"] == 1
    elapsed_time = int(time.time() - old_timestamp)
    assert (
        int(mock_trainer.store.trainer_parameter_client.params["trainer_walltime"])
        == elapsed_time
    )

    assert mock_trainer.store.trainer_parameter_client.call_set_and_get_async == True

    assert mock_trainer.store.trainer_logger.written == {"next_sample": 2, "sample": 1}


def test_on_training_step_without_timestamp(mock_trainer: Trainer) -> None:
    """Test on_training_step method from TrainerStep case of no timestamp"""
    trainer_step = DefaultTrainerStep()
    del mock_trainer.store.timestamp
    trainer_step.on_training_step(trainer=mock_trainer)

    assert int(mock_trainer.store.timestamp) == int(time.time())

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

    assert (
        mock_trainer.store.full_batch_size
        == mock_trainer.store.sample_batch_size
        * (mock_trainer.store.sequence_length - 1)
    )


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

    updates = {
        net_key: mock_trainer.store.networks["networks"][net_key].params
        for net_key in mock_trainer.store.networks["networks"].keys()
    }
    
    assert metrics["norm_params"]==optax.global_norm(jax.tree_leaves(updates))

    observations = jax.tree_map(lambda x: x[:, :-1], dummy_sample.data.observations)
    assert metrics["observations_mean"] == 0.47703704
    
    assert metrics["observations_std"] == 0.18814814

    assert metrics["rewards_mean"] == {
        "agent_0":jnp.array([0.465]),
        "agent_1":jnp.array([0.41]),
        "agent_2":jnp.array([0.39]),
    }
    
    assert list(metrics["rewards_std"].keys()) == ['agent_0', 'agent_1', 'agent_2']
    assert float(list(metrics["rewards_std"].values())[0])== 0.2899568974971771
    assert float(list(metrics["rewards_std"].values())[1])== 0.25149551033973694
    assert float(list(metrics["rewards_std"].values())[2])== 0.08124038577079773

    random_key, _ = jax.random.split(old_key)
    assert list(mock_trainer.store.key) == list(random_key)

    assert mock_trainer.store.opt_states == {
        "network_agent_0": 0,
        "network_agent_1": 1,
        "network_agent_2": 2,
    }
