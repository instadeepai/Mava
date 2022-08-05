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

import jax
import jax.numpy as jnp
import optax
import pytest
import rlax
from acme.jax import utils
from jax.numpy import array, float32, int32
from optax._src import numerics
from reverb import ReplaySample

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
    def __init__(self) -> None:
        self.data = SimpleNamespace(
            observations={
                "agent_0": OLT(
                    observation=array(
                        [
                            [
                                [
                                    0.0,
                                    0.0,
                                    0.8539885,
                                    -0.518841,
                                    0.0,
                                    -1.1304581,
                                    0.13541625,
                                    -0.8415354,
                                    0.8974394,
                                    -0.53403145,
                                    0.541699,
                                    0.030427,
                                    0.5862218,
                                    -1.7108788,
                                    0.96656066,
                                ],
                                [
                                    0.0,
                                    -0.0,
                                    0.8539885,
                                    -0.518841,
                                    0.02,
                                    -1.1304581,
                                    0.13541625,
                                    -0.8415354,
                                    0.84743947,
                                    -0.58403146,
                                    0.541699,
                                    0.030427,
                                    0.5862218,
                                    -1.7108788,
                                    0.96656066,
                                ],
                                [
                                    0.0,
                                    -0.0,
                                    0.8539885,
                                    -0.518841,
                                    0.04,
                                    -1.1304581,
                                    0.13541625,
                                    -0.7915354,
                                    0.80993944,
                                    -0.6215314,
                                    0.541699,
                                    0.030427,
                                    0.5862218,
                                    -1.7108788,
                                    0.96656066,
                                ],
                            ],
                            [
                                [
                                    0.11865234,
                                    0.65267944,
                                    1.0183928,
                                    -0.11464486,
                                    0.2,
                                    -1.2948624,
                                    -0.2687799,
                                    -0.9584569,
                                    0.4132104,
                                    -0.3758324,
                                    -0.09247538,
                                    -0.1339773,
                                    0.18202564,
                                    -1.8752831,
                                    0.5623645,
                                ],
                                [
                                    0.08898926,
                                    -0.01049042,
                                    1.0272918,
                                    -0.1156939,
                                    0.22,
                                    -1.3037614,
                                    -0.26773086,
                                    -1.0292267,
                                    0.4842679,
                                    -0.36538208,
                                    -0.13393201,
                                    -0.14287622,
                                    0.18307468,
                                    -1.884182,
                                    0.56341356,
                                ],
                                [
                                    0.06674194,
                                    0.4921322,
                                    1.033966,
                                    -0.06648068,
                                    0.24,
                                    -1.3104355,
                                    -0.3169441,
                                    -1.0823039,
                                    0.487561,
                                    -0.30754435,
                                    -0.21502449,
                                    -0.14955041,
                                    0.13386145,
                                    -1.8908563,
                                    0.5142003,
                                ],
                            ],
                        ],
                        dtype=float32,
                    ),
                    legal_actions=array(
                        [
                            [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                            [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                        ]
                    ),
                    terminal=array(
                        [[[0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0]]], dtype=float32
                    ),
                ),
                "agent_1": OLT(
                    observation=array(
                        [
                            [
                                [
                                    0.0,
                                    0.0,
                                    0.01245314,
                                    0.37859842,
                                    0.0,
                                    0.87196237,
                                    -0.31121764,
                                    0.8415354,
                                    -0.8974394,
                                    0.30750394,
                                    -0.35574046,
                                    -0.28892276,
                                    -0.7620232,
                                    -0.86934346,
                                    0.06912122,
                                ],
                                [
                                    -0.0,
                                    -0.5,
                                    0.01245314,
                                    0.3285984,
                                    0.02,
                                    0.87196237,
                                    -0.26121765,
                                    0.8415354,
                                    -0.84743947,
                                    0.25750396,
                                    -0.30574045,
                                    -0.28892276,
                                    -0.7120232,
                                    -0.86934346,
                                    0.11912122,
                                ],
                                [
                                    0.5,
                                    -0.375,
                                    0.06245314,
                                    0.29109842,
                                    0.04,
                                    0.82196236,
                                    -0.22371764,
                                    0.7915354,
                                    -0.80993944,
                                    0.17000394,
                                    -0.26824045,
                                    -0.33892277,
                                    -0.6745232,
                                    -0.9193434,
                                    0.15662122,
                                ],
                            ],
                            [
                                [
                                    -0.82494366,
                                    0.26677936,
                                    0.05993592,
                                    0.29856554,
                                    0.2,
                                    0.8244796,
                                    -0.23118475,
                                    0.9584569,
                                    -0.4132104,
                                    0.5826245,
                                    -0.50568575,
                                    -0.33640555,
                                    -0.68199027,
                                    -0.9168262,
                                    0.14915411,
                                ],
                                [
                                    -0.6187078,
                                    0.7000845,
                                    -0.00193486,
                                    0.368574,
                                    0.22,
                                    0.8863504,
                                    -0.3011932,
                                    1.0292267,
                                    -0.4842679,
                                    0.6638445,
                                    -0.6181999,
                                    -0.27453476,
                                    -0.7519987,
                                    -0.85495543,
                                    0.07914566,
                                ],
                                [
                                    -0.46403083,
                                    0.5250634,
                                    -0.04833794,
                                    0.42108032,
                                    0.24,
                                    0.93275344,
                                    -0.35369954,
                                    1.0823039,
                                    -0.487561,
                                    0.77475953,
                                    -0.70258546,
                                    -0.22813168,
                                    -0.8045051,
                                    -0.8085523,
                                    0.02663932,
                                ],
                            ],
                        ],
                        dtype=float32,
                    ),
                    legal_actions=array(
                        [
                            [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                            [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                        ]
                    ),
                    terminal=array(
                        [[[0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0]]], dtype=float32
                    ),
                ),
                "agent_2": OLT(
                    observation=array(
                        [
                            [
                                [
                                    0.0,
                                    0.0,
                                    0.31995708,
                                    0.02285798,
                                    0.0,
                                    -1.1768473,
                                    0.42486167,
                                    -0.30750394,
                                    0.35574046,
                                    0.53403145,
                                    -0.541699,
                                    0.56445843,
                                    0.0445228,
                                    -0.5964267,
                                    -0.40628275,
                                ],
                                [
                                    -0.5,
                                    -0.0,
                                    0.2699571,
                                    0.02285798,
                                    0.02,
                                    -1.1268474,
                                    0.42486167,
                                    -0.25750396,
                                    0.30574045,
                                    0.58403146,
                                    -0.541699,
                                    0.61445844,
                                    0.0445228,
                                    -0.5464267,
                                    -0.40628275,
                                ],
                                [
                                    -0.375,
                                    -0.0,
                                    0.23245709,
                                    0.02285798,
                                    0.04,
                                    -1.0893474,
                                    0.42486167,
                                    -0.17000394,
                                    0.26824045,
                                    0.6215314,
                                    -0.541699,
                                    0.65195847,
                                    0.0445228,
                                    -0.5089267,
                                    -0.40628275,
                                ],
                            ],
                            [
                                [
                                    0.25799003,
                                    -0.56674236,
                                    0.6425604,
                                    -0.20712024,
                                    0.2,
                                    -1.4994507,
                                    0.6548399,
                                    -0.5826245,
                                    0.50568575,
                                    0.3758324,
                                    0.09247538,
                                    0.24185511,
                                    0.274501,
                                    -0.91903,
                                    -0.17630453,
                                ],
                                [
                                    0.19349252,
                                    -0.42505676,
                                    0.66190964,
                                    -0.2496259,
                                    0.22,
                                    -1.5187999,
                                    0.69734555,
                                    -0.6638445,
                                    0.6181999,
                                    0.36538208,
                                    0.13393201,
                                    0.22250587,
                                    0.31700668,
                                    -0.9383793,
                                    -0.13379885,
                                ],
                                [
                                    0.64511937,
                                    -0.31879258,
                                    0.7264216,
                                    -0.28150517,
                                    0.24,
                                    -1.5833119,
                                    0.7292248,
                                    -0.77475953,
                                    0.70258546,
                                    0.30754435,
                                    0.21502449,
                                    0.15799393,
                                    0.34888595,
                                    -1.0028912,
                                    -0.1019196,
                                ],
                            ],
                        ],
                        dtype=float32,
                    ),
                    legal_actions=array(
                        [
                            [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                            [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                        ]
                    ),
                    terminal=array(
                        [[[0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0]]], dtype=float32
                    ),
                ),
            },
            actions={
                "agent_0": array([[3, 2, 0], [4, 0, 3]]),
                "agent_1": array([[1, 0, 3], [0, 2, 0]]),
                "agent_2": array([[0, 0, 4], [3, 4, 2]]),
            },
            rewards={
                "agent_0": array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float32),
                "agent_1": array(
                    [
                        [0.0897511, 0.14813632, 0.19152136],
                        [0.06387269, 0.00243677, 0.0],
                    ],
                    dtype=float32,
                ),
                "agent_2": array(
                    [[0.0, 0.0, 0.0], [0.0897511, 0.14813632, 0.19152136]],
                    dtype=float32,
                ),
            },
            discounts={
                "agent_0": array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=float32),
                "agent_1": array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=float32),
                "agent_2": array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=float32),
            },
            start_of_episode=array([[True, False, False], [False, False, False]]),
            extras={
                "network_keys": {
                    "agent_0": array([[0, 0, 0], [0, 0, 0]], dtype=int32),
                    "agent_1": array([[0, 0, 0], [0, 0, 0]], dtype=int32),
                    "agent_2": array([[0, 0, 0], [0, 0, 0]], dtype=int32),
                },
                "policy_info": {
                    "agent_0": array(
                        [
                            [-1.5010276, -1.5574824, -1.7098966],
                            [-1.6839617, -1.8447837, -1.4597069],
                        ],
                        dtype=float32,
                    ),
                    "agent_1": array(
                        [
                            [-1.6333038, -1.7833046, -1.4482918],
                            [-1.6957064, -1.4800832, -1.66526],
                        ],
                        dtype=float32,
                    ),
                    "agent_2": array(
                        [
                            [-1.4521754, -1.4560769, -1.8592778],
                            [-1.4220893, -1.78906, -1.5569873],
                        ],
                        dtype=float32,
                    ),
                },
            },
        )


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

    assert metrics["norm_params"] == optax.global_norm(jax.tree_leaves(updates))

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

    random_key, _ = jax.random.split(old_key)
    assert list(mock_trainer.store.key) == list(random_key)

    assert mock_trainer.store.opt_states == {
        "network_agent_0": 0,
        "network_agent_1": 1,
        "network_agent_2": 2,
    }
