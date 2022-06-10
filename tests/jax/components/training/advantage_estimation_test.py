# python3
# Copyright 2022 InstaDeep Ltd. All rights reserved.
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

from types import SimpleNamespace
from typing import Any, List, Tuple

import jax.numpy as jnp
import pytest

from mava.components.jax.training.advantage_estimation import GAE, GAEConfig
from mava.core_jax import SystemTrainer


def different_reward_values() -> List[Tuple]:
    """Create test case data"""

    return [
        (
            jnp.array(
                [
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                ]
            ),
            jnp.array(
                [
                    100.0,
                    0.0,
                    100.0,
                    100.0,
                ]
            ),
            jnp.array(
                [
                    0.99,
                    0.99,
                    0.99,
                    0.99,
                ]
            ),
            jnp.array(
                [
                    0.3,
                    0.3,
                    0.2,
                    0.3,
                ]
            ),
        )
    ]


def similar_reward_values() -> List[Tuple]:
    """Create test case data"""

    return [
        (
            jnp.array(
                [
                    0.5,
                    0.0,
                    0.8,
                    0.8,
                ]
            ),
            jnp.array(
                [
                    0.5,
                    0.0,
                    0.8,
                    0.8,
                ]
            ),
            jnp.array(
                [
                    0.99,
                    0.99,
                    0.99,
                    0.99,
                ]
            ),
            jnp.array(
                [
                    0.3,
                    0.3,
                    0.2,
                    0.3,
                ]
            ),
        )
    ]


class MockTrainer(SystemTrainer):
    """Abstract system trainer."""

    def __init__(
        self,
    ) -> None:
        """System trainer init"""

        # Simple namespace for assigning system executor attributes dynamically
        self.store = SimpleNamespace(gae_fn=None)

        self._inputs: Any

    def step(self) -> None:
        """Trainer forward and backward passes."""
        pass


def test_gae_creation() -> None:
    """Test whether gae function is successfully created"""

    test_gae = GAE()
    mock_trainer = MockTrainer()
    test_gae.on_training_utility_fns(trainer=mock_trainer)

    assert mock_trainer.store.gae_fn is not None


@pytest.mark.parametrize(
    "rewards_low,rewards_high,discounts,values", different_reward_values()
)
def test_gae_function_reward_clipping_different_rewards(
    rewards_low: jnp.ndarray,
    rewards_high: jnp.ndarray,
    discounts: jnp.ndarray,
    values: jnp.ndarray,
) -> None:
    """Test whether reward clipping in gae_advantages is working

    Done by verifying whether it is returning similar advantage
    and target values when given rewards with different scales.
    """

    test_gae = GAE(config=GAEConfig(max_abs_reward=1.0))
    mock_trainer = MockTrainer()
    test_gae.on_training_utility_fns(trainer=mock_trainer)

    gae_fn = mock_trainer.store.gae_fn

    advantages_high, target_values_high = gae_fn(
        rewards=rewards_high, discounts=discounts, values=values
    )

    advantages_low, target_values_low = gae_fn(
        rewards=rewards_low, discounts=discounts, values=values
    )

    assert jnp.array_equal(advantages_high, advantages_low)
    assert jnp.array_equal(target_values_high, target_values_low)


@pytest.mark.parametrize(
    "rewards_low,rewards_high,discounts,values", different_reward_values()
)
def test_gae_function_reward_not_clipping_different_rewards(
    rewards_low: jnp.ndarray,
    rewards_high: jnp.ndarray,
    discounts: jnp.ndarray,
    values: jnp.ndarray,
) -> None:
    """Test whether gae_advantages is working

    Done by verifying whether it is returning different advantage
    and target values when given rewards with different scales.
    """

    test_gae = GAE(config=GAEConfig())
    mock_trainer = MockTrainer()
    test_gae.on_training_utility_fns(trainer=mock_trainer)

    gae_fn = mock_trainer.store.gae_fn

    advantages_high, target_values_high = gae_fn(
        rewards=rewards_high, discounts=discounts, values=values
    )

    advantages_low, target_values_low = gae_fn(
        rewards=rewards_low, discounts=discounts, values=values
    )

    assert not jnp.array_equal(advantages_high, advantages_low)
    assert not jnp.array_equal(target_values_high, target_values_low)


@pytest.mark.parametrize(
    "rewards_1,rewards_2,discounts,values", similar_reward_values()
)
def test_gae_function_reward_clipping_similar_rewards(
    rewards_1: jnp.ndarray,
    rewards_2: jnp.ndarray,
    discounts: jnp.ndarray,
    values: jnp.ndarray,
) -> None:
    """Test whether reward clipping in gae_advantages is working

    Done by verifying whether it is returning similar advantage
    and target values when given identical rewards with ranges
    below the clipping threshold.
    """

    test_gae = GAE(config=GAEConfig(max_abs_reward=1.0))
    mock_trainer = MockTrainer()
    test_gae.on_training_utility_fns(trainer=mock_trainer)

    gae_fn = mock_trainer.store.gae_fn

    advantages_1, target_values_1 = gae_fn(
        rewards=rewards_1, discounts=discounts, values=values
    )

    advantages_2, target_values_2 = gae_fn(
        rewards=rewards_2, discounts=discounts, values=values
    )

    assert jnp.array_equal(advantages_1, advantages_2)
    assert jnp.array_equal(target_values_1, target_values_2)
