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

"""Optimisers unit tests"""

import optax
import pytest

from mava.components.building.optimisers import (
    DefaultOptimisers,
    DefaultOptimisersConfig,
    Optimisers,
)
from mava.core_jax import SystemBuilder
from mava.systems import Builder


@pytest.fixture
def test_builder() -> SystemBuilder:
    """Pytest fixture for system builder.

    Adds mock env specs and agent net keys to store.

    Returns:
        System builder with no components.
    """
    test_builder = Builder(components=[])
    return test_builder


@pytest.fixture
def default_optimisers_with_config() -> Optimisers:
    """Create default optimisers"""
    return DefaultOptimisers(
        config=DefaultOptimisersConfig(
            policy_learning_rate=1e-4,
            critic_learning_rate=1e-4,
            adam_epsilon=1e-4,
            max_gradient_norm=0.1,
        )
    )


@pytest.fixture
def default_optimisers_empty_config_optimisers() -> Optimisers:
    """Create default optimisers"""
    return DefaultOptimisers(
        config=DefaultOptimisersConfig(
            policy_optimiser=optax.chain(
                optax.clip_by_global_norm(40.0),
                optax.scale_by_adam(),
                optax.scale(-1e-4),
            ),
            critic_optimiser=optax.chain(
                optax.clip_by_global_norm(40.0),
                optax.scale_by_adam(),
                optax.scale(-1e-4),
            ),
        )
    )


def test_default_optimisers_with_config(
    default_optimisers_with_config: Optimisers, test_builder: SystemBuilder
) -> None:
    """Test default optmisers have been initialiased.

    Args:
        default_optimisers_with_config: Pytest fixture for default optimisers component,
        initliased with parameters.
        test_builder: Pytest fixture for test system builder

    Returns:
        None.
    """
    assert default_optimisers_with_config.config.policy_learning_rate == 1e-4
    assert default_optimisers_with_config.config.critic_learning_rate == 1e-4
    assert default_optimisers_with_config.config.adam_epsilon == 1e-4
    assert default_optimisers_with_config.config.max_gradient_norm == 0.1
    assert not default_optimisers_with_config.config.policy_optimiser
    assert not default_optimisers_with_config.config.policy_optimiser
    default_optimisers_with_config.on_building_init_start(test_builder)
    assert hasattr(test_builder.store, "policy_optimiser")
    assert hasattr(test_builder.store, "critic_optimiser")
    assert isinstance(test_builder.store.policy_optimiser, optax.GradientTransformation)
    assert isinstance(test_builder.store.critic_optimiser, optax.GradientTransformation)


def test_default_optimisers_empty_config_optimisers(
    default_optimisers_empty_config_optimisers: Optimisers, test_builder: SystemBuilder
) -> None:
    """Test default optmisers have been initialiased.

    Args:
        default_optimisers_empty_config_optimisers: Pytest fixture for
        default optimisers component.
        intialised externally
        test_builder: Pytest fixture for test system builder

    Returns:
        None.
    """
    default_optimisers_empty_config_optimisers.on_building_init_start(test_builder)
    assert (
        test_builder.store.policy_optimiser
        == default_optimisers_empty_config_optimisers.config.policy_optimiser
    )
    assert isinstance(test_builder.store.policy_optimiser, optax.GradientTransformation)
    assert (
        test_builder.store.critic_optimiser
        == default_optimisers_empty_config_optimisers.config.critic_optimiser
    )
    assert isinstance(test_builder.store.critic_optimiser, optax.GradientTransformation)
