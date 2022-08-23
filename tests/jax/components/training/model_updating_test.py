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

"""Tests for MAPGMinibatchUpdate and MAPGEpochUpdate class for Jax-based Mava systems"""

import copy
from types import SimpleNamespace
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
import pytest

from mava.components.jax.training import Batch
from mava.components.jax.training.model_updating import (
    MAPGEpochUpdate,
    MAPGEpochUpdateSeparateNetworks,
    MAPGMinibatchUpdate,
    MAPGMinibatchUpdateSeparateNetworks,
)
from mava.systems.jax.trainer import Trainer
from mava.types import OLT


def fake_ppo_grad_fn(
    params: Any,
    observations: Any,
    actions: Dict[str, jnp.ndarray],
    behaviour_log_probs: Dict[str, jnp.ndarray],
    target_values: Dict[str, jnp.ndarray],
    advantages: Dict[str, jnp.ndarray],
    behavior_values: Dict[str, jnp.ndarray],
) -> Tuple[Dict, Dict]:
    """Fake grad function to be used in MockTrainer

    Args:
        params
        observations
        actions
        behaviour_log_probs
        target_values
        advantages
        behavior_values

    Returns:
        gradient: fake gradient
        agent_metrics: fake metrics dictionary
    """

    gradient: Dict[str, Any] = {}
    agent_metrics: Dict[str, Any] = {}
    for agent_key in actions.keys():
        gradient[agent_key] = jnp.array([5.0, 5.0, 5.0])
        agent_metrics[agent_key] = {}

    return (gradient, agent_metrics)


def fake_ppo_policy_grad_fn(
    policy_params: Any,
    observations: Any,
    actions: Dict[str, jnp.ndarray],
    behaviour_log_probs: Dict[str, jnp.ndarray],
    advantages: Dict[str, jnp.ndarray],
) -> Tuple[Dict, Dict]:
    """Fake policy grad function to be used in MockTrainerSeparateNetworks

    Args:
        policy_params
        observations
        actions
        behaviour_log_probs
        target_values
        advantages
        behavior_values

    Returns:
        gradient: fake gradient
        agent_metrics: fake metrics dictionary
    """

    gradient: Dict[str, Any] = {}
    agent_metrics: Dict[str, Any] = {}
    for agent_key in actions.keys():
        gradient[agent_key] = jnp.array([5.0, 5.0, 5.0])
        agent_metrics[agent_key] = {}

    return (gradient, agent_metrics)


def fake_ppo_critic_grad_fn(
    critic_params: Any,
    observations: Any,
    target_values: Dict[str, jnp.ndarray],
    behavior_values: Dict[str, jnp.ndarray],
) -> Tuple[Dict, Dict]:
    """Fake critic grad function to be used in MockTrainerSeparateNetworks

    Args:
        critic_params
        observations
        target_values
        behavior_values

    Returns:
        gradient: fake gradient
        agent_metrics: fake metrics dictionary
    """

    gradient: Dict[str, Any] = {}
    agent_metrics: Dict[str, Any] = {}
    for agent_key in ["agent_0", "agent_1", "agent_2"]:
        gradient[agent_key] = jnp.array([5.0, 5.0, 5.0])
        agent_metrics[agent_key] = {}

    return (gradient, agent_metrics)


class MockTrainer(Trainer):
    """Mock trainer component"""

    def __init__(self) -> None:
        """Initialize mock trainer component."""
        networks = {
            "networks": {
                "network_agent_0": SimpleNamespace(params=jnp.array([0.0, 0.0, 0.0])),
                "network_agent_1": SimpleNamespace(params=jnp.array([1.0, 1.0, 1.0])),
                "network_agent_2": SimpleNamespace(params=jnp.array([2.0, 2.0, 2.0])),
            }
        }
        trainer_agents = {"agent_0", "agent_1", "agent_2"}
        trainer_agent_net_keys = {
            "agent_0": "network_agent_0",
            "agent_1": "network_agent_1",
            "agent_2": "network_agent_2",
        }
        self.store = SimpleNamespace(
            networks=networks,
            grad_fn=fake_ppo_grad_fn,
            trainer_agents=trainer_agents,
            trainer_agent_net_keys=trainer_agent_net_keys,
            full_batch_size=2,
        )


class MockTrainerSeparateNetworks(Trainer):
    """Mock trainer component for separate networks."""

    def __init__(self) -> None:
        """Initialize mock trainer component."""
        networks = {
            "networks": {
                "network_agent_0": SimpleNamespace(
                    policy_params=jnp.array([0.0, 0.0, 0.0]),
                    critic_params=jnp.array([0.0, 0.0, 0.0]),
                ),
                "network_agent_1": SimpleNamespace(
                    policy_params=jnp.array([1.0, 1.0, 1.0]),
                    critic_params=jnp.array([1.0, 1.0, 1.0]),
                ),
                "network_agent_2": SimpleNamespace(
                    policy_params=jnp.array([2.0, 2.0, 2.0]),
                    critic_params=jnp.array([2.0, 2.0, 2.0]),
                ),
            }
        }
        trainer_agents = {"agent_0", "agent_1", "agent_2"}
        trainer_agent_net_keys = {
            "agent_0": "network_agent_0",
            "agent_1": "network_agent_1",
            "agent_2": "network_agent_2",
        }
        self.store = SimpleNamespace(
            networks=networks,
            policy_grad_fn=fake_ppo_policy_grad_fn,
            critic_grad_fn=fake_ppo_critic_grad_fn,
            trainer_agents=trainer_agents,
            trainer_agent_net_keys=trainer_agent_net_keys,
            full_batch_size=2,
        )


@pytest.fixture
def mock_trainer() -> MockTrainer:
    """Create mock trainer component"""
    return MockTrainer()


@pytest.fixture
def mock_trainer_separate_networks() -> MockTrainerSeparateNetworks:
    """Create mock trainer component for separate networks"""
    return MockTrainerSeparateNetworks()


class MockOptimizer:
    """Mock optimizer configuration"""

    def __init__(self) -> None:
        """Initialize mock optimizer."""
        self.initialized = "Done"
        pass

    def init(self, params: Dict[str, Any]) -> list:
        """Mock optax optimzer init method"""
        return list(params)

    def update(
        self, gradient: Dict[str, Any], opt_states: Dict[str, Any]
    ) -> Tuple[Dict, str]:
        """Mock optax optimizer update method."""
        return (gradient, "opt_states_after_update")


@pytest.fixture
def mock_optimizer() -> MockOptimizer:
    """Create mock optimizer"""
    return MockOptimizer()


@pytest.fixture
def fake_batch() -> Batch:
    """Fake batch"""
    batch = Batch(
        observations={
            "agent_0": OLT(
                observation=jnp.array([[0.1, 0.5, 0.7], [1.1, 1.5, 1.7]]),
                legal_actions=jnp.array([[1], [1], [1], [1]]),
                terminal=jnp.array([[1], [1]]),
            ),
            "agent_1": OLT(
                observation=jnp.array([[0.8, 0.3, 0.7], [1.8, 1.3, 1.7]]),
                legal_actions=jnp.array([[1], [1], [1], [1]]),
                terminal=jnp.array([[1], [1]]),
            ),
            "agent_2": OLT(
                observation=jnp.array([[0.9, 0.9, 0.8], [1.9, 1.9, 1.8]]),
                legal_actions=jnp.array([[1], [1], [1], [1]]),
                terminal=jnp.array([[1], [1]]),
            ),
        },
        actions={
            "agent_0": jnp.array([[0.2, 0.6, 0.8], [1.2, 1.6, 1.8]]),
            "agent_1": jnp.array([[0.9, 0.4, 0.8], [1.9, 1.4, 1.8]]),
            "agent_2": jnp.array([[0.8, 0.8, 0.8], [1.8, 1.8, 1.8]]),
        },
        advantages=jnp.array([2.1, 2.5, 2.7]),
        target_values=jnp.array([3.1, 3.5, 3.7]),
        behavior_values=jnp.array([4.1, 4.5, 4.7]),
        behavior_log_probs=jnp.array([5.1, 5.5, 5.7]),
    )
    return batch


@pytest.fixture
def mock_state_and_trainer(
    mock_trainer: MockTrainer, fake_batch: Batch, mock_optimizer: MockOptimizer
) -> Any:
    """Fake state dictionary and mock trainer component

    Args:
        mock_trainer: mock trainer
        fake_batch: fake batch
        mock_optimizer: mock optimizer

    Returns:
        state dictionary: include random_key, params, opt_states and batch
        mock_trainer: mock trainer
    """
    mini_batch_update = MAPGMinibatchUpdate()
    mini_batch_update.config.optimizer = mock_optimizer
    mini_batch_update.on_training_utility_fns(trainer=mock_trainer)

    random_key = jax.random.PRNGKey(5)
    params = {
        "network_agent_0": mock_trainer.store.networks["networks"][
            "network_agent_0"
        ].params,
        "network_agent_1": mock_trainer.store.networks["networks"][
            "network_agent_1"
        ].params,
        "network_agent_2": mock_trainer.store.networks["networks"][
            "network_agent_2"
        ].params,
    }
    optstate = mock_trainer.store.opt_states

    return (
        {
            "random_key": random_key,
            "params": params,
            "opt_states": optstate,
            "batch": fake_batch,
        },
        mock_trainer,
    )


@pytest.fixture
def mock_state_and_trainer_separate_networks(
    mock_trainer_separate_networks: MockTrainerSeparateNetworks,
    fake_batch: Batch,
    mock_optimizer: MockOptimizer,
) -> Any:
    """Fake state dictionary and mock trainer component

    Args:
        mock_trainer_separate_networks: mock trainer
        fake_batch: fake batch
        mock_optimizer: mock optimizer
    Returns:
        state dictionary: with keys - random_key, params,
            policy_opt_states, critic_opt_states, batch
        mock_trainer: mock_trainer_separate_networks
    """
    mini_batch_update = MAPGMinibatchUpdateSeparateNetworks()
    mini_batch_update.config.policy_optimizer = copy.copy(mock_optimizer)
    mini_batch_update.config.critic_optimizer = copy.copy(mock_optimizer)
    mini_batch_update.on_training_utility_fns(trainer=mock_trainer_separate_networks)

    random_key = jax.random.PRNGKey(5)
    policy_params = {
        "network_agent_0": mock_trainer_separate_networks.store.networks["networks"][
            "network_agent_0"
        ].policy_params,
        "network_agent_1": mock_trainer_separate_networks.store.networks["networks"][
            "network_agent_1"
        ].policy_params,
        "network_agent_2": mock_trainer_separate_networks.store.networks["networks"][
            "network_agent_2"
        ].policy_params,
    }
    critic_params = {
        "network_agent_0": mock_trainer_separate_networks.store.networks["networks"][
            "network_agent_0"
        ].critic_params,
        "network_agent_1": mock_trainer_separate_networks.store.networks["networks"][
            "network_agent_1"
        ].critic_params,
        "network_agent_2": mock_trainer_separate_networks.store.networks["networks"][
            "network_agent_2"
        ].critic_params,
    }
    policy_opt_state = mock_trainer_separate_networks.store.policy_opt_states
    critic_opt_state = mock_trainer_separate_networks.store.critic_opt_states

    return (
        {
            "random_key": random_key,
            "policy_params": policy_params,
            "policy_opt_states": policy_opt_state,
            "critic_params": critic_params,
            "critic_opt_states": critic_opt_state,
            "batch": fake_batch,
        },
        mock_trainer_separate_networks,
    )


def mock_minibatch_update(carry: Tuple[Any, Any], minibatches: Any) -> Tuple[Any, Any]:
    """Mock minibatch update to test model_update_epoch in MAPGEpochUpdate class

    Args:
        carry: tuple include params and opt_states
        minibatches

    Returns:
        carry:same carry
        metrics: to test metrics value in test of model_update_epoch
    """
    metrics = {
        "agent_0": {
            "norm_updates": jnp.array([2, 2, 2]),
            "norm_grad": jnp.array([2, 2, 2]),
        },
        "agent_1": {
            "norm_updates": jnp.array([2, 2, 2]),
            "norm_grad": jnp.array([2, 2, 2]),
        },
        "agent_2": {
            "norm_updates": jnp.array([2, 2, 2]),
            "norm_grad": jnp.array([2, 2, 2]),
        },
    }
    return carry, metrics


def mock_minibatch_update_separate_networks(
    carry: Tuple[Any, Any], minibatches: Any
) -> Tuple[Any, Any]:
    """Mock minibatch update to test model_update_epoch in \
        MAPGEpochUpdateSeparateNetworks class

    Args:
        carry: tuple includings params and opt_states
        minibatches: place holder variable
    Returns:
        carry:same carry
        metrics: to test metrics value in test of model_update_epoch
    """
    metrics = {
        "agent_0": {
            "norm_policy_updates": jnp.array([2, 2, 2]),
            "norm_policy_grad": jnp.array([2, 2, 2]),
            "norm_critic_updates": jnp.array([2, 2, 2]),
            "norm_critic_grad": jnp.array([2, 2, 2]),
        },
        "agent_1": {
            "norm_policy_updates": jnp.array([2, 2, 2]),
            "norm_policy_grad": jnp.array([2, 2, 2]),
            "norm_critic_updates": jnp.array([2, 2, 2]),
            "norm_critic_grad": jnp.array([2, 2, 2]),
        },
        "agent_2": {
            "norm_policy_updates": jnp.array([2, 2, 2]),
            "norm_policy_grad": jnp.array([2, 2, 2]),
            "norm_critic_updates": jnp.array([2, 2, 2]),
            "norm_critic_grad": jnp.array([2, 2, 2]),
        },
    }
    return carry, metrics


@pytest.fixture
def mock_minibatch_update_fn() -> Callable:
    """Create mock minibatch_update function"""
    return mock_minibatch_update


@pytest.fixture
def mock_minibatch_update_fn_separate_networks() -> Callable:
    """Create mock minibatch_update function \
        for separate networks tests"""
    return mock_minibatch_update_separate_networks


######################
# SINGLE NETWORK TESTS
######################


def test_on_training_utility_fns_empty_config_optimizer(
    mock_trainer: MockTrainer,
) -> None:
    """Test on_training_utility_fns from \
        MAPGMinibatchUpdate with MAPGMinibatchUpdateConfig does not include optimizer

    Args:
        mock_trainer: Trainer
    """
    mini_batch_update = MAPGMinibatchUpdate()
    mini_batch_update.config.optimizer = None
    mini_batch_update.on_training_utility_fns(trainer=mock_trainer)

    assert mock_trainer.store.optimizer is not None
    assert isinstance(mock_trainer.store.optimizer, optax.GradientTransformation)

    assert list(mock_trainer.store.opt_states.keys()) == [
        "network_agent_0",
        "network_agent_1",
        "network_agent_2",
    ]
    for net_key in mock_trainer.store.networks["networks"].keys():
        assert mock_trainer.store.opt_states[net_key][0] == optax.EmptyState()
        assert isinstance(
            mock_trainer.store.opt_states[net_key][1], optax.ScaleByAdamState
        )
        assert mock_trainer.store.opt_states[net_key][1][0] == jnp.array([0])
        assert list(mock_trainer.store.opt_states[net_key][1][1]) == list(
            jax.tree_util.tree_map(
                lambda t: jnp.zeros_like(t, dtype=float),
                mock_trainer.store.networks["networks"][net_key].params,
            )
        )
        assert list(mock_trainer.store.opt_states[net_key][1][2]) == list(
            jax.tree_util.tree_map(
                jnp.zeros_like, mock_trainer.store.networks["networks"][net_key].params
            )
        )
        assert mock_trainer.store.opt_states[net_key][2] == optax.EmptyState()

    assert callable(mock_trainer.store.minibatch_update_fn)


def test_on_training_utility_fns(
    mock_trainer: MockTrainer, mock_optimizer: MockOptimizer
) -> None:
    """Test on_training_utility_fns from MAPGMinibatchUpdate

    Args:
        mock_trainer: Trainer
        mock_optimizer: Optimizer
    """
    mini_batch_update = MAPGMinibatchUpdate()
    mini_batch_update.config.optimizer = mock_optimizer
    mini_batch_update.on_training_utility_fns(trainer=mock_trainer)

    assert mock_trainer.store.optimizer.initialized == "Done"

    for net_key in mock_trainer.store.networks["networks"].keys():
        assert mock_trainer.store.opt_states[net_key] == list(
            mock_trainer.store.networks["networks"][net_key].params
        )

    assert callable(mock_trainer.store.minibatch_update_fn)


def test_minibatch_update_fn(
    mock_state_and_trainer: Tuple[Dict[str, Any], MockTrainer]
) -> None:
    """Test on_minibatch_update_fn

    Args:
        mock_state_and_trainer: tuple include fake state and mock trainer
    """
    state = mock_state_and_trainer[0]
    mock_trainer = mock_state_and_trainer[1]
    carry = [state["params"], state["opt_states"]]
    (
        new_params,
        new_opt_states,
    ), metrics = mock_trainer.store.minibatch_update_fn(
        carry=carry, minibatch=state["batch"]
    )

    assert list(new_params.keys()) == [
        "network_agent_0",
        "network_agent_1",
        "network_agent_2",
    ]
    assert list(new_params["network_agent_0"]) == [5.0, 5.0, 5.0]
    assert list(new_params["network_agent_1"]) == [6.0, 6.0, 6.0]
    assert list(new_params["network_agent_2"]) == [7.0, 7.0, 7.0]

    for net_key in new_opt_states.keys():
        assert new_opt_states[net_key] == "opt_states_after_update"

    assert list(metrics.keys()) == ["agent_0", "agent_1", "agent_2"]
    for agent in metrics.keys():
        assert list(metrics[agent].keys()) == ["norm_grad", "norm_updates"]
        assert metrics[agent]["norm_grad"] == optax.global_norm(
            jnp.array([5.0, 5.0, 5.0])
        )
        assert metrics[agent]["norm_updates"] == optax.global_norm(
            jnp.array([5.0, 5.0, 5.0])
        )


def test_on_training_utility_fns_epoch(mock_trainer: MockTrainer) -> None:
    """Test on_training_utility_fns from MAPGEpochUpdate

    Args:
        mock_trainer: trainer
    """
    mini_epoch_update = MAPGEpochUpdate()
    mini_epoch_update.on_training_utility_fns(trainer=mock_trainer)

    assert callable(mock_trainer.store.epoch_update_fn)


def test_epoch_update_fn(
    mock_state_and_trainer: Tuple[Dict[str, Any], MockTrainer],
    mock_minibatch_update_fn: Callable,
) -> None:
    """Test epoch_update_fn function

    Args:
        mock_state_and_trainer: tuple include fake state and mock trainer
        mock_minibatch_update_fn: minibatch update function
    """
    mock_trainer = mock_state_and_trainer[1]
    mini_epoch_update = MAPGEpochUpdate()
    mini_epoch_update.on_training_utility_fns(trainer=mock_trainer)
    mock_trainer.store.minibatch_update_fn = mock_minibatch_update_fn

    state = mock_state_and_trainer[0]
    # update params and opt_state to function in jax.lax.scan function
    state["params"] = jnp.array([0, 0, 0])
    state["opt_states"] = jnp.array([1, 1, 1])

    carry = [state["random_key"], state["params"], state["opt_states"], state["batch"]]
    (
        new_key,
        new_params,
        new_opt_states,
        batch,
    ), metrics = mock_trainer.store.epoch_update_fn(carry=carry, unused_t=None)

    assert list(new_key) == list(jax.random.split(state["random_key"])[0])

    assert list(new_params) == [0, 0, 0]

    assert list(new_opt_states) == [1, 1, 1]

    assert batch == state["batch"]

    assert list(metrics.keys()) == ["agent_0", "agent_1", "agent_2"]
    for agent in metrics.keys():
        assert list(metrics[agent].keys()) == ["norm_grad", "norm_updates"]
        assert list(metrics[agent]["norm_grad"][0]) == [2, 2, 2]
        assert list(metrics[agent]["norm_updates"][0]) == [2, 2, 2]


########################
# SEPARATE NETWORK TESTS
########################


def test_on_training_utility_fns_empty_config_optimizer_sep_netwoks(
    mock_trainer_separate_networks: MockTrainerSeparateNetworks,
) -> None:
    """Test on_training_utility_fns from \
        MAPGMinibatchUpdate with MAPGMinibatchUpdateConfig does not include optimizer

    Args:
        mock_trainer_separate_networks: Trainer
    """
    mini_batch_update = MAPGMinibatchUpdateSeparateNetworks()
    mini_batch_update.config.policy_optimizer = None
    mini_batch_update.config.critic_optimizer = None
    mini_batch_update.on_training_utility_fns(trainer=mock_trainer_separate_networks)

    assert mock_trainer_separate_networks.store.policy_optimizer is not None
    assert isinstance(
        mock_trainer_separate_networks.store.policy_optimizer,
        optax.GradientTransformation,
    )
    assert mock_trainer_separate_networks.store.critic_optimizer is not None
    assert isinstance(
        mock_trainer_separate_networks.store.critic_optimizer,
        optax.GradientTransformation,
    )

    assert list(mock_trainer_separate_networks.store.policy_opt_states.keys()) == [
        "network_agent_0",
        "network_agent_1",
        "network_agent_2",
    ]
    assert list(mock_trainer_separate_networks.store.critic_opt_states.keys()) == [
        "network_agent_0",
        "network_agent_1",
        "network_agent_2",
    ]

    for net_key in mock_trainer_separate_networks.store.networks["networks"].keys():
        assert (
            mock_trainer_separate_networks.store.policy_opt_states[net_key][0]
            == optax.EmptyState()
        )
        assert (
            mock_trainer_separate_networks.store.critic_opt_states[net_key][0]
            == optax.EmptyState()
        )

        assert isinstance(
            mock_trainer_separate_networks.store.policy_opt_states[net_key][1],
            optax.ScaleByAdamState,
        )
        assert isinstance(
            mock_trainer_separate_networks.store.critic_opt_states[net_key][1],
            optax.ScaleByAdamState,
        )

        assert mock_trainer_separate_networks.store.policy_opt_states[net_key][1][
            0
        ] == jnp.array([0])
        assert mock_trainer_separate_networks.store.critic_opt_states[net_key][1][
            0
        ] == jnp.array([0])

        assert list(
            mock_trainer_separate_networks.store.policy_opt_states[net_key][1][1]
        ) == list(
            jax.tree_map(
                lambda t: jnp.zeros_like(t, dtype=float),
                mock_trainer_separate_networks.store.networks["networks"][
                    net_key
                ].policy_params,
            )
        )
        assert list(
            mock_trainer_separate_networks.store.critic_opt_states[net_key][1][1]
        ) == list(
            jax.tree_map(
                lambda t: jnp.zeros_like(t, dtype=float),
                mock_trainer_separate_networks.store.networks["networks"][
                    net_key
                ].critic_params,
            )
        )

        assert list(
            mock_trainer_separate_networks.store.policy_opt_states[net_key][1][2]
        ) == list(
            jax.tree_map(
                jnp.zeros_like,
                mock_trainer_separate_networks.store.networks["networks"][
                    net_key
                ].policy_params,
            )
        )
        assert list(
            mock_trainer_separate_networks.store.critic_opt_states[net_key][1][2]
        ) == list(
            jax.tree_map(
                jnp.zeros_like,
                mock_trainer_separate_networks.store.networks["networks"][
                    net_key
                ].critic_params,
            )
        )

        assert (
            mock_trainer_separate_networks.store.policy_opt_states[net_key][2]
            == optax.EmptyState()
        )
        assert (
            mock_trainer_separate_networks.store.critic_opt_states[net_key][2]
            == optax.EmptyState()
        )

    assert callable(mock_trainer_separate_networks.store.minibatch_update_fn)


def test_on_training_utility_fns_separate_networks(
    mock_trainer_separate_networks: MockTrainerSeparateNetworks,
    mock_optimizer: MockOptimizer,
) -> None:
    """Test on_training_utility_fns from MAPGMinibatchUpdate

    Args:
        mock_trainer_separate_networks: Trainer
        mock_optimizer: Optimizer
    """
    mini_batch_update = MAPGMinibatchUpdateSeparateNetworks()
    mini_batch_update.config.policy_optimizer = copy.copy(mock_optimizer)
    mini_batch_update.config.critic_optimizer = copy.copy(mock_optimizer)
    mini_batch_update.on_training_utility_fns(trainer=mock_trainer_separate_networks)

    assert mock_trainer_separate_networks.store.policy_optimizer.initialized == "Done"
    assert mock_trainer_separate_networks.store.critic_optimizer.initialized == "Done"

    for net_key in mock_trainer_separate_networks.store.networks["networks"].keys():
        assert mock_trainer_separate_networks.store.policy_opt_states[net_key] == list(
            mock_trainer_separate_networks.store.networks["networks"][
                net_key
            ].policy_params
        )
        assert mock_trainer_separate_networks.store.critic_opt_states[net_key] == list(
            mock_trainer_separate_networks.store.networks["networks"][
                net_key
            ].critic_params
        )

    assert callable(mock_trainer_separate_networks.store.minibatch_update_fn)


def test_minibatch_update_fn_separate_networks(
    mock_state_and_trainer_separate_networks: Tuple[
        Dict[str, Any], MockTrainerSeparateNetworks
    ]
) -> None:
    """Test on_minibatch_update_fn

    Args:
        mock_state_and_trainer_separate_networks: tuple
            include fake state and mock trainer
    """
    state = mock_state_and_trainer_separate_networks[0]
    mock_trainer = mock_state_and_trainer_separate_networks[1]
    carry = [
        state["policy_params"],
        state["critic_params"],
        state["policy_opt_states"],
        state["critic_opt_states"],
    ]
    (
        new_policy_params,
        new_critic_params,
        new_policy_opt_states,
        new_critic_opt_states,
    ), metrics = mock_trainer.store.minibatch_update_fn(
        carry=carry, minibatch=state["batch"]
    )

    assert list(new_policy_params.keys()) == [
        "network_agent_0",
        "network_agent_1",
        "network_agent_2",
    ]
    assert list(new_critic_params.keys()) == [
        "network_agent_0",
        "network_agent_1",
        "network_agent_2",
    ]

    assert list(new_policy_params["network_agent_0"]) == [5.0, 5.0, 5.0]
    assert list(new_policy_params["network_agent_1"]) == [6.0, 6.0, 6.0]
    assert list(new_policy_params["network_agent_2"]) == [7.0, 7.0, 7.0]
    assert list(new_critic_params["network_agent_0"]) == [5.0, 5.0, 5.0]
    assert list(new_critic_params["network_agent_1"]) == [6.0, 6.0, 6.0]
    assert list(new_critic_params["network_agent_2"]) == [7.0, 7.0, 7.0]

    for net_key in new_policy_opt_states.keys():
        assert new_policy_opt_states[net_key] == "opt_states_after_update"
    for net_key in new_critic_opt_states.keys():
        assert new_critic_opt_states[net_key] == "opt_states_after_update"

    assert sorted(list(metrics.keys())) == ["agent_0", "agent_1", "agent_2"]
    for agent in metrics.keys():
        assert list(metrics[agent].keys()) == [
            "norm_policy_grad",
            "norm_policy_updates",
            "norm_critic_grad",
            "norm_critic_updates",
        ]
        assert metrics[agent]["norm_policy_grad"] == optax.global_norm(
            jnp.array([5.0, 5.0, 5.0])
        )
        assert metrics[agent]["norm_policy_updates"] == optax.global_norm(
            jnp.array([5.0, 5.0, 5.0])
        )
        assert metrics[agent]["norm_critic_grad"] == optax.global_norm(
            jnp.array([5.0, 5.0, 5.0])
        )
        assert metrics[agent]["norm_critic_updates"] == optax.global_norm(
            jnp.array([5.0, 5.0, 5.0])
        )


def test_on_training_utility_fns_epoch_separate_networks(
    mock_trainer_separate_networks: MockTrainerSeparateNetworks,
) -> None:
    """Test on_training_utility_fns from MAPGEpochUpdateSeparateNetworks

    Args:
        mock_trainer_separate_networks: trainer
    """
    mini_epoch_update = MAPGEpochUpdateSeparateNetworks()
    mini_epoch_update.on_training_utility_fns(trainer=mock_trainer_separate_networks)

    assert callable(mock_trainer_separate_networks.store.epoch_update_fn)


def test_epoch_update_fn_separate_networks(
    mock_state_and_trainer_separate_networks: Tuple[
        Dict[str, Any], MockTrainerSeparateNetworks
    ],
    mock_minibatch_update_fn_separate_networks: Callable,
) -> None:
    """Test epoch_update_fn function for separate networks case

    Args:
        mock_state_and_trainer_separate_networks: tuple including \
            fake state and mock trainer
        mock_minibatch_update_fn_separate_networks: minibatch \
            update function
    """
    mock_trainer = mock_state_and_trainer_separate_networks[1]
    mini_epoch_update = MAPGEpochUpdateSeparateNetworks()
    mini_epoch_update.on_training_utility_fns(trainer=mock_trainer)
    mock_trainer.store.minibatch_update_fn = mock_minibatch_update_fn_separate_networks

    state = mock_state_and_trainer_separate_networks[0]
    # update params and opt_state to function in jax.lax.scan function
    state["policy_params"] = jnp.array([0, 0, 0])
    state["critic_params"] = jnp.array([0, 0, 0])
    state["policy_opt_states"] = jnp.array([1, 1, 1])
    state["critic_opt_states"] = jnp.array([1, 1, 1])

    carry = [
        state["random_key"],
        state["policy_params"],
        state["critic_params"],
        state["policy_opt_states"],
        state["critic_opt_states"],
        state["batch"],
    ]
    (
        new_key,
        new_policy_params,
        new_critic_params,
        new_policy_opt_states,
        new_critic_opt_states,
        batch,
    ), metrics = mock_trainer.store.epoch_update_fn(carry=carry, unused_t=None)

    assert list(new_key) == list(jax.random.split(state["random_key"])[0])

    assert list(new_policy_params) == [0, 0, 0]
    assert list(new_critic_params) == [0, 0, 0]

    assert list(new_policy_opt_states) == [1, 1, 1]
    assert list(new_critic_opt_states) == [1, 1, 1]

    assert batch == state["batch"]

    assert sorted(list(metrics.keys())) == ["agent_0", "agent_1", "agent_2"]
    for agent in metrics.keys():
        assert sorted(list(metrics[agent].keys())) == [
            "norm_critic_grad",
            "norm_critic_updates",
            "norm_policy_grad",
            "norm_policy_updates",
        ]
        assert list(metrics[agent]["norm_policy_grad"][0]) == [2, 2, 2]
        assert list(metrics[agent]["norm_policy_updates"][0]) == [2, 2, 2]
        assert list(metrics[agent]["norm_critic_grad"][0]) == [2, 2, 2]
        assert list(metrics[agent]["norm_critic_updates"][0]) == [2, 2, 2]
