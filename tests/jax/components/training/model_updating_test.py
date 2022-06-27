from types import SimpleNamespace
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
import pytest

from mava.components.jax.training import Batch
from mava.components.jax.training.model_updating import (
    MAPGEpochUpdate,
    MAPGMinibatchUpdate,
)
from mava.systems.jax.trainer import Trainer
from mava.types import OLT


def grad_fn(
    params,
    observations,
    actions,
    behavior_log_probs,
    target_values,
    advantages,
    behavior_values,
) -> Tuple:
    gradient = {
        "agent_0": jnp.array([5.0, 5.0, 5.0]),
        "agent_1": jnp.array([5.0, 5.0, 5.0]),
        "agent_2": jnp.array([5.0, 5.0, 5.0]),
    }
    agent_metrics = {}
    for agent_key, value in actions.items():
        agent_metrics[agent_key] = {}

    return (gradient, agent_metrics)


class MockTrainer(Trainer):
    def __init__(self):
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
            grad_fn=grad_fn,
            trainer_agents=trainer_agents,
            trainer_agent_net_keys=trainer_agent_net_keys,
            full_batch_size=2,
        )


@pytest.fixture
def mock_trainer() -> MockTrainer:
    return MockTrainer()


class MockOptimizer:
    def __init__(self) -> None:
        self.test = "Done"
        pass

    def init(self, params) -> str:
        return str(params)

    def update(self, gradient, opt_states):
        return (gradient, "opt_states_after_update")


@pytest.fixture
def mock_optimizer() -> MockOptimizer:
    return MockOptimizer()


@pytest.fixture
def fake_batch() -> Batch:
    batch = Batch(
        observations={
            "agent_0": OLT(
                observation=jnp.array([[0.1, 0.5, 0.7], [1.1, 1.5, 1.7]]),
                legal_actions=jnp.array([[0.1, 0.5, 0.7], [1.1, 1.5, 1.7]]),
                terminal=jnp.array([[0.1, 0.5, 0.7], [1.1, 1.5, 1.7]]),
            ),
            "agent_1": OLT(
                observation=jnp.array([[0.8, 0.3, 0.7], [1.8, 1.3, 1.7]]),
                legal_actions=jnp.array([[0.1, 0.5, 0.7], [1.1, 1.5, 1.7]]),
                terminal=jnp.array([[0.1, 0.5, 0.7], [1.1, 1.5, 1.7]]),
            ),
            "agent_2": OLT(
                observation=jnp.array([[0.9, 0.9, 0.8], [1.9, 1.9, 1.8]]),
                legal_actions=jnp.array([[0.1, 0.5, 0.7], [1.1, 1.5, 1.7]]),
                terminal=jnp.array([[0.1, 0.5, 0.7], [1.1, 1.5, 1.7]]),
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
def fake_state(
    mock_trainer: MockTrainer, fake_batch: Batch, mock_optimizer: MockOptimizer
) -> Any:
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


def test_on_training_utility_fns_empty_config_optimizer(
    mock_trainer: MockTrainer,
) -> None:
    mini_batch_update = MAPGMinibatchUpdate()
    mini_batch_update.config.optimizer = None
    mini_batch_update.on_training_utility_fns(trainer=mock_trainer)

    assert mock_trainer.store.optimizer != None
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
            jax.tree_map(
                lambda t: jnp.zeros_like(t, dtype=float),
                mock_trainer.store.networks["networks"][net_key].params,
            )
        )
        assert list(mock_trainer.store.opt_states[net_key][1][2]) == list(
            jax.tree_map(
                jnp.zeros_like, mock_trainer.store.networks["networks"][net_key].params
            )
        )
        assert mock_trainer.store.opt_states[net_key][2] == optax.EmptyState()

    assert mock_trainer.store.minibatch_update_fn


def test_on_training_utility_fns(
    mock_trainer: MockTrainer, mock_optimizer: MockOptimizer
) -> None:
    mini_batch_update = MAPGMinibatchUpdate()
    mini_batch_update.config.optimizer = mock_optimizer
    mini_batch_update.on_training_utility_fns(trainer=mock_trainer)

    assert mock_trainer.store.optimizer.test == "Done"

    for net_key in mock_trainer.store.networks["networks"].keys():
        assert mock_trainer.store.opt_states[net_key] == str(
            mock_trainer.store.networks["networks"][net_key].params
        )

    assert mock_trainer.store.minibatch_update_fn


def test_minibatch_update_fn(fake_state: Any) -> None:
    state = fake_state[0]
    mock_trainer = fake_state[1]
    carry = [state["params"], state["opt_states"]]

    (
        state["params"],
        state["opt_state"],
    ), metrics = mock_trainer.store.minibatch_update_fn(
        carry=carry, minibatch=state["batch"]
    )

    assert list(state["params"].keys()) == [
        "network_agent_0",
        "network_agent_1",
        "network_agent_2",
    ]
    assert list(state["params"]["network_agent_0"]) == [5.0, 5.0, 5.0]
    assert list(state["params"]["network_agent_1"]) == [6.0, 6.0, 6.0]
    assert list(state["params"]["network_agent_2"]) == [7.0, 7.0, 7.0]

    for net_key in state["opt_state"].keys():
        assert state["opt_state"][net_key] == "opt_states_after_update"

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
    mini_epoch_update = MAPGEpochUpdate()
    mini_epoch_update.on_training_utility_fns(trainer=mock_trainer)

    assert mock_trainer.store.num_epochs == mini_epoch_update.config.num_epochs
    assert (
        mock_trainer.store.num_minibatches == mini_epoch_update.config.num_minibatches
    )
    assert mock_trainer.store.epoch_update_fn


"""def test_epoch_update_fn(fake_state:Any)->None:
    mock_trainer= fake_state[1]
    mini_epoch_update= MAPGEpochUpdate()
    mini_epoch_update.on_training_utility_fns(trainer=mock_trainer)
    state= fake_state[0]

    carry= [state["random_key"], state["params"], state["opt_states"], state["batch"]]
    mock_trainer.store.epoch_update_fn(carry=carry, unused_t=None)"""
