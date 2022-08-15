from types import SimpleNamespace
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import pytest

from mava.components.jax.training.losses import (
    MAPGTrustRegionClippingLossConfig,
    MAPGWithTrustRegionClippingLoss,
)
from mava.systems.jax.trainer import Trainer


class MockNet:
    """Creates a mock network for loss function"""

    @staticmethod
    def apply(
        parameters: Dict[str, jnp.ndarray], observation: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Mock function to apply the network to training data"""
        return observation, parameters["mlp/~/linear_0"]["w"]


def log_prob(distribution_params: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
    """Mock function to return fixed log probs"""
    return distribution_params + actions


def entropy(distribution_params: jnp.ndarray) -> jnp.ndarray:
    """Mock function to return fixed entropy"""

    return distribution_params


@pytest.fixture
def mock_trainer() -> Trainer:
    """Creates mock trainer fixture"""

    observation = SimpleNamespace(observation=jnp.array([0.5, 0.5, 0.7, 0.2]))
    observations = {
        "agent_0": observation,
        "agent_1": observation,
        "agent_2": observation,
    }

    agent_net_keys = {
        "agent_0": "network_agent",
        "agent_1": "network_agent",
        "agent_2": "network_agent",
    }

    network = {
        "networks": {
            "network_agent": SimpleNamespace(
                network=MockNet, log_prob=log_prob, entropy=entropy
            )
        }
    }

    key = jax.random.PRNGKey(5)
    action_info = "action_info_test"
    policy_info = "policy_info_test"

    parameters = {
        "network_agent": {
            "mlp/~/linear_0": {
                "w": jnp.array([[0.0, 0.0, 0.0, 0.0]]),
            }
        }
    }

    store = SimpleNamespace(
        is_evaluator=None,
        observations=observations,
        parameters=parameters,
        trainer_agents=["agent_0", "agent_1", "agent_2"],
        networks=network,
        network=network,
        agent_net_keys=agent_net_keys,
        trainer_agent_net_keys={
            "agent_0": "network_agent",
            "agent_1": "network_agent",
            "agent_2": "network_agent",
        },
        key=key,
        action_info=action_info,
        policy_info=policy_info,
    )

    mock_trainer = Trainer(store)

    return mock_trainer


@pytest.fixture
def mapg_trust_region_clipping_loss() -> MAPGWithTrustRegionClippingLoss:
    """Creates an MAPG loss fixture with trust region and clipping"""

    test_mapg = MAPGWithTrustRegionClippingLoss(
        config=MAPGTrustRegionClippingLossConfig(value_cost=0.5, entropy_cost=0.01)
    )

    return test_mapg


def test_mapg_creation(
    mock_trainer: Trainer,
    mapg_trust_region_clipping_loss: MAPGWithTrustRegionClippingLoss,
) -> None:
    """Test whether mapg function is successfully created"""

    mapg_trust_region_clipping_loss.on_training_loss_fns(trainer=mock_trainer)
    assert hasattr(mock_trainer.store, "grad_fn")
    assert isinstance(mock_trainer.store.grad_fn, Callable)  # type:ignore


def test_mapg_loss(
    mock_trainer: Trainer,
    mapg_trust_region_clipping_loss: MAPGWithTrustRegionClippingLoss,
) -> None:
    """Test whether mapg loss output is as expected"""
    mapg_trust_region_clipping_loss.on_training_loss_fns(trainer=mock_trainer)
    grad_fn = mock_trainer.store.grad_fn

    actions = {
        agent: jnp.array([1.0, 1.0, 1.0, 1.0])
        for agent in {"agent_0", "agent_1", "agent_2"}
    }
    behaviour_log_probs = {
        agent: jnp.array([-2.0, -2.0, -2.0, -2.0])
        for agent in {"agent_0", "agent_1", "agent_2"}
    }
    target_values = {
        agent: jnp.array([3.0, 3.0, 3.0, 3.0])
        for agent in {"agent_0", "agent_1", "agent_2"}
    }
    advantages = {
        agent: jnp.array([2.0, 2.0, 2.0, 2.0])
        for agent in {"agent_0", "agent_1", "agent_2"}
    }
    behavior_values = {
        agent: jnp.array([1.0, 1.0, 1.0, 1.0])
        for agent in {"agent_0", "agent_1", "agent_2"}
    }

    low_target_values = {
        agent: jnp.array([1.0, 1.0, 1.0, 1.0])
        for agent in {"agent_0", "agent_1", "agent_2"}
    }
    low_advantages = {
        agent: jnp.array([3.0, 3.0, 3.0, 3.0])
        for agent in {"agent_0", "agent_1", "agent_2"}
    }

    _, loss_info = grad_fn(
        params=mock_trainer.store.parameters,
        observations=mock_trainer.store.observations,
        actions=actions,
        behaviour_log_probs=behaviour_log_probs,
        target_values=target_values,
        advantages=advantages,
        behavior_values=behavior_values,
    )

    _, low_loss_info = grad_fn(
        params=mock_trainer.store.parameters,
        observations=mock_trainer.store.observations,
        actions=actions,
        behaviour_log_probs=behaviour_log_probs,
        target_values=low_target_values,
        advantages=low_advantages,
        behavior_values=behavior_values,
    )

    agent_0_loss = loss_info["agent_0"]
    loss_entropy = agent_0_loss["loss_entropy"]
    loss_value = agent_0_loss["loss_value"]
    loss_policy = agent_0_loss["loss_policy"]
    loss_total = agent_0_loss["loss_total"]

    low_agent_0_loss = low_loss_info["agent_0"]
    low_loss_policy = low_agent_0_loss["loss_policy"]

    assert loss_entropy == -0.47500002
    assert loss_value == 9
    assert loss_total == (loss_entropy * 0.01 + loss_policy + loss_value * 0.5)

    assert low_loss_policy < loss_policy
