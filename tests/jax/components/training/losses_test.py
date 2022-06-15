from types import SimpleNamespace

# from typing import Callable, List, Tuple
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import pytest

from mava.components.jax.training.losses import (
    MAPGTrustRegionClippingLossConfig,
    MAPGWithTrustRegionClippingLoss,
)
from mava.systems.jax.trainer import Trainer


class mock_net:
    """Creates a mock network for loss function"""

    def apply(
        observation: jnp.array, rng_key: jnp.array
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Mock function to apply the network to training data"""
        return jnp.array([-1.7, -1.7, -1.7, -1.7, -1.7]), jnp.array(
            [-1.7, -1.7, -1.7, -1.7, -1.7]
        )


def log_prob(distribution_params: jnp.array, actions: jnp.array) -> jnp.ndarray:
    """Mock function to return fixed log probs"""

    log_probs = jnp.array([-1.7, -1.7, -1.7, -1.7, -1.7])

    return log_probs


def entropy(distribution_params: jnp.array) -> jnp.ndarray:
    """Mock function to return fixed entropy"""

    entropy = jnp.array([-1.7, -1.7, -1.7, -1.7, -1.7])

    return entropy


def mock_actions() -> Dict[str, jnp.ndarray]:
    """Returns set of mock actions"""
    actions = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
    return {"agent_0": actions, "agent_1": actions, "agent_2": actions}


def mock_behaviour_log_probs() -> Dict[str, jnp.ndarray]:
    """Returns set of mock_behaviour_log_probs"""
    actions = jnp.array([-1.7, -1.7, -1.7, -1.7, -1.7])
    return {"agent_0": actions, "agent_1": actions, "agent_2": actions}


def mock_target_values() -> Dict[str, jnp.ndarray]:
    """Returns set of mock_target_values"""
    actions = jnp.array([3.0, 3.0, 3.0, 3.0, 3.0])
    return {"agent_0": actions, "agent_1": actions, "agent_2": actions}


def mock_advantages() -> Dict[str, jnp.ndarray]:
    """Returns set of mock_advantages"""
    actions = jnp.array([2.0, 2.0, 2.0, 2.0, 2.0])
    return {"agent_0": actions, "agent_1": actions, "agent_2": actions}


def mock_behavior_values() -> Dict[str, jnp.ndarray]:
    """Returns set of mock_behavior_values"""
    actions = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
    return {"agent_0": actions, "agent_1": actions, "agent_2": actions}


@pytest.fixture
def mock_trainer() -> Trainer:
    """Creates mock trainer fixture"""

    observation = SimpleNamespace(observation=[0.1, 0.5, 0.7, 0.2])
    observations = {
        "agent_0": observation,
        "agent_1": observation,
        "agent_2": observation,
    }

    agent_net_keys = {
        "agent_0": "network_agent_0",
        "agent_1": "network_agent_1",
        "agent_2": "network_agent_2",
    }

    network = {
        "networks": {
            "network_agent": SimpleNamespace(
                network=mock_net, log_prob=log_prob, entropy=entropy
            )
        }
    }

    key = jax.random.PRNGKey(5)
    action_info = "action_info_test"
    policy_info = "policy_info_test"

    parameters = {
        "network_agent": {
            "mlp/~/linear_0": {
                "w": jnp.array([[0.0, 0.0, 0.0]]),
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


# @pytest.fixture
def mapg_trust_region_clipping_loss_config() -> MAPGTrustRegionClippingLossConfig:
    """Creates an MAPG loss config fixture with trust region and clipping"""

    test_mapg_config = MAPGTrustRegionClippingLossConfig()

    return test_mapg_config


@pytest.fixture
def mapg_trust_region_clipping_loss() -> MAPGWithTrustRegionClippingLoss:
    """Creates an MAPG loss fixture with trust region and clipping"""

    test_mapg = MAPGWithTrustRegionClippingLoss(
        config=MAPGTrustRegionClippingLossConfig()
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


def test_mapg_config_creation() -> None:
    """Test whether mapg loss config variables are of correct type"""

    mapg_config = mapg_trust_region_clipping_loss_config()

    assert isinstance(mapg_config.clipping_epsilon, float)
    assert isinstance(mapg_config.clip_value, bool)
    assert isinstance(mapg_config.entropy_cost, float)
    assert isinstance(mapg_config.value_cost, float)


def test_mapg_loss(
    mock_trainer: Trainer,
    mapg_trust_region_clipping_loss: MAPGWithTrustRegionClippingLoss,
) -> None:
    """Test whether mapg loss output is as expected"""
    mapg_trust_region_clipping_loss.on_training_loss_fns(trainer=mock_trainer)
    grad_fn = mock_trainer.store.grad_fn

    actions = mock_actions()
    behaviour_log_probs = mock_behaviour_log_probs()
    target_values = mock_target_values()
    advantages = mock_advantages()
    behavior_values = mock_behavior_values()

    _, loss_info = grad_fn(
        params=mock_trainer.store.parameters,
        observations=mock_trainer.store.observations,
        actions=actions,
        behaviour_log_probs=behaviour_log_probs,
        target_values=target_values,
        advantages=advantages,
        behavior_values=behavior_values,
    )

    agent_0_loss = loss_info["agent_0"]
    loss_entropy = agent_0_loss["loss_entropy"]
    loss_policy = agent_0_loss["loss_policy"]
    loss_total = agent_0_loss["loss_total"]
    loss_value = agent_0_loss["loss_value"]

    assert loss_entropy == 1.7
    assert loss_policy == -2.0
    assert loss_total == 9.062
    assert loss_value == 22.09
