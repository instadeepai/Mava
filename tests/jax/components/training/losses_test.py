from types import SimpleNamespace
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import pytest

from mava.components.jax.training.losses import (
    MAPGTrustRegionClippingLossConfig,
    MAPGTrustRegionClippingPolicyLossConfig,
    MAPGTrustRegionClippingValueHuberLossConfig,
    MAPGTrustRegionClippingValueLossConfig,
    MAPGWithTrustRegionClippingLoss,
    MAPGWithTrustRegionClippingPolicyLoss,
    MAPGWithTrustRegionClippingValueHuberLoss,
    MAPGWithTrustRegionClippingValueLoss,
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


class MockPolicyNet:
    """Creates a mock policy network for loss function"""

    @staticmethod
    def apply(
        parameters: Dict[str, jnp.ndarray], observation: jnp.ndarray
    ) -> jnp.ndarray:
        """Mock function to apply the network to training data"""
        return observation


class MockCriticNet:
    """Creates a mock critic network for loss function"""

    @staticmethod
    def apply(
        parameters: Dict[str, jnp.ndarray], observation: jnp.ndarray
    ) -> jnp.ndarray:
        """Mock function to apply the network to training data"""
        return parameters["mlp/~/linear_0"]["w"]


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
        "network_agent": SimpleNamespace(
            policy_network=MockPolicyNet,
            critic_network=MockCriticNet,
            log_prob=log_prob,
            entropy=entropy,
        )
    }

    base_key = jax.random.PRNGKey(5)
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
        agent_net_keys=agent_net_keys,
        trainer_agent_net_keys={
            "agent_0": "network_agent",
            "agent_1": "network_agent",
            "agent_2": "network_agent",
        },
        base_key=base_key,
        action_info=action_info,
        policy_info=policy_info,
    )

    mock_trainer = Trainer(store)

    return mock_trainer


@pytest.fixture
def mock_trainer_separate() -> Trainer:
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
        "network_agent": SimpleNamespace(
            policy_network=MockPolicyNet,
            critic_network=MockCriticNet,
            log_prob=log_prob,
            entropy=entropy,
        )
    }

    base_key = jax.random.PRNGKey(5)
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
        agent_net_keys=agent_net_keys,
        trainer_agent_net_keys={
            "agent_0": "network_agent",
            "agent_1": "network_agent",
            "agent_2": "network_agent",
        },
        base_key=base_key,
        action_info=action_info,
        policy_info=policy_info,
    )

    mock_trainer_separate = Trainer(store)

    return mock_trainer_separate


@pytest.fixture
def mapg_trust_region_clipping_loss() -> MAPGWithTrustRegionClippingLoss:  # noqa: E501
    """Creates an MAPG loss fixture with trust region and clipping"""

    test_mapg = MAPGWithTrustRegionClippingLoss(
        config=MAPGTrustRegionClippingLossConfig(value_cost=0.5, entropy_cost=0.01)
    )

    return test_mapg


@pytest.fixture
def mapg_trust_region_policy_loss() -> MAPGWithTrustRegionClippingPolicyLoss:  # noqa: E501
    """Creates an MAPG loss fixture with trust region and clipping"""

    test_policy = MAPGWithTrustRegionClippingPolicyLoss(
        config=MAPGTrustRegionClippingPolicyLossConfig(entropy_cost=0.01)
    )

    return test_policy


@pytest.fixture
def mapg_trust_region_critic_loss() -> MAPGWithTrustRegionClippingValueLoss:  # noqa: E501
    """Creates an MAPG loss fixture with trust region and clipping"""

    test_critic = MAPGWithTrustRegionClippingValueLoss(
        config=MAPGTrustRegionClippingValueLossConfig(value_cost=0.5)
    )

    return test_critic


@pytest.fixture
def mapg_trust_region_critic_huber_loss() -> MAPGWithTrustRegionClippingValueHuberLoss:  # noqa: E501
    """Creates an MAPG loss fixture with trust region and clipping"""

    test_huber_critic = MAPGWithTrustRegionClippingValueHuberLoss(
        config=MAPGTrustRegionClippingValueHuberLossConfig(value_cost=0.5)
    )

    return test_huber_critic


# test mapg critic loss when policy and critic are separate
def test_mapg_separate_creation(
    mock_trainer: Trainer,
    mapg_trust_region_policy_loss: MAPGWithTrustRegionClippingPolicyLoss,  # noqa: E501
    mapg_trust_region_critic_loss: MAPGWithTrustRegionClippingValueLoss,  # noqa: E501
    mapg_trust_region_critic_huber_loss: MAPGWithTrustRegionClippingValueHuberLoss,  # noqa: E501
) -> None:
    """Test whether mapg functions are successfully created"""

    mapg_trust_region_policy_loss.on_training_loss_fns(trainer=mock_trainer)
    assert hasattr(mock_trainer.store, "policy_grad_fn")
    assert isinstance(
        mock_trainer.store.policy_grad_fn, Callable  # type:ignore
    )

    mapg_trust_region_critic_loss.on_training_loss_fns(trainer=mock_trainer)
    assert hasattr(mock_trainer.store, "critic_grad_fn")
    assert isinstance(
        mock_trainer.store.critic_grad_fn, Callable  # type:ignore
    )

    mapg_trust_region_critic_huber_loss.on_training_loss_fns(trainer=mock_trainer)
    assert hasattr(mock_trainer.store, "critic_grad_fn")
    assert isinstance(
        mock_trainer.store.critic_grad_fn, Callable  # type:ignore
    )


def test_mapg_creation(
    mock_trainer: Trainer,
    mapg_trust_region_clipping_loss: MAPGWithTrustRegionClippingLoss,  # noqa: E501
) -> None:
    """Test whether mapg functions are successfully created"""

    mapg_trust_region_clipping_loss.on_training_loss_fns(trainer=mock_trainer)
    assert hasattr(mock_trainer.store, "policy_grad_fn")
    assert hasattr(mock_trainer.store, "critic_grad_fn")
    assert isinstance(
        mock_trainer.store.policy_grad_fn, Callable  # type:ignore
    )
    assert isinstance(
        mock_trainer.store.critic_grad_fn, Callable  # type:ignore
    )


def test_critic_losses(
    mock_trainer_separate: Trainer,
    mapg_trust_region_critic_loss: MAPGWithTrustRegionClippingValueLoss,  # noqa: E501
    mapg_trust_region_critic_huber_loss: MAPGWithTrustRegionClippingValueHuberLoss,  # noqa: E501
) -> None:
    """Test whether critic loss output is as expected"""
    mapg_trust_region_critic_loss.on_training_loss_fns(trainer=mock_trainer_separate)
    critic_grad_fn = mock_trainer_separate.store.critic_grad_fn

    target_values = {
        agent: jnp.array([3.0, 3.0, 3.0, 3.0])
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

    _, critic_loss_info = critic_grad_fn(
        critic_params=mock_trainer_separate.store.parameters,
        observations=mock_trainer_separate.store.observations,
        target_values=target_values,
        behavior_values=behavior_values,
    )

    _, low_critic_loss_info = critic_grad_fn(
        critic_params=mock_trainer_separate.store.parameters,
        observations=mock_trainer_separate.store.observations,
        target_values=low_target_values,
        behavior_values=behavior_values,
    )

    agent_0_critic_loss = critic_loss_info["agent_0"]
    loss_critic = agent_0_critic_loss["loss_critic"]
    low_agent_0_critic_loss = low_critic_loss_info["agent_0"]
    low_loss_critic = low_agent_0_critic_loss["loss_critic"]
    assert low_loss_critic < loss_critic

    mapg_trust_region_critic_huber_loss.on_training_loss_fns(
        trainer=mock_trainer_separate
    )
    critic_grad_fn = mock_trainer_separate.store.critic_grad_fn

    _, critic_loss_info = critic_grad_fn(
        critic_params=mock_trainer_separate.store.parameters,
        observations=mock_trainer_separate.store.observations,
        target_values=target_values,
        behavior_values=behavior_values,
    )

    _, low_critic_loss_info = critic_grad_fn(
        critic_params=mock_trainer_separate.store.parameters,
        observations=mock_trainer_separate.store.observations,
        target_values=low_target_values,
        behavior_values=behavior_values,
    )

    agent_0_critic_loss = critic_loss_info["agent_0"]
    loss_critic = agent_0_critic_loss["loss_critic"]
    low_agent_0_critic_loss = low_critic_loss_info["agent_0"]
    low_loss_critic = low_agent_0_critic_loss["loss_critic"]
    assert low_loss_critic < loss_critic


def test_policy_losses(
    mock_trainer_separate: Trainer,
    mapg_trust_region_policy_loss: MAPGWithTrustRegionClippingValueLoss,  # noqa: E501
) -> None:
    """Test whether policy loss output is as expected"""
    mapg_trust_region_policy_loss.on_training_loss_fns(trainer=mock_trainer_separate)
    policy_grad_fn = mock_trainer_separate.store.policy_grad_fn

    actions = {
        agent: jnp.array([1.0, 1.0, 1.0, 1.0])
        for agent in {"agent_0", "agent_1", "agent_2"}
    }
    behaviour_log_probs = {
        agent: jnp.array([-2.0, -2.0, -2.0, -2.0])
        for agent in {"agent_0", "agent_1", "agent_2"}
    }

    advantages = {
        agent: jnp.array([2.0, 2.0, 2.0, 2.0])
        for agent in {"agent_0", "agent_1", "agent_2"}
    }

    low_advantages = {
        agent: jnp.array([3.0, 3.0, 3.0, 3.0])
        for agent in {"agent_0", "agent_1", "agent_2"}
    }

    _, policy_loss_info = policy_grad_fn(
        policy_params=mock_trainer_separate.store.parameters,
        observations=mock_trainer_separate.store.observations,
        actions=actions,
        behaviour_log_probs=behaviour_log_probs,
        advantages=advantages,
    )

    _, low_policy_loss_info = policy_grad_fn(
        policy_params=mock_trainer_separate.store.parameters,
        observations=mock_trainer_separate.store.observations,
        actions=actions,
        behaviour_log_probs=behaviour_log_probs,
        advantages=low_advantages,
    )

    agent_0_policy_loss = policy_loss_info["agent_0"]

    loss_entropy = agent_0_policy_loss["loss_entropy"]
    loss_policy = agent_0_policy_loss["loss_policy"]
    loss_policy_total = agent_0_policy_loss["policy_loss_total"]

    low_agent_0_policy_loss = low_policy_loss_info["agent_0"]
    low_loss_policy = low_agent_0_policy_loss["loss_policy"]

    assert jnp.isclose(loss_entropy, -0.47500002)
    assert loss_policy_total == (loss_entropy * 0.01 + loss_policy)

    assert low_loss_policy < loss_policy


def test_mapg_loss(
    mock_trainer: Trainer,
    mapg_trust_region_clipping_loss: MAPGWithTrustRegionClippingLoss,  # noqa: E501
) -> None:
    """Test whether mapg loss output is as expected"""
    mapg_trust_region_clipping_loss.on_training_loss_fns(trainer=mock_trainer)
    policy_grad_fn = mock_trainer.store.policy_grad_fn
    critic_grad_fn = mock_trainer.store.critic_grad_fn

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

    _, policy_loss_info = policy_grad_fn(
        policy_params=mock_trainer.store.parameters,
        observations=mock_trainer.store.observations,
        actions=actions,
        behaviour_log_probs=behaviour_log_probs,
        advantages=advantages,
    )

    _, low_policy_loss_info = policy_grad_fn(
        policy_params=mock_trainer.store.parameters,
        observations=mock_trainer.store.observations,
        actions=actions,
        behaviour_log_probs=behaviour_log_probs,
        advantages=low_advantages,
    )

    _, critic_loss_info = critic_grad_fn(
        critic_params=mock_trainer.store.parameters,
        observations=mock_trainer.store.observations,
        target_values=target_values,
        behavior_values=behavior_values,
    )

    _, low_critic_loss_info = critic_grad_fn(
        critic_params=mock_trainer.store.parameters,
        observations=mock_trainer.store.observations,
        target_values=low_target_values,
        behavior_values=behavior_values,
    )

    agent_0_policy_loss = policy_loss_info["agent_0"]
    agent_0_critic_loss = critic_loss_info["agent_0"]

    loss_entropy = agent_0_policy_loss["loss_entropy"]
    loss_critic = agent_0_critic_loss["loss_critic"]
    loss_policy = agent_0_policy_loss["loss_policy"]
    loss_policy_total = agent_0_policy_loss["policy_loss_total"]

    low_agent_0_policy_loss = low_policy_loss_info["agent_0"]
    low_loss_policy = low_agent_0_policy_loss["loss_policy"]

    low_agent_0_critic_loss = low_critic_loss_info["agent_0"]
    low_loss_critic = low_agent_0_critic_loss["loss_critic"]

    assert jnp.isclose(loss_entropy, -0.47500002)
    assert loss_policy_total == (loss_entropy * 0.01 + loss_policy)
    assert loss_critic == 4.5

    assert low_loss_policy < loss_policy
    assert low_loss_critic < loss_critic
