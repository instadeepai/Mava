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

"""Losses components unit tests"""

from types import SimpleNamespace
from typing import Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
import pytest

from mava.components.training.losses import (
    HuberValueLoss,
    HuberValueLossConfig,
    MAPGTrustRegionClippingLossConfig,
    MAPGWithTrustRegionClippingLoss,
    SquaredErrorValueLoss,
)
from mava.systems.trainer import Trainer


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
        parameters: Dict[str, jnp.ndarray],
        observation_state: List[jnp.ndarray],
    ) -> jnp.ndarray:
        """Mock function to apply the network to training data"""
        if len(observation_state) == 2:
            # Recurrent case
            return observation_state[0][0], observation_state[1]  # type: ignore
        else:
            # Feedforward case
            return observation_state[0]  # type: ignore


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

    observation = SimpleNamespace(observation=jnp.array([[0.5], [0.5], [0.7], [0.2]]))
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
        num_minibatches=1,
        base_key=base_key,
        action_info=action_info,
        policy_info=policy_info,
    )

    mock_trainer = Trainer(store)

    # Set sample batch size
    mock_trainer.store.sample_batch_size = 1
    mock_trainer.store.sequence_length = 5

    return mock_trainer


@pytest.fixture
def mapg_loss() -> MAPGWithTrustRegionClippingLoss:  # noqa: E501
    """Creates an MAPG loss fixture with trust region and clipping"""
    test_loss = MAPGWithTrustRegionClippingLoss(
        config=MAPGTrustRegionClippingLossConfig(value_cost=0.5, entropy_cost=0.01)
    )
    return test_loss


def test_mapg_creation(
    mock_trainer: Trainer,
    mapg_loss: MAPGWithTrustRegionClippingLoss,  # noqa: E501
) -> None:
    """Test whether mapg functions are successfully created"""
    default_loss = SquaredErrorValueLoss()
    default_loss.on_training_utility_fns(trainer=mock_trainer)
    mapg_loss.on_training_loss_fns(trainer=mock_trainer)
    assert hasattr(mock_trainer.store, "policy_grad_fn")
    assert hasattr(mock_trainer.store, "critic_grad_fn")
    assert isinstance(
        mock_trainer.store.policy_grad_fn, Callable  # type:ignore
    )
    assert isinstance(
        mock_trainer.store.critic_grad_fn, Callable  # type:ignore
    )
    huber_loss = HuberValueLoss(config=HuberValueLossConfig(huber_delta=2.0))
    huber_loss.on_training_utility_fns(trainer=mock_trainer)
    mapg_loss.on_training_loss_fns(trainer=mock_trainer)
    assert hasattr(mock_trainer.store, "policy_grad_fn")
    assert hasattr(mock_trainer.store, "critic_grad_fn")
    assert isinstance(
        mock_trainer.store.policy_grad_fn, Callable  # type:ignore
    )
    assert isinstance(
        mock_trainer.store.critic_grad_fn, Callable  # type:ignore
    )
    assert huber_loss.config.huber_delta == 2


def test_mapg_loss(
    mock_trainer: Trainer,
    mapg_loss: MAPGWithTrustRegionClippingLoss,  # noqa: E501
) -> None:
    """Test whether mapg loss output is as expected"""
    default_loss = SquaredErrorValueLoss()
    default_loss.on_training_utility_fns(trainer=mock_trainer)
    mapg_loss.on_training_loss_fns(trainer=mock_trainer)
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

    # Test the recurrent loss code on higher advantage values
    states = [jnp.array([1, 2, 3, 4])]
    policy_states = {"agent_0": states, "agent_1": states, "agent_2": states}
    _, recurrent_policy_loss_info = policy_grad_fn(
        policy_params=mock_trainer.store.parameters,
        observations=mock_trainer.store.observations,
        actions=actions,
        behaviour_log_probs=behaviour_log_probs,
        advantages=advantages,
        policy_states=policy_states,
    )

    agent_0_policy_loss = recurrent_policy_loss_info["agent_0"]
    loss_entropy = agent_0_policy_loss["loss_entropy"]
    loss_policy = agent_0_policy_loss["loss_policy"]
    loss_policy_total = agent_0_policy_loss["policy_loss_total"]
    assert jnp.isclose(loss_entropy, -0.47500002)
    assert loss_policy_total == (loss_entropy * 0.01 + loss_policy)

    # Test the feedforward loss code on lower advantage values.
    policy_states = {"agent_0": None, "agent_1": None, "agent_2": None}  # type: ignore
    _, feedforward_policy_loss_info = policy_grad_fn(
        policy_params=mock_trainer.store.parameters,
        observations=mock_trainer.store.observations,
        actions=actions,
        behaviour_log_probs=behaviour_log_probs,
        advantages=low_advantages,
        policy_states=policy_states,
    )

    agent_0_policy_loss = recurrent_policy_loss_info["agent_0"]
    loss_entropy = agent_0_policy_loss["loss_entropy"]
    loss_policy = agent_0_policy_loss["loss_policy"]
    loss_policy_total = agent_0_policy_loss["policy_loss_total"]
    assert jnp.isclose(loss_entropy, -0.47500002)
    assert loss_policy_total == (loss_entropy * 0.01 + loss_policy)

    # Check if the loss is lower now.
    low_loss_policy = feedforward_policy_loss_info["agent_0"]["loss_policy"]
    assert low_loss_policy < loss_policy

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

    agent_0_critic_loss = critic_loss_info["agent_0"]
    loss_critic = agent_0_critic_loss["loss_critic"]
    low_agent_0_critic_loss = low_critic_loss_info["agent_0"]
    low_loss_critic = low_agent_0_critic_loss["loss_critic"]

    assert loss_critic == 4.5
    assert low_loss_critic < loss_critic


def test_mapg_huber_loss(
    mock_trainer: Trainer,
    mapg_loss: MAPGWithTrustRegionClippingLoss,  # noqa: E501
) -> None:
    """Test whether mapg huber loss output is as expected"""
    huber_loss = HuberValueLoss()
    huber_loss.on_training_utility_fns(trainer=mock_trainer)
    mapg_loss.on_training_loss_fns(trainer=mock_trainer)
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

    policy_states = {"agent_0": None, "agent_1": None, "agent_2": None}  # type: ignore
    _, policy_loss_info = policy_grad_fn(
        policy_params=mock_trainer.store.parameters,
        observations=mock_trainer.store.observations,
        actions=actions,
        behaviour_log_probs=behaviour_log_probs,
        advantages=advantages,
        policy_states=policy_states,
    )

    _, low_policy_loss_info = policy_grad_fn(
        policy_params=mock_trainer.store.parameters,
        observations=mock_trainer.store.observations,
        actions=actions,
        behaviour_log_probs=behaviour_log_probs,
        advantages=low_advantages,
        policy_states=policy_states,
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

    assert jnp.isclose(loss_entropy, -0.5)
    assert loss_critic == 1.25
    assert loss_policy_total == (loss_entropy * 0.01 + loss_policy)

    assert low_loss_policy < loss_policy
    assert low_loss_critic < loss_critic
