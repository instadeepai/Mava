from types import SimpleNamespace

# from typing import Callable, List, Tuple
from typing import Callable, Dict

import jax
import jax.numpy as jnp
import pytest

from mava.components.jax.training.losses import (
    MAPGTrustRegionClippingLossConfig,
    MAPGWithTrustRegionClippingLoss,
)
from mava.systems.jax.trainer import Trainer

class mock_net():
    def apply(observation, rng_key):
        """Function used in the networks.
        Returns:
            action_info and policy info
        """
        return jnp.array([-1.7, -1.7, -1.7, -1.7, -1.7]),jnp.array([-1.7, -1.7, -1.7, -1.7, -1.7])

def log_prob(distribution_params, actions):
    actions = jnp.array([-1.7, -1.7, -1.7, -1.7, -1.7])
    return jnp.array([-1.7, -1.7, -1.7, -1.7, -1.7])

def entropy(distribution_params):
    actions = jnp.array([-1.7, -1.7, -1.7, -1.7, -1.7])
    return jnp.array([-1.7, -1.7, -1.7, -1.7, -1.7])

@pytest.fixture
def mock_trainer() -> Trainer:
    """Creates mock trainer fixture"""
    
    observation = SimpleNamespace(observation=[0.1, 0.5, 0.7,0.2])
    observations = {"agent_0":observation,"agent_1":observation,"agent_2":observation}
    
    agent_net_keys = {
            "agent_0": "network_agent_0",
            "agent_1": "network_agent_1",
            "agent_2": "network_agent_2",
        }

    network = {
            "networks": {
                "network_agent": SimpleNamespace(network= mock_net, log_prob=log_prob,entropy=entropy),
                "network_agent": SimpleNamespace(network= mock_net, log_prob=log_prob,entropy=entropy),
                "network_agent": SimpleNamespace(network= mock_net, log_prob=log_prob,entropy=entropy),
            }
        }
    
    key = jax.random.PRNGKey(5)
    action_info = "action_info_test"
    policy_info = "policy_info_test"

    store = SimpleNamespace(
            is_evaluator=None,
            observations=observations,
            trainer_agents = ['agent_0','agent_1','agent_2'],
            networks=network,
            network = network,
            agent_net_keys=agent_net_keys,
            trainer_agent_net_keys={'agent_0': 'network_agent', 'agent_1': 'network_agent', 'agent_2': 'network_agent'},
            key=key,
            action_info=action_info,
            policy_info=policy_info,
        )
    
    mock_trainer = Trainer(store)
    #mock_trainer = Trainer(config=SimpleNamespace())
    #mock_trainer.store.grad_fn = None
    #mock_trainer.store.trainer_agents = ['agent_0','agent_1','agent_2']
    #mock_trainer.store.networks = {'networks':{'network_agent': "test"}}
    #mock_trainer.store.trainer_agent_net_keys = {'agent_0': 'network_agent', 'agent_1': 'network_agent', 'agent_2': 'network_agent'}

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


def mock_actions() -> Dict[str, jnp.ndarray]:
    """Test whether mapg loss config variables are of correct type"""
    actions = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
    return {"agent_0": actions, "agent_1": actions, "agent_2": actions}


def mock_behaviour_log_probs() -> Dict[str, jnp.ndarray]:
    """Test whether mapg loss config variables are of correct type"""
    actions = jnp.array([-1.7, -1.7, -1.7, -1.7, -1.7])
    return {"agent_0": actions, "agent_1": actions, "agent_2": actions}


def mock_target_values() -> Dict[str, jnp.ndarray]:
    """Test whether mapg loss config variables are of correct type"""
    actions = jnp.array([3.0, 3.0, 3.0, 3.0, 3.0])
    return {"agent_0": actions, "agent_1": actions, "agent_2": actions}


def mock_advantages() -> Dict[str, jnp.ndarray]:
    """Test whether mapg loss config variables are of correct type"""
    actions = jnp.array([2.0, 2.0, 2.0, 2.0, 2.0])
    return {"agent_0": actions, "agent_1": actions, "agent_2": actions}


def mock_behavior_values() -> Dict[str, jnp.ndarray]:
    """Test whether mapg loss config variables are of correct type"""
    actions = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
    return {"agent_0": actions, "agent_1": actions, "agent_2": actions}


def test_mapg_loss(
    mock_trainer: Trainer,
    mapg_trust_region_clipping_loss: MAPGWithTrustRegionClippingLoss,
) -> None:
    """Test whether mapg loss output is as expected"""
    # https://github.com/deepmind/rlax/blob/master/rlax/_src/policy_gradients_test.py
    mapg_trust_region_clipping_loss.on_training_loss_fns(trainer=mock_trainer)
    grad_fn = mock_trainer.store.grad_fn

    actions = mock_actions()
    behaviour_log_probs = mock_behaviour_log_probs()
    target_values = mock_target_values()
    advantages = mock_advantages()
    behavior_values = mock_behavior_values()
    p = {'network_agent': {'mlp/~/linear_0': {'w': jnp.array([[-0.28178176, -0.023124  , -0.22449262]]), 'b': jnp.array([0., 0., 0.])},
  'mlp/~/linear_1': {'w': jnp.array([[ 0.9697799 ,  0.48716304, -0.3466035 ],
          [ 0.83237547,  0.30383456,  0.4261682 ],
          [-0.17294468, -0.8475233 , -0.63647276]]),
   'b': jnp.array([0., 0., 0.])},
  'mlp/~/linear_2': {'w': jnp.array([[ 0.12169227, -0.1703035 ,  0.1628023 ],
          [-0.09848932,  0.36621055, -0.8668541 ],
          [-0.5537667 , -0.5817129 , -0.19160846]]),
   'b': jnp.array([0., 0., 0.])},
  'categorical_value_head/~/linear': {'w': jnp.array([[ 0.44159946, -1.147092  ,  0.1579899 ,  0.4251042 , -0.39571774],
          [-0.41283226, -0.695266  , -0.54058385, -0.8377876 ,  0.29822928],
          [-0.70897424,  1.1377946 , -0.45089862, -0.4194547 ,  0.08218037]]),
   'b': jnp.array([0., 0., 0., 0., 0.])},
  'categorical_value_head/~/linear_1': {'w': jnp.array([[-0.5159884 ],
          [ 0.71016085],
          [ 1.015577  ]]),
   'b': jnp.array([0.],)}}}
    observation = SimpleNamespace(observation=[0.1, 0.5, 0.7,0.2])
    observations = {"agent_0":observation,"agent_1":observation,"agent_2":observation}
    grads, loss_info = grad_fn(
        params=p,
        observations=observations,
        actions=actions,
        behaviour_log_probs=behaviour_log_probs,
        target_values=target_values,
        advantages=advantages,
        behavior_values=behavior_values,
    )

    print(grads)
    print(loss_info)
    mapg_trust_region_clipping_loss.on_training_loss_fns
