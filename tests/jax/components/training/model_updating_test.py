from mava.components.jax.training.model_updating import MAPGMinibatchUpdateConfig, MAPGMinibatchUpdate
from mava.systems.jax.trainer import Trainer
from mava.components.jax.training import Batch


import pytest
from types import SimpleNamespace
import optax
import jax.numpy as jnp

class MockTrainer(Trainer):
    def __init__(self):
        networks={
            "networks":{
                "network_0": SimpleNamespace(params=jnp.array([0.0, 0.0]) ),
                "network_1": SimpleNamespace(params=jnp.array([1.0, 1.0]) ),
                "network_2": SimpleNamespace(params=jnp.array([2.0, 2.0]) )
            }
        }
        self.store=SimpleNamespace(
            networks=networks
        )

@pytest.fixture
def mock_trainer()->MockTrainer:
    return MockTrainer()

class MockOptimizer():
    def __init__(self) -> None:
        self.test="Done"
        pass

    def init(self, params)->None:
        return str(params)

@pytest.fixture
def mock_optimizer()->MockOptimizer:
    return MockOptimizer()


def test_on_training_utility_fns_empty_config_optimizer( mock_trainer:MockTrainer)->None:
    mini_batch_update= MAPGMinibatchUpdate()
    mini_batch_update.config.optimizer=None
    mini_batch_update.on_training_utility_fns(mock_trainer)

    assert mock_trainer.store.optimizer!=None
    assert type(mock_trainer.store.optimizer)==optax.GradientTransformation
    
    for net_key in mock_trainer.store.networks["networks"].keys():
        assert mock_trainer.store.opt_states[net_key]!=None

 
    assert mock_trainer.store.minibatch_update_fn


def test_on_training_utility_fns(mock_trainer:MockTrainer, mock_optimizer: MockOptimizer)->None:
    mini_batch_update= MAPGMinibatchUpdate()
    mini_batch_update.config.optimizer=mock_optimizer
    mini_batch_update.on_training_utility_fns(mock_trainer)

    assert mock_trainer.store.optimizer.test=="Done"

    for net_key in mock_trainer.store.networks["networks"].keys():
        assert mock_trainer.store.opt_states[net_key]== str(
            mock_trainer.store.networks["networks"][net_key].params
        )
 
    assert mock_trainer.store.minibatch_update_fn


def test_minibatch_update_fn(mock_trainer: MockTrainer)->None:
    mini_batch_update= MAPGMinibatchUpdate()
    mini_batch_update.config.optimizer=None
    mini_batch_update.on_training_utility_fns(mock_trainer)
    


