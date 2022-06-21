from typing import Tuple
from mava.components.jax.training.model_updating import MAPGMinibatchUpdate, MAPGEpochUpdate
from mava.systems.jax.trainer import Trainer
from mava.components.jax.training import Batch

from typing import Any, Dict, Tuple
import pytest
from types import SimpleNamespace
import optax
import jax.numpy as jnp
import jax

def grad_fn(params, observations, actions, behavior_log_probs, target_values, advantages, behavior_values)-> Tuple:
    gradient={}
    agent_metrics={}
    for agent_key, value in actions.items():
        gradient[agent_key]=value[0][:]
        agent_metrics[agent_key]={}

    return (gradient, agent_metrics)


class MockTrainer(Trainer):
    def __init__(self):
        networks={
            "networks":{
                "network_agent_0": SimpleNamespace(params=jnp.array([0.0, 0.0, 0.0]) ),
                "network_agent_1": SimpleNamespace(params=jnp.array([1.0, 1.0, 1.0]) ),
                "network_agent_2": SimpleNamespace(params=jnp.array([2.0, 2.0, 2.0]) )
            }
        }
        trainer_agents={
            "agent_0",
            "agent_1",
            "agent_2"
        }
        trainer_agent_net_keys={
            "agent_0": "network_agent_0",
            "agent_1": "network_agent_1",
            "agent_2": "network_agent_2"
        }
        self.store=SimpleNamespace(
            networks=networks,
            grad_fn=grad_fn,
            trainer_agents=trainer_agents,
            trainer_agent_net_keys=trainer_agent_net_keys,
            full_batch_size= 2
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

@pytest.fixture
def fake_batch()->Batch:
    batch=Batch(
        observations={
                "agent_0": SimpleNamespace(observation=jnp.array(
                    [
                        [0.1, 0.5, 0.7],
                        [1.1, 1.5, 1.7]
                    ])),
                "agent_1": SimpleNamespace(observation=jnp.array(
                    [
                        [0.8, 0.3, 0.7],
                        [1.8, 1.3, 1.7]
                    ])),
                "agent_2": SimpleNamespace(observation=jnp.array(
                    [
                        [0.9, 0.9, 0.8],
                        [1.9, 1.9, 1.8]
                    ]))
            },
        actions={
                "agent_0": jnp.array(
                    [
                        [0.2, 0.6, 0.8],
                        [1.2, 1.6, 1.8]
                    ]),
                "agent_1": jnp.array(
                    [
                        [0.9, 0.4, 0.8],
                        [1.9, 1.4, 1.8]
                    ]),
                "agent_2": jnp.array(
                    [
                        [0.8, 0.8, 0.8],
                        [1.8, 1.8, 1.8]
                    ])
            },
        advantages= jnp.array([2.1, 2.5, 2.7]),
        target_values= jnp.array([3.1, 3.5, 3.7]),
        behavior_values= jnp.array([4.1, 4.5, 4.7]),
        behavior_log_probs= jnp.array([5.1, 5.5, 5.7])
    )
    return batch


@pytest.fixture
def fake_state(mock_trainer: MockTrainer, fake_batch: Batch)->Any:
    mini_batch_update= MAPGMinibatchUpdate()
    mini_batch_update.config.optimizer=None
    mini_batch_update.on_training_utility_fns(trainer=mock_trainer)

    random_key= jax.random.PRNGKey(5)
    params = {
        "network_agent_0": mock_trainer.store.networks["networks"]["network_agent_0"].params,
        "network_agent_1": mock_trainer.store.networks["networks"]["network_agent_1"].params,
        "network_agent_2": mock_trainer.store.networks["networks"]["network_agent_2"].params
    }
    optstate=  mock_trainer.store.opt_states

    return ({
        "random_key": random_key,
        "params": params,
        "opt_states": optstate,
        "batch": fake_batch
    }, mock_trainer)



def test_on_training_utility_fns_empty_config_optimizer( mock_trainer:MockTrainer)->None:
    mini_batch_update= MAPGMinibatchUpdate()
    mini_batch_update.config.optimizer=None
    mini_batch_update.on_training_utility_fns(trainer=mock_trainer)

    assert mock_trainer.store.optimizer!=None
    assert type(mock_trainer.store.optimizer)==optax.GradientTransformation
    
    for net_key in mock_trainer.store.networks["networks"].keys():
        assert mock_trainer.store.opt_states[net_key]!=None
 
    assert mock_trainer.store.minibatch_update_fn


def test_on_training_utility_fns(mock_trainer:MockTrainer, mock_optimizer: MockOptimizer)->None:
    mini_batch_update= MAPGMinibatchUpdate()
    mini_batch_update.config.optimizer=mock_optimizer
    mini_batch_update.on_training_utility_fns(trainer=mock_trainer)

    assert mock_trainer.store.optimizer.test=="Done"

    for net_key in mock_trainer.store.networks["networks"].keys():
        assert mock_trainer.store.opt_states[net_key]== str(
            mock_trainer.store.networks["networks"][net_key].params
        )
 
    assert mock_trainer.store.minibatch_update_fn


def test_minibatch_update_fn(fake_state: Any)->None:
    state=fake_state[0]
    mock_trainer=fake_state[1]
    carry=[state["params"], state["opt_states"]]

    (state["params"], state["opt_state"]), metrics=mock_trainer.store.minibatch_update_fn(
        carry=carry, minibatch=state["batch"]
    )


def test_on_training_utility_fns_epoch(mock_trainer:MockTrainer)->None:
    mini_epoch_update= MAPGEpochUpdate()
    mini_epoch_update.on_training_utility_fns(trainer=mock_trainer)

    assert mock_trainer.store.num_epochs==mini_epoch_update.config.num_epochs
    assert mock_trainer.store.num_minibatches== mini_epoch_update.config.num_minibatches
    assert mock_trainer.store.epoch_update_fn


def test_epoch_update_fn(fake_state:Any)->None:
    mock_trainer= fake_state[1]
    mini_epoch_update= MAPGEpochUpdate()
    mini_epoch_update.on_training_utility_fns(trainer=mock_trainer)
    state= fake_state[0]

    new_key, subkey = jax.random.split(state["random_key"])
    permutation = jax.random.permutation(subkey, mock_trainer.store.full_batch_size)

    batch=Batch(
        observations={
                "agent_0": jnp.array(
                    [
                        [0.1, 0.5, 0.7],
                        [1.1, 1.5, 1.7]
                    ]),
                "agent_1": jnp.array(
                    [
                        [0.8, 0.3, 0.7],
                        [1.8, 1.3, 1.7]
                    ]),
                "agent_2": jnp.array(
                    [
                        [0.9, 0.9, 0.8],
                        [1.9, 1.9, 1.8]
                    ])
            },
        actions={
                "agent_0": jnp.array(
                    [
                        [0.2, 0.6, 0.8],
                        [1.2, 1.6, 1.8]
                    ]),
                "agent_1": jnp.array(
                    [
                        [0.9, 0.4, 0.8],
                        [1.9, 1.4, 1.8]
                    ]),
                "agent_2": jnp.array(
                    [
                        [0.8, 0.8, 0.8],
                        [1.8, 1.8, 1.8]
                    ])
            },
        advantages= jnp.array([2.1, 2.5, 2.7]),
        target_values= jnp.array([3.1, 3.5, 3.7]),
        behavior_values= jnp.array([4.1, 4.5, 4.7]),
        behavior_log_probs= jnp.array([5.1, 5.5, 5.7]))
    shuffled_batch = jax.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
    )
    
    carry= [state["random_key"], state["params"], state["opt_states"], batch]
    mock_trainer.store.epoch_update_fn(carry=carry, unused_t=None)    