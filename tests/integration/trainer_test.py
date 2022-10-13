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

"""Integration test of the Trainer for Jax-based Mava"""

import jax.numpy as jnp
import pytest

from mava import constants
from mava.systems import System
from tests.systems.systems_test_data import (
    ippo_system_single_process,
    ippo_system_single_process_norm,
)


@pytest.fixture
def test_system_sp() -> System:
    """A single process built system"""
    return ippo_system_single_process()


@pytest.fixture
def test_system_sp_norm() -> System:
    """A single process built system with feature and value normlisation set to true"""
    return ippo_system_single_process_norm()


def test_trainer_single_process(test_system_sp: System) -> None:
    """Test if the trainer instantiates processes as expected."""
    # extract the nodes
    (
        data_server,
        parameter_server,
        executor,
        evaluator,
        trainer,
    ) = test_system_sp._builder.store.system_build

    executor.run_episode()

    # Before run step method
    for net_key in trainer.store.networks.keys():
        # Policy
        mu = trainer.store.policy_opt_states[net_key]["opt_state"][1].mu  # network
        for categorical_value_head in mu.values():
            assert jnp.all(categorical_value_head["b"] == 0)
            assert jnp.all(categorical_value_head["w"] == 0)

        # Critic
        mu = trainer.store.critic_opt_states[net_key]["opt_state"][1].mu  # network
        for categorical_value_head in mu.values():
            assert jnp.all(categorical_value_head["b"] == 0)
            assert jnp.all(categorical_value_head["w"] == 0)

    trainer.step()

    # After run step method
    for net_key in trainer.store.networks.keys():
        # Policy
        mu = trainer.store.policy_opt_states[net_key]["opt_state"][1].mu  # network
        for categorical_value_head in mu.values():
            assert not jnp.all(categorical_value_head["b"] == 0)
            assert not jnp.all(categorical_value_head["w"] == 0)

        # Critic
        mu = trainer.store.critic_opt_states[net_key]["opt_state"][1].mu  # network
        for categorical_value_head in mu.values():
            assert not jnp.all(categorical_value_head["b"] == 0)
            assert not jnp.all(categorical_value_head["w"] == 0)


def test_trainer_single_process_norm(test_system_sp_norm: System) -> None:
    """Test if the trainer instantiates processes as expected."""
    # extract the nodes
    (
        data_server,
        parameter_server,
        executor,
        evaluator,
        trainer,
    ) = test_system_sp_norm._builder.store.system_build

    executor.run_episode()

    # Before run step method
    for net_key in trainer.store.networks.keys():
        # Policy
        mu = trainer.store.policy_opt_states[net_key]["opt_state"][1].mu  # network
        for categorical_value_head in mu.values():
            assert jnp.all(categorical_value_head["b"] == 0)
            assert jnp.all(categorical_value_head["w"] == 0)

        # Critic
        mu = trainer.store.critic_opt_states[net_key]["opt_state"][1].mu  # network
        for categorical_value_head in mu.values():
            assert jnp.all(categorical_value_head["b"] == 0)
            assert jnp.all(categorical_value_head["w"] == 0)

    # Check normalisation parameters
    for agent in trainer.store.trainer_agent_net_keys.keys():
        obs_norm_params = trainer.store.norm_params[constants.OBS_NORM_STATE_DICT_KEY][
            agent
        ]
        assert jnp.all(obs_norm_params["mean"] == 0)
        assert jnp.all(obs_norm_params["var"] == 0)
        assert jnp.all(obs_norm_params["std"] == 1)
        assert jnp.all(obs_norm_params["count"] == 1e-4)

        target_stats = trainer.store.norm_params[constants.VALUES_NORM_STATE_DICT_KEY][
            agent
        ]
        assert jnp.all(target_stats["mean"] == 0)
        assert jnp.all(target_stats["var"] == 0)
        assert jnp.all(target_stats["std"] == 1)
        assert jnp.all(target_stats["count"] == 1e-4)

    trainer.step()

    # After run step method
    for net_key in trainer.store.networks.keys():
        # Policy
        mu = trainer.store.policy_opt_states[net_key]["opt_state"][1].mu  # network
        for categorical_value_head in mu.values():
            assert not jnp.all(categorical_value_head["b"] == 0)
            assert not jnp.all(categorical_value_head["w"] == 0)

        # Critic
        mu = trainer.store.critic_opt_states[net_key]["opt_state"][1].mu  # network
        for categorical_value_head in mu.values():
            assert not jnp.all(categorical_value_head["b"] == 0)
            assert not jnp.all(categorical_value_head["w"] == 0)

    for agent in trainer.store.trainer_agent_net_keys.keys():
        obs_norm_params = trainer.store.norm_params[constants.OBS_NORM_STATE_DICT_KEY][
            agent
        ]
        assert not jnp.all(obs_norm_params["mean"] == 0)
        assert not jnp.all(obs_norm_params["var"] == 0)
        assert not jnp.all(obs_norm_params["std"] == 1)
        assert not jnp.all(obs_norm_params["count"] == 1e-4)

        target_stats = trainer.store.norm_params[constants.VALUES_NORM_STATE_DICT_KEY][
            agent
        ]
        assert not jnp.all(target_stats["mean"] == 0)
        assert not jnp.all(target_stats["var"] == 0)
        assert not jnp.all(target_stats["std"] == 1)
        assert not jnp.all(target_stats["count"] == 1e-4)
