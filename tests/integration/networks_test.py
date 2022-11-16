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

"""Network initialisation integration test"""

import functools
from typing import Callable

import jax
import pytest

from mava import specs
from mava.systems.ippo.networks import make_default_networks
from mava.utils.environments import debugging_utils


@pytest.fixture
def recurrent_network() -> Callable:
    """Recurrent network fixture"""

    # Environment.
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name="simple_spread",
        action_space="discrete",
    )

    env, _ = environment_factory()
    ma_environment_spec = specs.MAEnvironmentSpec(env)

    agent_net_keys = {
        "agent_0": "network_agent",
        "agent_1": "network_agent",
        "agent_2": "network_agent",
    }

    _, network_key = jax.random.split(jax.random.PRNGKey(42))

    return lambda: make_default_networks(
        environment_spec=ma_environment_spec,
        agent_net_keys=agent_net_keys,
        base_key=network_key,
        policy_recurrent_layer_sizes=(64,),
        policy_layers_after_recurrent=(64,),
        activation_function=jax.nn.tanh,
        orthogonal_initialisation=True,
    )


@pytest.fixture
def feedforward_network() -> Callable:
    """Feeforward network fixture"""

    # Environment.
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name="simple_spread",
        action_space="discrete",
    )

    env, _ = environment_factory()
    ma_environment_spec = specs.MAEnvironmentSpec(env)

    agent_net_keys = {
        "agent_0": "network_agent",
        "agent_1": "network_agent",
        "agent_2": "network_agent",
    }

    _, network_key = jax.random.split(jax.random.PRNGKey(42))

    return lambda: make_default_networks(
        environment_spec=ma_environment_spec,
        agent_net_keys=agent_net_keys,
        base_key=network_key,
        orthogonal_initialisation=True,
    )


def test_recurrent_network(recurrent_network: Callable) -> None:
    """Recurrent network smoke test"""

    assert recurrent_network()


def test_feedforward_network(feedforward_network: Callable) -> None:
    """Feedforward network smoke test"""

    assert feedforward_network()
