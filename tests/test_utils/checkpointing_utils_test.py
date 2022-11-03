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

"""Checkpointer util functions unit test"""
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import pytest

from mava.utils.checkpointing_utils import update_best_checkpoint, update_to_best_net


def fake_networks(k: int = 0) -> Tuple:
    """Generate fake networks

    Args:
        k: value to randomize the networks
    """
    policy_param: list = []
    critic_param: list = []
    policy_opt_state_val: list = []
    critic_opt_state_val: list = []
    for i in range(0, 3):
        policy_per_agent: Dict[str, list] = {"w": []}
        critic_per_agent: Dict[str, list] = {"w": []}
        for j in range(0, 4):
            policy_per_agent["w"].append(j + 3 * i + k)
            critic_per_agent["w"].append(2 * j + i + k)
        policy_param.append(policy_per_agent)
        critic_param.append(critic_per_agent)
        policy_opt_state_val.append(i + k)
        critic_opt_state_val.append(i + k + 1)
    networks = {
        "agent_0": SimpleNamespace(
            policy_params=policy_param[0], critic_params=critic_param[0]
        ),
        "agent_1": SimpleNamespace(
            policy_params=policy_param[1], critic_params=critic_param[1]
        ),
        "agent_2": SimpleNamespace(
            policy_params=policy_param[2], critic_params=critic_param[2]
        ),
    }
    policy_opt_states = {
        "agent_0": policy_opt_state_val[0],
        "agent_1": policy_opt_state_val[1],
        "agent_2": policy_opt_state_val[2],
    }
    critic_opt_states = {
        "agent_0": critic_opt_state_val[0],
        "agent_1": critic_opt_state_val[1],
        "agent_2": critic_opt_state_val[2],
    }

    return (networks, policy_opt_states, critic_opt_states)


class MockParameterClient:
    """Mock for the parameter client"""

    def __init__(
        self,
    ) -> None:
        """Initialization"""
        self.store = SimpleNamespace(
            parameters={},
            metrics_checkpoint=["win_rate", "mean_return"],
            agents_net_keys=["agent_0", "agent_1", "agent_2"],
        )
        (networks, policy_opt_states, critic_opt_states) = fake_networks()
        self.store.parameters["best_checkpoint"] = {}
        for metric in self.store.metrics_checkpoint:
            self.store.parameters["best_checkpoint"][metric] = {}
            self.store.parameters["best_checkpoint"][metric]["best_performance"] = 20
            for agent_net_key in networks.keys():
                self.store.parameters["best_checkpoint"][metric][
                    f"policy_network-{agent_net_key}"
                ] = networks[agent_net_key].policy_params
                self.store.parameters["best_checkpoint"][metric][
                    f"critic_network-{agent_net_key}"
                ] = networks[agent_net_key].critic_params
                self.store.parameters["best_checkpoint"][metric][
                    f"policy_opt_state-{agent_net_key}"
                ] = policy_opt_states[agent_net_key]
                self.store.parameters["best_checkpoint"][metric][
                    f"critic_opt_state-{agent_net_key}"
                ] = critic_opt_states[agent_net_key]

    def set_async(self, params: Dict[Any, Any] = {}) -> None:
        """Set and wait function to update the params"""
        names = params.keys()
        for var_key in names:
            if type(self.store.parameters[var_key]) != dict:
                self.store.parameters[var_key] = params[var_key]
            else:
                for name in params[var_key].keys():
                    self.store.parameters[var_key][name] = params[var_key][name]


class MockExecutor:
    """Mock for the executor"""

    def __init__(self) -> None:
        """Initialization"""
        (networks, policy_opt_states, critic_opt_states) = fake_networks(k=2)
        executor_parameter_client = MockParameterClient()
        self.store = SimpleNamespace(
            networks=networks,
            executor_parameter_client=executor_parameter_client,
            policy_opt_states=policy_opt_states,
            critic_opt_states=critic_opt_states,
        )


class MockParameterServer(MockParameterClient):
    """Mock for the parameter server"""

    def __init__(self) -> None:
        """Initialization"""
        super().__init__()
        (networks, policy_opt_states, critic_opt_states) = fake_networks(k=4)
        for agent_net_key in self.store.agents_net_keys:
            self.store.parameters[f"policy_network-{agent_net_key}"] = networks[
                agent_net_key
            ].policy_params
            self.store.parameters[f"critic_network-{agent_net_key}"] = networks[
                agent_net_key
            ].critic_params
            self.store.parameters[
                f"policy_opt_state-{agent_net_key}"
            ] = policy_opt_states[agent_net_key]
            self.store.parameters[
                f"critic_opt_state-{agent_net_key}"
            ] = critic_opt_states[agent_net_key]


@pytest.fixture
def mock_executor() -> MockExecutor:
    """Create a mock for the executor"""
    return MockExecutor()


@pytest.fixture
def mock_parameter_server() -> MockParameterServer:
    """Create a mock for the parameter server"""
    return MockParameterServer()


def test_update_best_checkpoint(mock_executor: MockExecutor) -> None:
    """Test update_best_checkpoint function"""

    results = {
        "win_rate": 70,
        "mean_return": 18,
        "max_return": 20,
    }
    best_performance = update_best_checkpoint(
        executor=mock_executor, results=results, metric="win_rate"  # type:ignore
    )
    mock_parameter_client = mock_executor.store.executor_parameter_client

    assert best_performance == 70
    assert (
        mock_parameter_client.store.parameters["best_checkpoint"]["win_rate"][
            "best_performance"
        ]
        == 70
    )
    assert (
        mock_parameter_client.store.parameters["best_checkpoint"]["mean_return"][
            "best_performance"
        ]
        != 70
    )
    # Check that the best checkpoint params are updated for the win_rate
    for agent_net_key in mock_executor.store.networks.keys():
        assert (
            mock_parameter_client.store.parameters["best_checkpoint"]["win_rate"][
                f"policy_network-{agent_net_key}"
            ]
            == mock_executor.store.networks[agent_net_key].policy_params
        )
        assert (
            mock_parameter_client.store.parameters["best_checkpoint"]["win_rate"][
                f"critic_network-{agent_net_key}"
            ]
            == mock_executor.store.networks[agent_net_key].critic_params
        )
        assert (
            mock_parameter_client.store.parameters["best_checkpoint"]["win_rate"][
                f"policy_opt_state-{agent_net_key}"
            ]
            == mock_executor.store.policy_opt_states[agent_net_key]
        )
        assert (
            mock_parameter_client.store.parameters["best_checkpoint"]["win_rate"][
                f"critic_opt_state-{agent_net_key}"
            ]
            == mock_executor.store.critic_opt_states[agent_net_key]
        )

    # Check that the best checkpoint params didn't get updated for the mean return
    identical = True
    for agent_net_key in mock_executor.store.networks.keys():
        if (
            (
                mock_parameter_client.store.parameters["best_checkpoint"][
                    "mean_return"
                ][f"policy_network-{agent_net_key}"]
                != mock_executor.store.networks[agent_net_key].policy_params
            )
            or (
                mock_parameter_client.store.parameters["best_checkpoint"][
                    "mean_return"
                ][f"critic_network-{agent_net_key}"]
                != mock_executor.store.networks[agent_net_key].critic_params
            )
            or (
                mock_parameter_client.store.parameters["best_checkpoint"][
                    "mean_return"
                ][f"policy_opt_state-{agent_net_key}"]
                != mock_executor.store.policy_opt_states[agent_net_key]
            )
            or (
                mock_parameter_client.store.parameters["best_checkpoint"][
                    "mean_return"
                ][f"critic_opt_state-{agent_net_key}"]
                != mock_executor.store.critic_opt_states[agent_net_key]
            )
        ):
            identical = False
            break
    assert not identical


def test_update_to_best_net(mock_parameter_server: MockParameterServer) -> None:
    """Test update_to_best_net function"""
    update_to_best_net(mock_parameter_server, "win_rate")  # type:ignore

    # Check that the networks got updated by the one belong to the win rate
    network = mock_parameter_server.store.parameters["best_checkpoint"]["win_rate"]
    for agent_net_key in mock_parameter_server.store.agents_net_keys:
        assert (
            mock_parameter_server.store.parameters[f"policy_network-{agent_net_key}"]
            == network[f"policy_network-{agent_net_key}"]
        )
        assert (
            mock_parameter_server.store.parameters[f"critic_network-{agent_net_key}"]
            == network[f"critic_network-{agent_net_key}"]
        )
        assert (
            mock_parameter_server.store.parameters[f"policy_opt_state-{agent_net_key}"]
            == network[f"policy_opt_state-{agent_net_key}"]
        )
        assert (
            mock_parameter_server.store.parameters[f"critic_opt_state-{agent_net_key}"]
            == network[f"critic_opt_state-{agent_net_key}"]
        )

    # Check the case  metric doesn't exist
    with pytest.raises(Exception):
        update_to_best_net(mock_parameter_server, "reward")  # type:ignore

    # Check the case the parameters don't have best_checkpoint
    del mock_parameter_server.store.parameters["best_checkpoint"]
    with pytest.raises(Exception):
        update_to_best_net(mock_parameter_server, "reward")  # type:ignore
