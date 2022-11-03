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

"""Utils to checkpoint the network of the best performance of an algorithm"""
from typing import Any, Dict

from mava.core_jax import SystemExecutor, SystemParameterServer


def update_best_checkpoint(
    executor: SystemExecutor, results: Dict[str, Any], metric: str
) -> float:
    """Update the best_checkpoint parameter in the server"""
    best_performance = results[metric]
    params: Dict[str, Any] = {}
    params[metric] = {}
    for agent_net_key in executor.store.networks.keys():
        params[metric][f"policy_network-{agent_net_key}"] = executor.store.networks[
            agent_net_key
        ].policy_params
        params[metric][f"critic_network-{agent_net_key}"] = executor.store.networks[
            agent_net_key
        ].critic_params
        params[metric][
            f"policy_opt_state-{agent_net_key}"
        ] = executor.store.policy_opt_states[agent_net_key]
        params[metric][
            f"critic_opt_state-{agent_net_key}"
        ] = executor.store.critic_opt_states[agent_net_key]
        executor.store.executor_parameter_client.set_async({"best_checkpoint": params})
    return best_performance


def update_to_best_net(server: SystemParameterServer, metric: str) -> None:
    """Restore the network to have the values of the network with best performance"""
    assert (
        "best_checkpoint" in server.store.parameters.keys()
    ), "Can't find the restored best network checkpointed"
    assert (
        metric in server.store.parameters["best_checkpoint"].keys()
    ), f"The metric chosen does not exist in the checkpointed networks.\
        The best checkpointed network only available for the following\
            metrics {server.store.parameters['best_checkpoint'].keys()}"
    network = server.store.parameters["best_checkpoint"][metric]
    # Update network
    for agent_net_key in server.store.agents_net_keys:
        server.store.parameters[f"policy_network-{agent_net_key}"] = network[
            f"policy_network-{agent_net_key}"
        ]
        server.store.parameters[f"critic_network-{agent_net_key}"] = network[
            f"critic_network-{agent_net_key}"
        ]
        server.store.parameters[f"policy_opt_state-{agent_net_key}"] = network[
            f"policy_opt_state-{agent_net_key}"
        ]
        server.store.parameters[f"critic_opt_state-{agent_net_key}"] = network[
            f"critic_opt_state-{agent_net_key}"
        ]
