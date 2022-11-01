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

from mava.core_jax import SystemExecutor


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
        executor.store.executor_parameter_client.set_and_wait(
            {"best_checkpoint": params}
        )
    return best_performance
