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

"""Execution components for system builders"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Type

import jax
from acme.jax import networks as networks_lib
from acme.jax import utils

from mava.components.executing.action_selection import ExecutorSelectAction
from mava.core_jax import SystemExecutor
from mava.types import NestedArray

@dataclass
class FeedforwardExecutorSelectActionConfig:
    epsilon_min: float = 0.05
    epsilon_decay_timesteps: float = 10_000

# TODO (sasha): rename to IDQN....
class FeedforwardExecutorSelectAction(ExecutorSelectAction):
    def __init__(
        self,
        config: FeedforwardExecutorSelectActionConfig = FeedforwardExecutorSelectActionConfig(),
    ):
        """Component defines hooks for the executor selecting actions.

        Args:
            config: SimpleNamespace.
        """
        self.config = config

    def on_execution_init_end(self, executor: SystemExecutor) -> None:
        """Create function that is used to select actions.

        Args:
            executor : SystemExecutor.

        Returns:
            None.
        """
        # Epsilon Scheduling
        executor.store.num_action_selections = 0.


        networks = executor.store.networks
        agent_net_keys = executor.store.agent_net_keys

        def select_action(
            observation: NestedArray,
            current_params: NestedArray,
            network: Any,
            base_key: networks_lib.PRNGKey,
            epsilon: float
        ) -> Tuple[NestedArray, NestedArray, networks_lib.PRNGKey]:
            """Action selection across a single agent.

            Args:
                observation : The observation for the current agent.
                current_params : The parameters for current agent's network.
                network : The network object used by the current agent.
                key : A JAX prng key.

            Returns:
                action info, policy info and new key.
            """
            observation_data = utils.add_batch_dim(observation.observation)

            # We use the action_key immediately and keep the new key for future splits.
            base_key, action_key = jax.random.split(base_key)
            action_info = network.get_action(
                observations=observation_data,
                params=current_params,
                epsilon=epsilon,
                base_key=action_key,
                mask=utils.add_batch_dim(observation.legal_actions),
            )
            return jax.numpy.squeeze(action_info), base_key

        def select_actions(
            observations: Dict[str, NestedArray],
            current_params: Dict[str, NestedArray],
            base_key: networks_lib.PRNGKey,
            epsilon: float
        ) -> Tuple[
            Dict[str, NestedArray], Dict[str, NestedArray], networks_lib.PRNGKey
        ]:
            """Select actions across all agents - this is jitted below.

            Args:
                observations : The observations for all the agents.
                current_params : The parameters for all the agents.
                base_key : A JAX prng_key.

            Returns:
                action info, policy info and new prng key.
            """
            actions_info = {}
            # TODO Look at tree mapping this forloop.
            # Since this is jitted, compiling a forloop with lots of agents could take
            # long, we should vectorize this.
            for agent, observation in observations.items():
                network = networks[agent_net_keys[agent]]
                actions_info[agent], base_key = select_action(
                    observation=observation,
                    # TODO (sasha): why does the PPO system not need to index 
                    #  ["policy_network"] into this dict?
                    current_params=current_params[agent_net_keys[agent]]["policy_network"],
                    network=network,
                    base_key=base_key,
                    epsilon=epsilon
                )
            return actions_info, base_key

        # executor.store.select_actions_fn = jax.jit(select_actions)
        executor.store.select_actions_fn = select_actions


    # Select actions
    def on_execution_select_actions(self, executor: SystemExecutor) -> None:
        """Select actions for each agent and save info in store.

        Args:
            executor: SystemExecutor.

        Returns:
            None.
        """

        # Dict with params per network
        current_agent_params = {
            network: executor.store.networks[network].get_params()
            for network in executor.store.agent_net_keys.values()
        }

        # Epsilon Scheduling
        # TODO add epsilon to logs
        executor.store.num_action_selections += 1.
        epsilon = max(self.config.epsilon_min, 1. - (1. / self.config.epsilon_decay_timesteps) * executor.store.num_action_selections)
        if executor._evaluator:
            epsilon = 0.
        
        executor.store.episode_metrics["epsilon"] = epsilon

        # TODO (sasha): the base executor requires this to be set, 
        #  so maybe set it once and forget or refactor the base executor?
        executor.store.policies_info = None  
        (
            executor.store.actions_info,
            executor.store.base_key,
        ) = executor.store.select_actions_fn(
            executor.store.observations, current_agent_params, executor.store.base_key, epsilon
        )