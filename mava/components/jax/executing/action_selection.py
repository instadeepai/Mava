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
# For precommit
import abc
from dataclasses import dataclass

import jax
from acme.jax import utils

from mava.components.jax import Component
from mava.core_jax import SystemExecutor


@dataclass
class ExecutorSelectActionConfig:
    pass


class ExecutorSelectAction(Component):
    @abc.abstractmethod
    def __init__(
        self,
        config: ExecutorSelectActionConfig = ExecutorSelectActionConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    # Select actions
    @abc.abstractmethod
    def on_execution_select_actions(self, executor: SystemExecutor) -> None:
        """Summary"""
        pass

    # Select action
    @abc.abstractmethod
    def on_execution_select_action_compute(self, executor: SystemExecutor) -> None:
        """Summary"""
        pass

    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "executor_select_action"


class FeedforwardExecutorSelectAction(ExecutorSelectAction):
    def __init__(
        self,
        config: ExecutorSelectActionConfig = ExecutorSelectActionConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    # Select actions
    def on_execution_select_actions(self, executor: SystemExecutor) -> None:
        """Summary"""
        executor.store.actions_info = {}
        executor.store.policies_info = {}
        for agent, observation in executor.store.observations.items():
            action_info, policy_info = executor.select_action(agent, observation)
            executor.store.actions_info[agent] = action_info
            executor.store.policies_info[agent] = policy_info

    # Select action
    def on_execution_select_action_compute(self, executor: SystemExecutor) -> None:
        """Summary"""

        agent = executor.store.agent
        network = executor.store.networks["networks"][
            executor.store.agent_net_keys[agent]
        ]

        observation = utils.add_batch_dim(executor.store.observation.observation)
        rng_key, executor.store.key = jax.random.split(executor.store.key)

        # TODO (dries): We are currently using jit in the networks per agent.
        # We can also try jit over all the agents in a for loop. This would
        # allow the jit function to save us even more time.
        executor.store.action_info, executor.store.policy_info = network.get_action(
            observation,
            rng_key,
            utils.add_batch_dim(executor.store.observation.legal_actions),
        )


class FeedforwardExecutorSelectActionValueBased(ExecutorSelectAction):
    """Feedforward executor that selects actions based on the q-values.

    TODO: this class has method on_execution_select_actions which identical to the same
     method in FeedforwardExecutorSelectAction. They can be children of a parent class.
    """

    def __init__(
        self,
        config: ExecutorSelectActionConfig = ExecutorSelectActionConfig(),
    ) -> None:
        """_summary_

        Args:
            config : a config is passed, works with an empty config as well.
        """
        self.config = config

    # Select actions
    def on_execution_select_actions(self, executor: SystemExecutor) -> None:
        """Select actions for all the agents."""
        executor.store.actions_info = {}
        executor.store.policies_info = {}
        for agent, observation in executor.store.observations.items():
            action_info, policy_info = executor.select_action(agent, observation)
            executor.store.actions_info[agent] = action_info
            executor.store.policies_info[agent] = policy_info

    # Select action
    def on_execution_select_action_compute(self, executor: SystemExecutor) -> None:
        """Compute the action for the agent."""

        agent = executor.store.agent
        network = executor.store.networks["networks"][
            executor.store.agent_net_keys[agent]
        ]

        # observation = executor.store.observation.observation.reshape((1, -1))
        observation = utils.add_batch_dim(executor.store.observation.observation)
        rng_key, executor.store.key = jax.random.split(executor.store.key)

        # TODO (dries): We are currently using jit in the networks per agent.
        # We can also try jit over all the agents in a for loop. This would
        # allow the jit function to save us even more time.
        epsilon = executor.store.epsilon_scheduler.epsilon

        executor.store.action_info, executor.store.policy_info = network.get_action(
            observation,
            rng_key,
            epsilon=epsilon,
            mask=utils.add_batch_dim(executor.store.observation.legal_actions),
        )

        """
         executor.store.action_info, executor.store.policy_info = network.get_action(
            observation,
            rng_key,
            utils.add_batch_dim(executor.store.observation.legal_actions),
        )
        """
